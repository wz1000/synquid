-- | Refinement type reconstruction for programs with holes
{-# LANGUAGE PartialTypeSignatures, ScopedTypeVariables, FlexibleContexts, GADTs, TypeApplications, DataKinds, NoStarIsType,TypeOperators, DeriveAnyClass, DeriveGeneric, MultiParamTypeClasses, TypeSynonymInstances, AllowAmbiguousTypes, FlexibleInstances, BangPatterns, ViewPatterns #-}
{-# OPTIONS_GHC -Wall -Wno-name-shadowing -Wno-partial-type-signatures -O0 -freduction-depth=0 #-}
module Synquid.Train where

import Synquid.Logic
import Synquid.Type hiding (set)
import Synquid.Program
import Synquid.Explorer

import qualified Data.Map as Map
import Data.Map (Map)
import qualified Data.Foldable as F
import Control.Monad.Logic
import Data.List
import Data.Function

import qualified Torch.NN as U
import Torch.Typed hiding (DType, Device, shape, sin, any, length, replicate, round)
import qualified Torch.Functional as U
import qualified Torch.Autograd as U
import GHC.Exts (IsList(toList))
import System.Directory
import Text.Read
import System.IO
import Data.Maybe
import Data.List.Split
import GHC.TypeLits
import Data.Proxy
import System.Random.Shuffle

trainAndUpdateModel :: MonadIO s => BaseWeights -> ExampleDataset -> s BaseWeights
trainAndUpdateModel model examples' = liftIO $ do
  if (not $ null examples')
  then do
    let groups = groupBy ((==) `on` (fst . fst)) $ examples'
    let n = Prelude.floor $ 0.1*(fromIntegral $ length groups)
    (concat -> testdata,concat -> examples) <- splitAt n <$> shuffleM groups
    print ("DATA",length testdata, length examples)
    let
        maxLearningRate = 1e-2
        finalLearningRate = 1e-4
        numEpochs = 100
        numWarmupEpochs = 10
        numCooldownEpochs = 10

        -- single-cycle learning rate schedule, see for instance https://arxiv.org/abs/1803.09820
        learningRateSchedule epoch
          | epoch <= 0 = 0.0
          | 0 < epoch && epoch <= numWarmupEpochs =
            let a :: Float = fromIntegral epoch / fromIntegral numWarmupEpochs
             in mulScalar a maxLearningRate
          | numWarmupEpochs < epoch && epoch < numEpochs - numCooldownEpochs =
            let a :: Float =
                  fromIntegral (numEpochs - numCooldownEpochs - epoch)
                    / fromIntegral (numEpochs - numCooldownEpochs - numWarmupEpochs)
             in mulScalar a maxLearningRate + mulScalar (1 - a) finalLearningRate
          | otherwise = finalLearningRate

        optimInit = mkAdam 0 0.9 0.999 (flattenParameters model)
        init = (model, optimInit, 0)

        step (model,optim,epoch) = do
          let learningRate = learningRateSchedule epoch
              _ = optim `asTypeOf` optimInit
          print ("Epoch",epoch)
          (model',optim') <- train model optim learningRate examples
          pure (model', optim', epoch+1)

    (model',_optim',_epochs') <- loop numEpochs step init
    let res = map (toFloat . reshape . runModel @1 model' . (:. HNil) . fst) testdata
        yact = map (toFloat . reshape . encodeBool' . snd) testdata
        re = zipWith (\x y -> (round x,round y)) res yact
        count = length $ filter (uncurry (==)) re
    print (count,length yact)
    print ("Eval",eval testdata res)
    print re
    saveParams "synquid-model" model'
    pure model'
  else pure model

eval :: [Example] -> [Float] -> (Int,Int,Int)
eval xs ys = foldl' (\(!a,!b,!x) (!c,!d,!y) -> (a+c,b+d,x+y)) (0,0,0) $ map go groups
  where
    groups = groupBy ((==) `on` (fst . fst . fst)) $ zip xs ys
    go xs = case find (snd . fst) xs of
      Just (_,val)
        | val >= maximum (map snd xs) -> (1,1,0)
        | otherwise -> (0,1,0)
      Nothing -> (0,0,1)

loop :: Monad m => Int -> (a -> m a) -> a -> m a
loop n f = foldr (<=<) pure (replicate n f)

data Untype = Untype
instance Apply' Untype (Parameter device dtype shape) (U.Parameter) where
  apply' _ = untypeParam

data EncodeBool = EncodeBool BaseWeights
instance Apply' EncodeBool Bool (Tensor Dev DT '[]) where
  apply' (EncodeBool bw) b = reshape $ encodeBool' b

data MkChunk = MkChunk
instance Apply MkChunk [a] HNothing where
  apply _ [] = HNothing
instance Apply MkChunk [a] (HJust (a,[a])) where
  apply _ (x:xs) = HJust (x,xs)

chunked :: forall batch a. (KnownNat batch, _) => [a] -> ([HList (HReplicateR batch a)],[a])
chunked xs = go chunks
  where
    size = fromInteger $ natVal $ Proxy @batch
    chunks = chunksOf size xs
    go [] = ([],[])
    go (xs:xss)
      | length xs < size = (xss',xs ++ slop)
      | otherwise = ((hunfoldr MkChunk xs :: HList (HReplicateR batch a)) : xss', slop)
      where (xss', slop) = go xss

-- | Train the model for one epoch
train ::
  _ =>
  -- | initial model datatype holding the weights
  BaseWeights ->
  -- | initial optimizer, e.g. Adam
  optim ->
  -- | learning rate, 'LearningRate device dtype' is a type alias for 'Tensor device dtype '[]'
  LearningRate Dev DT ->
  -- | stream of training examples consisting of inputs and outputs
  [((RType,RSchema), Bool)] ->
  -- | final model and optimizer
  IO (BaseWeights, optim)
train model optim learningRate examples = do
  let -- training step function
      stepOne (x,yact') (model,optim) = do
        let y' = runModel @1 model (x :. HNil)
            yact = encodeBool' yact'
            loss = useless + binaryCrossEntropy @'ReduceMean ones y' yact
            useless = mulScalar (0:: Float) ( UnsafeMkTensor $ sum $ map U.sumAll $ map U.toDependent $ toList . Just . hmap' Untype . flattenParameters $ model)
        print ("StepOne Loss",loss,y',yact)
        runStep model optim loss learningRate

      step :: forall n. (KnownNat n , _) => _
      step ex (model,optim) = do
        let y' = runModel @n model xs
            yact = stack @0 $ hmap' (EncodeBool model) yact'
            (xs,yact') = hunzip ex
            loss = useless + binaryCrossEntropy @'ReduceMean ones y' yact
            useless = mulScalar (0:: Float) ( UnsafeMkTensor $ sum $ map U.sumAll $ map U.toDependent $ toList . Just . hmap' Untype . flattenParameters $ model)
        print ("Loss",loss,y',yact)
        runStep model optim loss learningRate

      (chunks,slop) = chunked @256 examples
  -- training is a fold over the 'examples' stream
  (model', optim') <- F.foldrM (step @256) (model,optim) chunks
  let (chunks',slop') = chunked @32 slop
  (model'',optim'') <- F.foldrM (step @32) (model',optim') chunks'
  F.foldrM stepOne (model'',optim'') slop'

type ExampleDataset = [Example]
type Example = ((RType,RSchema),Bool)

readData :: FilePath -> IO ExampleDataset
readData fp = do
  exists <- doesFileExist fp
  if exists then mapMaybe readMaybe . lines <$> readFile fp
  else pure []

writeData :: FilePath -> [Example] -> IO ()
writeData file dat = withFile file AppendMode $ \h -> mapM_ (hPutStrLn h . show) dat

collectData :: RProgram -> Decisions -> [Example]
collectData prog decisions = go prog
  where
    go (Program p _) = case p of
      PSymbol' (Just prov) id ->
        [ ((target,cand),symbol == id) | let (target,xs) = decisions Map.! prov, (symbol,cand) <- xs ]
      PSymbol' Nothing _ -> []
      PApp a b -> go a ++ go b
      PFun _ b -> go b
      PIf a b c -> go a ++ go b ++ go c
      PMatch a xs -> go a ++ concatMap (go . expr) xs
      PFix _ a -> go a
      PLet _ a b -> go a ++ go b
      PHole -> []
      PErr -> []
