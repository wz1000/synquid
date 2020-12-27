{-# LANGUAGE PartialTypeSignatures, ScopedTypeVariables, FlexibleContexts, GADTs, TypeApplications, DataKinds, NoStarIsType,TypeOperators, DeriveAnyClass, DeriveGeneric, MultiParamTypeClasses, TypeSynonymInstances, AllowAmbiguousTypes, FlexibleInstances, BangPatterns, ViewPatterns, StandaloneDeriving #-}
{-# OPTIONS_GHC -Wall -Wno-name-shadowing -O0 #-}
module Synquid.Learn where

import Synquid.Logic
import Synquid.Type hiding (set)
import Synquid.Program
import Synquid.Explorer
import Synquid.Train (train,Example(..))
import Synquid.Util
import Synquid.TypeConstraintSolver

import qualified Data.Map as Map
import Control.Monad.Reader
import Data.List
import Data.Function

import Torch.Typed hiding (DType, Device, shape, sin, any, length, replicate, round, div)
import System.Directory
import Text.Read
import System.IO
import Data.Maybe
import System.Random.Shuffle
import System.Time.Extra
import qualified Data.Vector.Sized as V
import System.FilePath

upsample :: [(a,Bool)] -> [(a,Bool)]
upsample xs = case find snd xs of
  Just x -> replicate l x ++ xs
  where
    l = length xs - 2

splitDataset :: MonadIO m => FilePath -> ExampleDataset -> m ()
splitDataset fp examples = liftIO $ do
  let n = Prelude.floor $ 0.1*(fromIntegral $ length examples)
  (testdata,examples') <- splitAt n <$> shuffleM examples
  writeData (fp <.> "train") examples'
  writeData (fp <.> "test") testdata

trainAndUpdateModel :: MonadIO s => BaseWeights -> ExampleDataset -> ExampleDataset -> s BaseWeights
trainAndUpdateModel model examples testdata = liftIO $ do
  if (not $ null examples)
  then do
    print ("DATA",length testdata, length $ concatMap cands testdata, length examples, length $ concatMap cands examples)
    let
        maxLearningRate = 1e-3
        finalLearningRate = 1e-5
        numEpochs = 50
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
          start <- offsetTime
          (model',optim') <- train model optim learningRate examples
          duration <- start
          print ("Took",duration)
          pure (model', optim', epoch+1)

    (model',_optim',_epochs') <- loop numEpochs step init
    let testexamples :: [((EncodeEnv,Encoding,RSchema),Bool)]
        testexamples = do
          Example spec cands bound subst _qmap <- testdata
          let env = EncodeEnv (reverse bound) substMap mempty model'
              substMap = fmap (flip runReader (emptyEncodeEnv model') . encode) subst
              espec = runReader (encode spec) env
          (cand,res) <- cands
          pure ((env,espec,cand),res)
    let res = map (toFloat . reshape . runModel @1 model' . V.singleton . fst) testexamples
        yact = map (toFloat . reshape . encodeBool' . snd) testexamples
        re = zipWith (\x y -> (round x,round y)) res yact
        count = length $ filter (uncurry (==)) re
    print (count,length yact)
    let ev@(_,_,xs) = eval testdata res
    print ("Eval",ev)
    print (avg xs)
    print re
    saveParams "synquid-model" model'
    pure model
  else pure model

avg :: [(Int,Int)] -> Double
avg xs = (sum (map (\(x,y) -> (fromIntegral x) / (fromIntegral y)) xs)) / (fromIntegral $ length xs)

eval :: [Example] -> [Float] -> (Int,Int,[(Int,Int)])
eval xs ys = foldl' (\(!a,!b,!x) (!c,!d,!y) -> (a+c,b+d,maybe x (:x) y)) (0,0,[]) $ go xs ys
  where
    go :: [Example] -> [Float] -> [(Int,Int,Maybe (Int,Int))]
    go [] [] = []
    go (ex:exs) scores' = case find (snd . fst) xs of
      Just (_,val)
        | val >= head vals -> (1,1,Nothing) : go exs rest
        | otherwise -> let Just idx = elemIndex val vals in (0,1,Just (idx,length vals)) : go exs rest
      Nothing -> (0,0,Nothing) : go exs rest
      where
        vals = reverse $ sort scores
        xs :: [((RSchema,Bool),Float)]
        xs = zip (cands ex) scores
        (scores,rest) = splitAt (length (cands ex)) scores'


loop :: Monad m => Int -> (a -> m a) -> a -> m a
loop n f = foldr (<=<) pure (replicate n f)

type ExampleDataset = [Example]

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
        [ Example target [(cand,symbol == id) | (symbol,cand) <- xs] (_boundTypeVars env) (_typeAssignment tass) (_qualifierMap tass)
        | let (target,xs,env,tass) = decisions Map.! prov ]
      PSymbol' Nothing _ -> []
      PApp a b -> go a ++ go b
      PFun _ b -> go b
      PIf a b c -> go a ++ go b ++ go c
      PMatch a xs -> go a ++ concatMap (go . expr) xs
      PFix _ a -> go a
      PLet _ a b -> go a ++ go b
      PHole -> []
      PErr -> []
