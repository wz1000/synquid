{-# LANGUAGE PartialTypeSignatures, ScopedTypeVariables, FlexibleContexts, GADTs, TypeApplications, DataKinds, NoStarIsType,TypeOperators, DeriveAnyClass, DeriveGeneric, MultiParamTypeClasses, TypeSynonymInstances, AllowAmbiguousTypes, FlexibleInstances, BangPatterns, ViewPatterns, StandaloneDeriving #-}
{-# OPTIONS_GHC -Wall -Wno-name-shadowing -O0 #-}
module Synquid.Learn where

import Synquid.Logic
import Synquid.Type hiding (set)
import Synquid.Program
import Synquid.Explorer
import Synquid.Train (train)

import qualified Data.Map as Map
import Control.Monad.Logic
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

upsample :: [(a,Bool)] -> [(a,Bool)]
upsample xs = case find snd xs of
  Just x -> replicate l x ++ xs
  where
    l = length xs - 2

trainAndUpdateModel :: MonadIO s => BaseWeights -> ExampleDataset -> s BaseWeights
trainAndUpdateModel model examples' = liftIO $ do
  if (not $ null examples')
  then do
    let groups = groupBy ((==) `on` (fst . fst)) $ examples'
    let n = Prelude.floor $ 0.1*(fromIntegral $ length groups)
    (concat -> testdata,examples') <- splitAt n <$> shuffleM groups
    writeData "traindataset" (concat examples')
    writeData "testdataset" testdata
    let examples = map (\xs -> (fst . fst . head $ xs, map (\((_,b),c) -> (b,c)) xs)) examples'
    print ("DATA",length testdata, length $ concat examples')
    let
        maxLearningRate = 1e-2
        finalLearningRate = 1e-4
        numEpochs = 120
        numWarmupEpochs = 20
        numCooldownEpochs = 20

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
    let res = map (toFloat . reshape . runModel @1 model' . V.singleton . fst) testdata
        yact = map (toFloat . reshape . encodeBool' . snd) testdata
        re = zipWith (\x y -> (round x,round y)) res yact
        count = length $ filter (uncurry (==)) re
    print (count,length yact)
    print ("Eval",eval testdata res)
    print re
    saveParams "synquid-model" model'
    pure model
  else pure model

eval :: [Example] -> [Float] -> (Int,Int,[(Int,Int)])
eval xs ys = foldl' (\(!a,!b,!x) (!c,!d,!y) -> (a+c,b+d,maybe x (:x) y)) (0,0,[]) $ map go groups
  where
    groups = groupBy ((==) `on` (fst . fst . fst)) $ zip xs ys
    go :: [(Example, Float)] -> (Int,Int,Maybe (Int,Int))
    go xs = case find (snd . fst) xs of
      Just (_,val)
        | val >= head vals -> (1,1,Nothing)
        | otherwise -> let Just idx = elemIndex val vals in (0,1,Just (idx,length vals))
      Nothing -> (0,0,Nothing)
      where vals = reverse $ sort $ map snd xs

loop :: Monad m => Int -> (a -> m a) -> a -> m a
loop n f = foldr (<=<) pure (replicate n f)

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
