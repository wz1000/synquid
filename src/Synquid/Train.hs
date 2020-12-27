{-# LANGUAGE PartialTypeSignatures, ScopedTypeVariables, FlexibleContexts, GADTs, TypeApplications, DataKinds, NoStarIsType,TypeOperators, DeriveAnyClass, DeriveGeneric, MultiParamTypeClasses, TypeSynonymInstances, AllowAmbiguousTypes, FlexibleInstances, BangPatterns, ViewPatterns, StandaloneDeriving, RankNTypes #-}
{-# OPTIONS_GHC -Wall -Wno-name-shadowing -O0 #-}
module Synquid.Train (train,Example(..)) where

import Synquid.Logic
import Synquid.Util
import Synquid.TypeConstraintSolver
import Synquid.Type hiding (set)

import qualified Torch.NN as U
import Torch.Typed hiding (DType, Device, shape, sin, any, length, replicate, round, div)
import qualified Torch.Functional as U
import qualified Torch.Autograd as U
import GHC.Exts (IsList(toList))
import Data.Vector.Sized (Vector)
import qualified Data.Vector.Sized as V
import qualified Data.Vector as UV
import GHC.TypeLits
import System.Mem
import Control.Parallel.Strategies
import Data.List.Split
import Control.DeepSeq
import Data.List
import Control.Monad.Reader
import qualified Data.Map as Map
import Data.Map (Map)

data SumAll = SumAll
instance Apply' SumAll ((Tensor Dev DT shape),(Tensor Dev DT '[])) (Tensor Dev DT '[]) where
  apply' _ (this,acc) = acc + (sumAll this)

data Add = Add
instance Apply' Add ((Tensor Dev DT shape),(Tensor Dev DT shape)) (Tensor Dev DT shape) where
  apply' _ (a,b) = a + b

instance NFData (HList '[]) where
  rnf x = x `seq` ()
instance (NFData x, NFData (HList xs)) => NFData (HList (x:xs)) where
  rnf (x :. xs) = rnf x `seq` rnf xs

data Example
  = Example
  { target :: RType
  , cands :: [(RSchema, Bool)]
  , boundvars :: [Id]
  , typeSubst :: TypeSubstitution
  , quals :: QMap
  } deriving (Eq,Read,Show)

-- | Train the model for one epoch
train ::
  forall optim.
  (_) =>
  -- | initial model datatype holding the weights
  BaseWeights ->
  -- | initial optimizer, e.g. Adam
  optim ->
  -- | learning rate, 'LearningRate device dtype' is a type alias for 'Tensor device dtype '[]'
  LearningRate Dev DT ->
  -- | stream of training examples consisting of inputs and outputs
  [Example] ->
  -- | final BaseWeights and optimizer
  IO (BaseWeights, optim)
train model optim learningRate examples' = do
  performGC
  let getLoss (V.SomeSized exs) = loss
        where
          (xs,yact') = V.unzip exs
          y' = runModel model xs
          yact = vecStack @0 $ fmap (reshape . encodeBool') yact'
          loss = useless + binaryCrossEntropy @'ReduceMean (ones + mulScalar (7 :: Float) yact) y' yact

      examples :: [((EncodeEnv,Encoding,RSchema),Bool)]
      examples = do
        Example spec cands bound subst qmap <- examples'
        let env = EncodeEnv (reverse bound) (substMap <> qualMap) mempty model
            substMap = fmap (flip runReader (EncodeEnv (reverse bound) mempty mempty model) . encode) subst
            qualMap = fmap (flip runReader (EncodeEnv (reverse bound) substMap mempty model) . encode) qmap
            espec = runReader (encode spec) env
        (cand,res) <- cands
        pure ((env,espec,cand),res)

      numThreads = 8
      len = length examples `div` numThreads
      chunks' = map UV.fromList $ chunksOf len examples
      chunks
        | (x : xs@(_:_)) <- chunks'
        , let l = last xs
        , UV.length l < len
        = (x <> l) : init xs
        | otherwise = chunks'

      losses = parMap rdeepseq getLoss chunks

      grads = parMap rseq (`grad` parameters) losses

      useless = mulScalar (0:: Float) (hfoldr SumAll (zeros :: Tensor Dev DT '[]) tensors)

      parameters = flattenParameters model
      gradients = foldl1' (hzipWith Add) grads
      tensors = hmap' ToDependent parameters

      (tensors', optim') = step learningRate gradients tensors optim

  let average xs = sum (map toFloat xs) / (fromIntegral $ length xs)
  print ("Loss",average losses, losses,map UV.length chunks)
  parameters' <- hmapM' MakeIndependent tensors'
  let model' = replaceParameters model parameters'
  pure (model', optim')

