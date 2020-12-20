{-# LANGUAGE PartialTypeSignatures, ScopedTypeVariables, FlexibleContexts, GADTs, TypeApplications, DataKinds, NoStarIsType,TypeOperators, DeriveAnyClass, DeriveGeneric, MultiParamTypeClasses, TypeSynonymInstances, AllowAmbiguousTypes, FlexibleInstances, BangPatterns, ViewPatterns, StandaloneDeriving, RankNTypes #-}
{-# OPTIONS_GHC -Wall -Wno-name-shadowing -O0 #-}
module Synquid.Train (train) where

import Synquid.Logic

import qualified Torch.NN as U
import Torch.Typed hiding (DType, Device, shape, sin, any, length, replicate, round, div)
import qualified Torch.Functional as U
import qualified Torch.Autograd as U
import GHC.Exts (IsList(toList))
import Data.Vector.Sized (Vector)
import qualified Data.Vector.Sized as V
import qualified Data.Vector as UV
import GHC.TypeLits

data SumAll = SumAll
instance Apply' SumAll ((Parameter Dev DT shape),(Tensor Dev DT '[])) (Tensor Dev DT '[]) where
  apply' _ (this,acc) = acc + (sumAll $ toDependent this)

-- | Train the model for one epoch
train ::
  forall optim a b.
  (Encode a, Encode b, _) =>
  -- | initial model datatype holding the weights
  BaseWeights ->
  -- | initial optimizer, e.g. Adam
  optim ->
  -- | learning rate, 'LearningRate device dtype' is a type alias for 'Tensor device dtype '[]'
  LearningRate Dev DT ->
  -- | stream of training examples consisting of inputs and outputs
  [((a,b),Bool)] ->
  -- | final BaseWeights and optimizer
  IO (BaseWeights, optim)
train model optim learningRate (UV.fromList -> V.SomeSized examples) = do
  let y' = runModel model xs
      yact = vecStack @0 $ fmap (reshape . encodeBool') yact'
      (xs,yact') = V.unzip examples
      loss = useless + binaryCrossEntropy @'ReduceMean ones y' yact
      useless = mulScalar (0:: Float) (hfoldr SumAll (zeros :: Tensor Dev DT '[]) . flattenParameters $ model)
  print ("Loss",loss,length examples)
  runStep model optim loss learningRate

