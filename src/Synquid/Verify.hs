{-# LANGUAGE PartialTypeSignatures, ScopedTypeVariables, FlexibleContexts, GADTs, TypeApplications, DataKinds, NoStarIsType,TypeOperators, DeriveAnyClass, DeriveGeneric, MultiParamTypeClasses, TypeSynonymInstances, AllowAmbiguousTypes, FlexibleInstances, BangPatterns, ViewPatterns, StandaloneDeriving, RankNTypes #-}
{-# OPTIONS_GHC -Wall -Wno-name-shadowing #-}
module Synquid.Verify where

import Synquid.Logic

import qualified Torch.NN
import Torch.Typed hiding (DType, Device, shape, sin, any, length, replicate, round, div, transpose, linear, tanh, sigmoid, exp)
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
import Control.Monad
import Control.Monad.IO.Class

import Z3.Monad

pieces :: Int
pieces = 5

zrelu :: AST -> Z3 AST
zrelu x = do
  z <- mkRealNum 0
  cond <- mkLt x z
  mkIte cond z x

ztanh :: AST -> Z3 AST
ztanh = approximate pieces (-1) 1 Prelude.tanh

sigmoid :: Floating a => a -> a
sigmoid z = 1 / (1+exp(-z))

zsigmoid :: AST -> Z3 AST
zsigmoid = approximate pieces 0 1 sigmoid

approximate :: Int -> Float -> Float -> (Float -> Float) -> AST -> Z3 AST
approximate n lbound ubound f x = do
  lower <- mkRealNum $ head xs
  cond <- (mkLt x lower)
  bound <- mkRealNum lbound
  mkIte cond bound =<< approxRest
  where
    xs = [-5,-5+(10 / (fromIntegral n))..5]
    windows = zip xs (tail xs)
    approxRest = go windows
    go [] = mkRealNum ubound
    go ((low,hig):xs) = do
      lo <- mkRealNum low
      flo <- mkRealNum (f low)
      hi <- mkRealNum hig
      fhi <- mkRealNum (f hig)
      boundedBelow <- mkGe x lo
      boundedAbove <- mkLt x hi
      inBounds <- mkAnd [boundedBelow,boundedAbove]
      rest <- go xs
      dx <- mkSub [hi,lo]
      dy <- mkSub [fhi,flo]
      slope <- mkDiv dy dx
      delx <- mkSub [x,lo]
      dely <- mkMul [slope,delx]
      interpolate <- mkAdd [flo,dely]
      mkIte inBounds interpolate rest

linear :: [[AST]] -> [AST] -> [AST] -> Z3 [AST]
linear weights biases x = do
  xs <- mapM (mkAdd <=< zipWithM (\a b -> mkMul [a,b]) x) weights
  zipWithM (\a b -> mkAdd [a,b]) xs biases

encodingToVec :: Encoding -> Z3 [AST]
encodingToVec e = mapM mkRealNum xs
  where xs = toList $ Just e

productionToMat :: KnownNat (n*Size) => Production n -> Z3 ([[AST]],[AST])
productionToMat (Production layer) = linearToMat layer

linearToMat :: (KnownNat inp, KnownNat out) => Layer inp out -> Z3 ([[AST]],[AST])
linearToMat layer = do
  weights <- mapM (mapM mkRealNum) $ toList $ Just $ toDependent $ linearWeight layer
  biases <- mapM mkRealNum $ toList $ Just $ toDependent $ linearBias layer
  pure (weights, biases)

test :: Z3 ()
test = do
  xs <- mapM (mkRealVar <=< mkIntSymbol) [1..2]
  as <- mapM mkRealNum [1,0]
  bs <- mapM mkRealNum [0,1]
  bias <- mapM mkRealNum [0,0]
  ast <- linear [as,bs] bias xs
  mapM_ (liftIO . putStrLn <=< astToString) ast

mkAbs :: AST -> Z3 AST
mkAbs x = do
  z <- mkRealNum 0
  cond <- mkLt x z
  x' <- mkSub [z,x]
  mkIte cond x' x

commutes :: BaseWeights -> (BaseWeights -> Terminal) -> Z3 ()
commutes bw op = do
  eps <- mkRealNum (1e-3)
  xs <- mapM (mkRealVar <=< mkIntSymbol) [1..25]
  ys <- mapM (mkRealVar <=< mkIntSymbol) [26..50]
  mapM_ (liftIO . putStrLn <=< astToString) ys
  op <- encodingToVec $ toDependent (op bw)
  (weights,biases) <- productionToMat (w_bin bw)
  liftIO $ print (length weights, length $ head weights, length op)
  one' <- linear weights biases (op ++ xs ++ ys)
  one <- mapM ztanh one'
  two' <- linear weights biases (op ++ ys ++ xs)
  two <- mapM ztanh two'
  allEqs <- zipWithM (\a b -> mkSub [a,b] >>= mkAbs >>= \x -> mkLt x eps ) one two
  cond <- mkAnd allEqs
  cond' <- mkNot cond
  assert cond'
  (res, mmodel) <- getModel
  liftIO $ print res
  case mmodel of
    Nothing -> pure ()
    Just model -> do
      str <- modelToString model
      liftIO $ putStrLn str

textEqual :: BaseWeights -> Z3 ()
textEqual bw = do
  eps <- mkRealNum (1e-3)
  o <- mkRealNum 1
  mineps <- mkSub [o,eps]
  xs' <- mapM (mkRealVar <=< mkIntSymbol) [1..25]
  forM_ xs' $ \x -> do
    cond0 <- mkEq x =<< mkRealNum 0
    cond1 <- mkEq x =<< mkRealNum 1
    cond01 <- mkOr [cond0,cond1]
    assert cond01

  (sym_w,sym_b) <- productionToMat (w_sym bw)
  xs <- mapM ztanh =<< linear sym_w sym_b (xs' ++ xs')
  (var_w,var_b) <- productionToMat (w_tyvart bw)
  xs <- mapM ztanh =<< linear var_w var_b xs

  (w0,b0) <- linearToMat (layer0 bw)
  one <- mapM zrelu =<< linear w0 b0 (xs ++ xs)
  (w1,b1) <- linearToMat (layer1 bw)
  two <- mapM ztanh =<< linear w1 b1 one
  (w2,b2) <- linearToMat (layer2 bw)
  [three] <- mapM zsigmoid =<< linear w2 b2 two

  cond <- mkGt three mineps
  cond' <- mkNot cond
  assert cond'
  (res, mmodel) <- getModel
  liftIO $ print res
  case mmodel of
    Nothing -> pure ()
    Just model -> do
      str <- modelToString model
      liftIO $ putStrLn str
