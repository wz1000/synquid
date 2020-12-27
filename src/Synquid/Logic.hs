{-# LANGUAGE TemplateHaskell, TypeApplications, TypeOperators, Rank2Types, DataKinds, TypeSynonymInstances, FlexibleInstances, NoStarIsType, NamedFieldPuns, DeriveGeneric, DeriveAnyClass, MultiParamTypeClasses, ScopedTypeVariables, PatternSynonyms, AllowAmbiguousTypes, GADTs, UndecidableInstances, FlexibleContexts, PartialTypeSignatures, KindSignatures #-}


-- | Formulas of the refinement logic
module Synquid.Logic where

import Prelude hiding (tanh)

import Synquid.Util

import Data.Tuple
import Data.List
import qualified Data.Set as Set
import Data.Set (Set)
import qualified Data.Map as Map
import Data.Map (Map)
import Data.Maybe (fromJust)

import Control.Lens hiding (both)
import Control.Monad

import qualified Torch.Typed as T
import Torch.Typed hiding (length, replicate, div)
import GHC.TypeLits
import Data.Hashable
import qualified Torch.Tensor as U
import qualified Torch.NN as U
import qualified Torch.Autograd as U
import Data.Int
import GHC.Generics
import GHC.Exts
import Control.Monad.State.Strict
import Control.Monad.Reader
import Data.Functor.Identity
import Data.Proxy
import Data.Kind
import System.Directory
import GHC.Stack (HasCallStack)
import Data.Vector.Sized (Vector, withVectorUnsafe)
import Data.Vector.Strategies
import Control.DeepSeq
import System.Mem
import Control.Applicative
import Data.Foldable
import GHC.Exts as GHC

instance NFData (Tensor device dtype shape) where
  rnf = rnf . numel

type Env = ()
type Size = 25
type Dev = '(CPU,0)
type DT = 'Float
type Encoding = Tensor Dev DT '[Size]

getParams :: String -> IO BaseWeights
getParams file = do
  m <- sampleBaseWeights
  exists <- doesFileExist file
  if exists
  then do
    let x = hmap' ToDependent . flattenParameters $ m
    ts1 <- load file `asTypeOf` (pure x)
    ts <- hmapM' MakeIndependent ts1
    pure $ replaceParameters m ts
  else pure m

saveParams :: String -> BaseWeights -> IO ()
saveParams file model = do
  save (hmap' ToDependent . flattenParameters $ model) file

sampleBaseWeights :: IO BaseWeights
sampleBaseWeights = sample ()

data EncodeEnv
  = EncodeEnv
  { boundStack :: [Id] -- ^ Unknown free vars
  , boundMap :: Map Id Encoding -- ^ Known Free Vars
  , recMap :: Map Id (Int,EncodeEnv,EncodeM Encoding)
  , baseWeights :: BaseWeights -- ^ The model
  }
instance Show EncodeEnv where
  show _ = "EncodeEnv"

emptyEncodeEnv :: BaseWeights -> EncodeEnv
emptyEncodeEnv = EncodeEnv mempty mempty mempty

-- | How far to unroll definitions
uNROLL_LIMIT :: Int
uNROLL_LIMIT = 3

bindKnownVar :: Id -> Encoding -> EncodeM a -> EncodeM a
bindKnownVar id encoding x = do
  local (\e -> e {boundMap = Map.insert id encoding (boundMap e)}) x

encodeId :: Id -> EncodeM Encoding
encodeId id = do
  st <- ask
  case elemIndex id (boundStack st) of
    Nothing
      | Just e <- Map.lookup id (boundMap st) -> pure e
      | Just (i,env,encodeThis) <- Map.lookup id (recMap st) -> do
          if i < uNROLL_LIMIT
          then do
            let env' = env { recMap = Map.insert id (i+1,env',encodeThis) $ recMap st }
            local (const env') encodeThis
          else embedHelper w_rec
      | otherwise -> pure $ forward (w_sym $ baseWeights st) $ encodeInt (hash id)
    Just idx -> pure $ forward (w_sym_idx $ baseWeights st) $ oneHot id idx

oneHot :: Id -> Int -> Encoding
oneHot var i
  | i < 25 = ten
  | otherwise = error $ "oneHot out of bounds " ++ show i ++ show var
    where
      xs = replicate i 0
      ys = replicate (25-i-1) 0
      Just ten = GHC.fromList (xs ++ [1.0 :: Float] ++ ys)
  
bindId :: Id -> EncodeM a -> EncodeM a
bindId id = local (\env -> env {boundStack = id:boundStack env})

bindRecursive :: Id -> EncodeM Encoding -> EncodeM Encoding
bindRecursive id x = local (\env -> let env' = env { recMap = Map.insert id (0,env,x) (recMap env) } in env') x

type EncodeM = Reader EncodeEnv

{- Sorts -}
class Encode a where
  encode :: a -> EncodeM Encoding

instance Encode r => Encode [r] where
  encode xs = do
    nil <- encodeNil
    foldrM (\x -> encodeLCons (encode x) . pure) nil xs

instance Encode () where
  encode _ = encodeUnit

instance (Encode a, Encode b) => Encode (a,b) where
  encode (a,b) = encodePair (encode a) (encode b)

instance Encode Encoding where
  encode = pure

type Layer inp out = Linear inp out DT Dev

type Hidden (n :: Nat) = (n*Size)
-- type Hidden (n :: Nat) = If (n <=? 1) Size ((2*n*Size) `Div` 3)

-- Takes n inputs to 1 output (each of size Size)
newtype Production (n :: Nat)
  = Production
  { top :: Layer (n*Size) Size
  } deriving (Show, Generic, Parameterized)

instance (inp ~ (n*Size)) => HasForward (Production n) (Tensor Dev DT '[inp]) (Tensor Dev DT '[Size]) where
  forward Production{top} = tanh . forward top -- . tanh . forward bot
  forwardStoch m x = pure $ forward m x

data ProductionSpec (n :: Nat) = ProductionSpec

instance (KnownNat (n*Size), KnownNat (Hidden n)) => Randomizable (ProductionSpec n) (Production n) where
  sample _ = Production <$> sample LinearSpec -- <*> sample LinearSpec

type Terminal = Parameter Dev DT '[Size]

runModel :: forall batch a b. (Encode a, Encode b, KnownNat batch) => BaseWeights -> Vector batch (EncodeEnv,a,b) -> Tensor Dev DT '[batch]
runModel bw examples = reshape $ runModelTop @batch bw $ vecStack @0 $ (withVectorUnsafe $ (`using` (parVector chunkSize)) . fmap (\(st,a,b) -> cat @0 (encode' st a :. encode' st b :. HNil))) examples
  where
    chunkSize = natValI @batch `div` 8
    encode' :: forall a. Encode a => EncodeEnv -> a -> Encoding
    encode' st x = runReader (encode x) st

runModelTop :: forall batch. BaseWeights -> Tensor Dev DT '[batch,2*Size] -> Tensor Dev DT '[batch,1]
runModelTop BaseWeights{layer2,layer1,layer0}
  = sigmoid
  . forward layer2
  . tanh
  . forward layer1
  . relu
  . forward layer0

data BaseWeights = BaseWeights
  { layer2 :: Layer 16 1
  , layer1 :: Layer 64 16
  , layer0 :: Layer (2*Size) 64

-- Primitives
  , w_Unit :: Terminal
  , w_lcons :: Production 2
  , w_Nil :: Terminal
  , w_pair :: Production 2
  , w_sym :: Production 1
  , w_sym_idx:: Production 1
  , w_rec :: Terminal
  , w_false :: Terminal
  , w_true :: Terminal

-- SchemaSkeleton
  , w_monotype :: Production 1
  , w_forallt :: Production 1
  , w_forallp :: Production 3

-- TypeSkeleton
  , w_scalart :: Production 2
  , w_functiont :: Production 3
  , w_AnyT :: Terminal

-- BaseType
  , w_tyvart :: Production 1
  , w_dttype :: Production 3
  , w_IntT :: Terminal
  , w_BoolT :: Terminal

-- Sorts
  , w_vars :: Production 1
  , w_IntS :: Terminal
  , w_BoolS :: Terminal
  , w_datas :: Production 2
  , w_sets :: Production 1
  , w_AnyS :: Terminal

-- Formula
  , w_all :: Production 2
  , w_cons :: Production 3
  , w_pred :: Production 3
  , w_ite :: Production 3
  , w_bin :: Production 3
  , w_unary :: Production 2
  , w_unknown :: Production 2
  , w_var :: Production 2
  , w_setl :: Production 2
  , w_intl :: Production 1
  , w_booll :: Production 1

-- Unary
  , w_Neg :: Terminal
  , w_Not :: Terminal

-- Binary
  , w_Times :: Terminal
  , w_Plus :: Terminal
  , w_Minus :: Terminal
  , w_Eq :: Terminal
  , w_Neq :: Terminal
  , w_Lt :: Terminal
  , w_Le :: Terminal
  , w_Gt :: Terminal
  , w_Ge :: Terminal
  , w_And :: Terminal
  , w_Or :: Terminal
  , w_Implies :: Terminal
  , w_Iff :: Terminal
  , w_Union :: Terminal
  , w_Intersect :: Terminal
  , w_Diff :: Terminal
  , w_Member :: Terminal
  , w_Subset :: Terminal
  } deriving (Show, Generic, Parameterized)

embedHelper :: (BaseWeights -> Terminal) -> EncodeM Encoding
embedHelper k = asks (toDependent . k . baseWeights)

encodeUnit :: EncodeM Encoding
encodeUnit = embedHelper w_Unit

encodeNil :: EncodeM Encoding
encodeNil = embedHelper w_Nil

encodeAnyT :: EncodeM Encoding
encodeAnyT = embedHelper w_AnyT

encodeIntT :: EncodeM Encoding
encodeIntT = embedHelper w_IntT

encodeBoolT :: EncodeM Encoding
encodeBoolT = embedHelper w_BoolT

encodeIntS :: EncodeM Encoding
encodeIntS = embedHelper w_IntS

encodeBoolS :: EncodeM Encoding
encodeBoolS = embedHelper w_BoolS

encodeAnyS :: EncodeM Encoding
encodeAnyS = embedHelper w_AnyS

encodeNeg :: EncodeM Encoding
encodeNeg = embedHelper w_Neg

encodeNot :: EncodeM Encoding
encodeNot = embedHelper w_Not

encodeTimes :: EncodeM Encoding
encodeTimes = embedHelper w_Times

encodePlus :: EncodeM Encoding
encodePlus = embedHelper w_Plus

encodeMinus :: EncodeM Encoding
encodeMinus = embedHelper w_Minus

encodeEq :: EncodeM Encoding
encodeEq = embedHelper w_Eq

encodeNeq :: EncodeM Encoding
encodeNeq = embedHelper w_Neq

encodeLt :: EncodeM Encoding
encodeLt = embedHelper w_Lt

encodeLe :: EncodeM Encoding
encodeLe = embedHelper w_Le

encodeGt :: EncodeM Encoding
encodeGt = embedHelper w_Gt

encodeGe :: EncodeM Encoding
encodeGe = embedHelper w_Ge

encodeAnd :: EncodeM Encoding
encodeAnd = embedHelper w_And

encodeOr :: EncodeM Encoding
encodeOr = embedHelper w_Or

encodeImplies :: EncodeM Encoding
encodeImplies = embedHelper w_Implies

encodeIff :: EncodeM Encoding
encodeIff = embedHelper w_Iff

encodeUnion :: EncodeM Encoding
encodeUnion = embedHelper w_Union

encodeIntersect :: EncodeM Encoding
encodeIntersect = embedHelper w_Intersect

encodeDiff :: EncodeM Encoding
encodeDiff = embedHelper w_Diff

encodeMember :: EncodeM Encoding
encodeMember = embedHelper w_Member

encodeSubset :: EncodeM Encoding
encodeSubset = embedHelper w_Subset

data ParamSpec = ParamSpec
instance Randomizable ParamSpec (Parameter Dev DT '[Size]) where
  sample _ = makeIndependent =<< randn

instance Randomizable () BaseWeights where
    sample ()
      = BaseWeights
     <$> sample LinearSpec
     <*> sample LinearSpec
     <*> sample LinearSpec
     -- Primitives
     <*> sample ParamSpec
     <*> sample ProductionSpec
     <*> sample ParamSpec
     <*> sample ProductionSpec
     <*> sample ProductionSpec
     <*> sample ProductionSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     -- SchemaSkeleton
     <*> sample ProductionSpec
     <*> sample ProductionSpec
     <*> sample ProductionSpec
     -- TypeSkeleton
     <*> sample ProductionSpec
     <*> sample ProductionSpec
     <*> sample ParamSpec
     -- BaseType
     <*> sample ProductionSpec
     <*> sample ProductionSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     -- Sorts
     <*> sample ProductionSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ProductionSpec
     <*> sample ProductionSpec
     <*> sample ParamSpec
     -- Formula
     <*> sample ProductionSpec
     <*> sample ProductionSpec
     <*> sample ProductionSpec
     <*> sample ProductionSpec
     <*> sample ProductionSpec
     <*> sample ProductionSpec
     <*> sample ProductionSpec
     <*> sample ProductionSpec
     <*> sample ProductionSpec
     <*> sample ProductionSpec
     <*> sample ProductionSpec
     -- Unary
     <*> sample ParamSpec
     <*> sample ParamSpec
     -- Binary
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ParamSpec
     <*> sample ParamSpec

monadic :: (BaseWeights -> Production 1) -> EncodeM Encoding -> EncodeM Encoding
monadic k a = forward <$> asks (k . baseWeights) <*> a

dyadic :: (BaseWeights -> Production 2) -> EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding
dyadic k ma mb = do
  a <- ma
  b <- mb
  m <- asks (k . baseWeights)
  pure $ forward m $ cat @0 (a :. b :. HNil)

triadic :: (BaseWeights -> Production 3) -> EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding
triadic k ma mb mc = do
  a <- ma
  b <- mb
  c <- mc
  m <- asks (k . baseWeights)
  pure $ forward m $ cat @0 (a :. b :. c :. HNil)

-- Primitives
encodeLCons :: EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding
encodeLCons = dyadic w_lcons

encodePair :: EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding
encodePair = dyadic w_pair

encodeInt :: Int -> Encoding
encodeInt i = ten
  where
    Just ten = GHC.fromList xs
    xs = take 25 $ map (fromIntegral . (`mod` 2)) $ iterate (`div` 2) $ (i `mod` (2^25))

encodeBool :: Bool -> EncodeM Encoding
encodeBool True  = embedHelper w_false
encodeBool False = embedHelper w_true

encodeBool' :: Bool -> Tensor Dev DT '[1]
encodeBool' True  = full (1 :: Float)
encodeBool' False = full (0 :: Float)
 
-- TypeSkeleton
encodeScalarT :: EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding
encodeScalarT = dyadic w_scalart

encodeFunctionT :: EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding
encodeFunctionT = triadic w_functiont

-- BaseType
encodeTypeVarT :: EncodeM Encoding -> EncodeM Encoding
encodeTypeVarT = monadic w_tyvart

encodeDatatype :: EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding
encodeDatatype = triadic w_dttype

-- Sorts
encodeVarS :: EncodeM Encoding -> EncodeM Encoding
encodeVarS = monadic w_vars

encodeDataS :: EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding
encodeDataS = dyadic w_datas

encodeSetS :: EncodeM Encoding -> EncodeM Encoding
encodeSetS = monadic w_sets

-- Formula
encodeAll :: EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding
encodeAll = dyadic w_all

encodeCons :: EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding
encodeCons = triadic w_cons

encodePred :: EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding
encodePred = triadic w_pred

encodeIte :: EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding
encodeIte = triadic w_ite

encodeBinary :: EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding
encodeBinary = triadic w_bin

encodeUnary :: EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding
encodeUnary = dyadic w_unary

encodeUnknown :: EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding
encodeUnknown = dyadic w_unknown

encodeVar :: EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding
encodeVar = dyadic w_var

encodeSetLit :: EncodeM Encoding -> EncodeM Encoding -> EncodeM Encoding
encodeSetLit = dyadic w_setl

encodeIntLit :: EncodeM Encoding -> EncodeM Encoding
encodeIntLit = monadic w_intl

encodeBoolLit :: EncodeM Encoding -> EncodeM Encoding
encodeBoolLit = monadic w_booll

-- | Sorts
data Sort = BoolS | IntS | VarS Id | DataS Id [Sort] | SetS Sort | AnyS
  deriving (Show, Read, Eq, Ord)

instance Encode Sort where
  encode t = case t of
    BoolS -> encodeBoolS
    IntS -> encodeIntS
    VarS x -> encodeVarS (encodeId x)
    DataS id xs -> encodeDataS (encodeId id) (encode xs)
    SetS x -> encodeSetS (encode x)
    AnyS -> encodeAnyS

isSetS (SetS _) = True
isSetS _ = False
elemSort (SetS s) = s
isData (DataS _ _) = True
isData _ = False
sortArgsOf (DataS _ sArgs) = sArgs
varSortName (VarS name) = name

-- | 'typeVarsOfSort' @s@ : all type variables in @s@
typeVarsOfSort :: Sort -> Set Id
typeVarsOfSort (VarS name) = Set.singleton name
typeVarsOfSort (DataS _ sArgs) = Set.unions (map typeVarsOfSort sArgs)
typeVarsOfSort (SetS s) = typeVarsOfSort s
typeVarsOfSort _ = Set.empty

-- Mapping from type variables to sorts
type SortSubstitution = Map Id Sort

sortSubstitute :: SortSubstitution -> Sort -> Sort
sortSubstitute subst s@(VarS a) = case Map.lookup a subst of
  Just s' -> sortSubstitute subst s'
  Nothing -> s
sortSubstitute subst (DataS name args) = DataS name (map (sortSubstitute subst) args)
sortSubstitute subst (SetS el) = SetS (sortSubstitute subst el)
sortSubstitute _ s = s

distinctTypeVars = map (\i -> "A" ++ show i) [0..]

noncaptureSortSubst :: [Id] -> [Sort] -> Sort -> Sort
noncaptureSortSubst sVars sArgs s =
  let sFresh = sortSubstitute (Map.fromList $ zip sVars (map VarS distinctTypeVars)) s
  in sortSubstitute (Map.fromList $ zip distinctTypeVars sArgs) sFresh

unifySorts :: Set Id -> [Sort] -> [Sort] -> Either (Sort, Sort) SortSubstitution
unifySorts boundTvs = unifySorts' Map.empty
  where
    unifySorts' subst [] []
      = Right subst
    unifySorts' subst (x : xs) (y : ys) | x == y
      = unifySorts' subst xs ys
    unifySorts' subst (SetS x : xs) (SetS y : ys)
      = unifySorts' subst (x:xs) (y:ys)
    unifySorts' subst (DataS name args : xs) (DataS name' args' :ys)
      = if name == name'
          then unifySorts' subst (args ++ xs) (args' ++ ys)
          else Left (DataS name [], DataS name' [])
    unifySorts' subst (AnyS : xs) (_ : ys) = unifySorts' subst xs ys
    unifySorts' subst (_ : xs) (AnyS : ys) = unifySorts' subst xs ys
    unifySorts' subst (VarS x : xs) (y : ys)
      | not (Set.member x boundTvs)
      = case Map.lookup x subst of
            Just s -> unifySorts' subst (s : xs) (y : ys)
            Nothing -> if x `Set.member` typeVarsOfSort y
              then Left (VarS x, y)
              else unifySorts' (Map.insert x y subst) xs ys
    unifySorts' subst (x : xs) (VarS y : ys)
      | not (Set.member y boundTvs)
      = unifySorts' subst (VarS y : ys) (x:xs)
    unifySorts' subst (x: _) (y: _)
      = Left (x, y)

-- | Constraints generated during formula resolution
data SortConstraint = SameSort Sort Sort  -- Two sorts must be the same
  | IsOrd Sort                            -- Sort must have comparisons

-- | Predicate signature: name and argument sorts
data PredSig = PredSig {
  predSigName :: Id,
  predSigArgSorts :: [Sort],
  predSigResSort :: Sort
} deriving (Show, Read, Eq, Ord)

{- Formulas of the refinement logic -}

-- | Unary operators
data UnOp = Neg | Not
  deriving (Show, Read, Eq, Ord)

instance Encode UnOp where
  encode Neg = encodeNeg
  encode Not = encodeNot

-- | Binary operators
data BinOp =
    Times | Plus | Minus |          -- ^ Int -> Int -> Int
    Eq | Neq |                      -- ^ a -> a -> Bool
    Lt | Le | Gt | Ge |             -- ^ Int -> Int -> Bool
    And | Or | Implies | Iff |      -- ^ Bool -> Bool -> Bool
    Union | Intersect | Diff |      -- ^ Set -> Set -> Set
    Member | Subset                 -- ^ Int/Set -> Set -> Bool
  deriving (Show, Read, Eq, Ord)

instance Encode BinOp where
  encode t = case t of
    Times -> encodeTimes
    Plus -> encodePlus
    Minus -> encodeMinus
    Eq -> encodeEq
    Neq -> encodeNeq
    Lt -> encodeLt
    Le -> encodeLe
    Gt -> encodeGt
    Ge -> encodeGe
    And -> encodeAnd
    Or -> encodeOr
    Implies -> encodeImplies
    Iff -> encodeIff
    Union -> encodeUnion
    Intersect -> encodeIntersect
    Diff -> encodeDiff
    Member -> encodeMember
    Subset -> encodeSubset

-- | Variable substitution
type Substitution = Map Id Formula

-- | Formulas of the refinement logic
data Formula =
  BoolLit Bool |                      -- ^ Boolean literal
  IntLit Integer |                    -- ^ Integer literal
  SetLit Sort [Formula] |             -- ^ Set literal ([1, 2, 3])
  Var Sort Id |                       -- ^ Input variable (universally quantified first-order variable)
  Unknown Substitution Id |           -- ^ Predicate unknown (with a pending substitution)
  Unary UnOp Formula |                -- ^ Unary expression
  Binary BinOp Formula Formula |      -- ^ Binary expression
  Ite Formula Formula Formula |       -- ^ If-then-else expression
  Pred Sort Id [Formula] |            -- ^ Logic function application
  Cons Sort Id [Formula] |            -- ^ Constructor application
  All Formula Formula                 -- ^ Universal quantification
  deriving (Show,Read, Eq, Ord)

instance Encode Formula where
  encode t = case t of
    BoolLit b -> encodeBoolLit (encodeBool b)
    IntLit i -> encodeIntLit (pure $ encodeInt $ fromIntegral i)
    SetLit s xs -> encodeSetLit (encode s) (encode xs)
    Var s id -> encodeVar (encode s) (encodeId id)
    Unknown subst id -> encodeUnknown (pure zeros) (encodeId id) -- TODO subst
    Unary op f -> encodeUnary (encode op) (encode f)
    Binary op a b -> encodeBinary (encode op) (encode a) (encode b)
    Ite a b c -> encodeIte (encode a) (encode b) (encode c)
    Pred s id xs -> encodePred (encode s) (encodeId id) (encode xs)
    Cons s id xs -> encodeCons (encode s) (encodeId id) (encode xs)
    All a b -> encodeAll (encode a) (encode b)

dontCare = "_"
valueVarName = "_v"
unknownName (Unknown _ name) = name
varName (Var _ name) = name
varType (Var t _) = t

isVar (Var _ _) = True
isVar _ = False
isCons (Cons _ _ _) = True
isCons _ = False

ftrue = BoolLit True
ffalse = BoolLit False
boolVar = Var BoolS
valBool = boolVar valueVarName
intVar = Var IntS
valInt = intVar valueVarName
vartVar n = Var (VarS n)
valVart n = vartVar n valueVarName
setVar s = Var (SetS (VarS s))
valSet s = setVar s valueVarName
fneg = Unary Neg
fnot = Unary Not
(|*|) = Binary Times
(|+|) = Binary Plus
(|-|) = Binary Minus
(|=|) = Binary Eq
(|/=|) = Binary Neq
(|<|) = Binary Lt
(|<=|) = Binary Le
(|>|) = Binary Gt
(|>=|) = Binary Ge
(|&|) = Binary And
(|||) = Binary Or
(|=>|) = Binary Implies
(|<=>|) = Binary Iff

andClean l r = if l == ftrue then r else (if r == ftrue then l else (if l == ffalse || r == ffalse then ffalse else l |&| r))
orClean l r = if l == ffalse then r else (if r == ffalse then l else (if l == ftrue || r == ftrue then ftrue else l ||| r))
conjunction fmls = foldr andClean ftrue (Set.toList fmls)
disjunction fmls = foldr orClean ffalse (Set.toList fmls)

(/+/) = Binary Union
(/*/) = Binary Intersect
(/-/) = Binary Diff
fin = Binary Member
(/<=/) = Binary Subset

infixl 9 |*|
infixl 8 |+|, |-|, /+/, /-/, /*/
infixl 7 |=|, |/=|, |<|, |<=|, |>|, |>=|, /<=/
infixl 6 |&|, |||
infixr 5 |=>|
infix 4 |<=>|

-- | 'varsOf' @fml@ : set of all input variables of @fml@
varsOf :: Formula -> Set Formula
varsOf (SetLit _ elems) = Set.unions $ map varsOf elems
varsOf v@(Var _ _) = Set.singleton v
varsOf (Unary _ e) = varsOf e
varsOf (Binary _ e1 e2) = varsOf e1 `Set.union` varsOf e2
varsOf (Ite e0 e1 e2) = varsOf e0 `Set.union` varsOf e1 `Set.union` varsOf e2
varsOf (Pred _ _ es) = Set.unions $ map varsOf es
varsOf (Cons _ _ es) = Set.unions $ map varsOf es
varsOf (All x e) = Set.delete x (varsOf e)
varsOf _ = Set.empty

-- | 'unknownsOf' @fml@ : set of all predicate unknowns of @fml@
unknownsOf :: Formula -> Set Formula
unknownsOf u@(Unknown _ _) = Set.singleton u
unknownsOf (Unary Not e) = unknownsOf e
unknownsOf (Binary _ e1 e2) = unknownsOf e1 `Set.union` unknownsOf e2
unknownsOf (Ite e0 e1 e2) = unknownsOf e0 `Set.union` unknownsOf e1 `Set.union` unknownsOf e2
unknownsOf (Pred _ _ es) = Set.unions $ map unknownsOf es
unknownsOf (Cons _ _ es) = Set.unions $ map unknownsOf es
unknownsOf (All _ e) = unknownsOf e
unknownsOf _ = Set.empty

-- | 'posNegUnknowns' @fml@: sets of positive and negative predicate unknowns in @fml@
posNegUnknowns :: Formula -> (Set Id, Set Id)
posNegUnknowns (Unknown _ u) = (Set.singleton u, Set.empty)
posNegUnknowns (Unary Not e) = swap $ posNegUnknowns e
posNegUnknowns (Binary Implies e1 e2) = both2 Set.union (swap $ posNegUnknowns e1) (posNegUnknowns e2)
posNegUnknowns (Binary Iff e1 e2) = both2 Set.union (posNegUnknowns $ e1 |=>| e2) (posNegUnknowns $ e2 |=>| e1)
posNegUnknowns (Binary _ e1 e2) = both2 Set.union (posNegUnknowns e1) (posNegUnknowns e2)
posNegUnknowns (Ite e e1 e2) = both2 Set.union (posNegUnknowns $ e |=>| e1) (posNegUnknowns $ fnot e |=>| e2)
posNegUnknowns _ = (Set.empty, Set.empty)

posUnknowns = fst . posNegUnknowns
negUnknowns = snd . posNegUnknowns

posNegPreds :: Formula -> (Set Id, Set Id)
posNegPreds (Pred BoolS p es) = (Set.singleton p, Set.empty)
posNegPreds (Unary Not e) = swap $ posNegPreds e
posNegPreds (Binary Implies e1 e2) = both2 Set.union (swap $ posNegPreds e1) (posNegPreds e2)
posNegPreds (Binary Iff e1 e2) = both2 Set.union (posNegPreds $ e1 |=>| e2) (posNegPreds $ e2 |=>| e1)
posNegPreds (Binary _ e1 e2) = both2 Set.union (posNegPreds e1) (posNegPreds e2)
posNegPreds (Ite e e1 e2) = both2 Set.union (posNegPreds $ e |=>| e1) (posNegPreds $ fnot e |=>| e2)
posNegPreds _ = (Set.empty, Set.empty)

posPreds = fst . posNegPreds
negPreds = snd . posNegPreds

predsOf :: Formula -> Set Id
predsOf (Pred _ p es) = Set.insert p (Set.unions $ map predsOf es)
predsOf (SetLit _ elems) = Set.unions $ map predsOf elems
predsOf (Unary _ e) = predsOf e
predsOf (Binary _ e1 e2) = predsOf e1 `Set.union` predsOf e2
predsOf (Ite e0 e1 e2) = predsOf e0 `Set.union` predsOf e1 `Set.union` predsOf e2
predsOf (All x e) = predsOf e
predsOf _ = Set.empty

-- | 'leftHandSide' @fml@ : left-hand side of a binary expression
leftHandSide (Binary _ l _) = l
-- | 'rightHandSide' @fml@ : right-hand side of a binary expression
rightHandSide (Binary _ _ r) = r

conjunctsOf (Binary And l r) = conjunctsOf l `Set.union` conjunctsOf r
conjunctsOf f = Set.singleton f

-- | Base type of a term in the refinement logic
sortOf :: Formula -> Sort
sortOf (BoolLit _)                               = BoolS
sortOf (IntLit _)                                = IntS
sortOf (SetLit s _)                              = SetS s
sortOf (Var s _ )                                = s
sortOf (Unknown _ _)                             = BoolS
sortOf (Unary op _)
  | op == Neg                                    = IntS
  | otherwise                                    = BoolS
sortOf (Binary op e1 _)
  | op == Times || op == Plus || op == Minus     = IntS
  | op == Union || op == Intersect || op == Diff = sortOf e1
  | otherwise                                    = BoolS
sortOf (Ite _ e1 _)                              = sortOf e1
sortOf (Pred s _ _)                              = s
sortOf (Cons s _ _)                              = s
sortOf (All _ _)                                 = BoolS

isExecutable :: Formula -> Bool
isExecutable (SetLit _ _) = False
isExecutable (Unary _ e) = isExecutable e
isExecutable (Binary _ e1 e2) = isExecutable e1 && isExecutable e2
isExecutable (Ite e0 e1 e2) = False
isExecutable (Pred _ _ _) = False
isExecutable (All _ _) = False
isExecutable _ = True

-- | 'substitute' @subst fml@: Replace first-order variables in @fml@ according to @subst@
substitute :: Substitution -> Formula -> Formula
substitute subst fml = case fml of
  SetLit b elems -> SetLit b $ map (substitute subst) elems
  Var s name -> case Map.lookup name subst of
    Just f -> f
    Nothing -> fml
  Unknown s name -> Unknown (s `composeSubstitutions` subst) name
  Unary op e -> Unary op (substitute subst e)
  Binary op e1 e2 -> Binary op (substitute subst e1) (substitute subst e2)
  Ite e0 e1 e2 -> Ite (substitute subst e0) (substitute subst e1) (substitute subst e2)
  Pred b name args -> Pred b name $ map (substitute subst) args
  Cons b name args -> Cons b name $ map (substitute subst) args
  All v@(Var _ x) e -> if x `Map.member` subst
                            then error $ unwords ["Scoped variable clashes with substitution variable", x]
                            else All v (substitute subst e)
  otherwise -> fml

-- | Compose substitutions
composeSubstitutions old new =
  let new' = removeId new
  in Map.map (substitute new') old `Map.union` new'
  where
    -- | Remove identity substitutions
    removeId = Map.filterWithKey (\x fml -> not $ isVar fml && varName fml == x)

deBrujns = map (\i -> dontCare ++ show i) [0..]

sortSubstituteFml :: SortSubstitution -> Formula -> Formula
sortSubstituteFml subst fml = case fml of
  SetLit el es -> SetLit (sortSubstitute subst el) (map (sortSubstituteFml subst) es)
  Var s name -> Var (sortSubstitute subst s) name
  Unknown s name -> Unknown (Map.map (sortSubstituteFml subst) s) name
  Unary op e -> Unary op (sortSubstituteFml subst e)
  Binary op l r -> Binary op (sortSubstituteFml subst l) (sortSubstituteFml subst r)
  Ite c l r -> Ite (sortSubstituteFml subst c) (sortSubstituteFml subst l) (sortSubstituteFml subst r)
  Pred s name es -> Pred (sortSubstitute subst s) name (map (sortSubstituteFml subst) es)
  Cons s name es -> Cons (sortSubstitute subst s) name (map (sortSubstituteFml subst) es)
  All x e -> All (sortSubstituteFml subst x) (sortSubstituteFml subst e)
  _ -> fml

noncaptureSortSubstFml :: [Id] -> [Sort] -> Formula -> Formula
noncaptureSortSubstFml sVars sArgs fml =
  let fmlFresh = sortSubstituteFml (Map.fromList $ zip sVars (map VarS distinctTypeVars)) fml
  in sortSubstituteFml (Map.fromList $ zip distinctTypeVars sArgs) fmlFresh

substitutePredicate :: Substitution -> Formula -> Formula
substitutePredicate pSubst fml = case fml of
  Pred b name args -> case Map.lookup name pSubst of
                      Nothing -> Pred b name (map (substitutePredicate pSubst) args)
                      Just value -> substitute (Map.fromList $ zip deBrujns args) (substitutePredicate pSubst value)
  Unary op e -> Unary op (substitutePredicate pSubst e)
  Binary op e1 e2 -> Binary op (substitutePredicate pSubst e1) (substitutePredicate pSubst e2)
  Ite e0 e1 e2 -> Ite (substitutePredicate pSubst e0) (substitutePredicate pSubst e1) (substitutePredicate pSubst e2)
  All v e -> All v (substitutePredicate pSubst e)
  _ -> fml

-- | Negation normal form of a formula:
-- no negation above boolean connectives, no boolean connectives except @&&@ and @||@
negationNF :: Formula -> Formula
negationNF fml = case fml of
  Unary Not e -> case e of
    Unary Not e' -> negationNF e'
    Binary And e1 e2 -> negationNF (fnot e1) ||| negationNF (fnot e2)
    Binary Or e1 e2 -> negationNF (fnot e1) |&| negationNF (fnot e2)
    Binary Implies e1 e2 -> negationNF e1 |&| negationNF (fnot e2)
    Binary Iff e1 e2 -> (negationNF e1 |&| negationNF (fnot e2)) ||| (negationNF (fnot e1) |&| negationNF e2)
    _ -> fml
  Binary Implies e1 e2 -> negationNF (fnot e1) ||| negationNF e2
  Binary Iff e1 e2 -> (negationNF e1 |&| negationNF e2) ||| (negationNF (fnot e1) |&| negationNF (fnot e2))
  Binary op e1 e2
    | op == And || op == Or -> Binary op (negationNF e1) (negationNF e2)
    | otherwise -> fml
  Ite cond e1 e2 -> (negationNF cond |&| negationNF e1) ||| (negationNF (fnot cond) |&| negationNF e2)
  _ -> fml

-- | Disjunctive normal form for unknowns (known predicates treated as atoms)
uDNF :: Formula -> [Formula]
uDNF = dnf' . negationNF
  where
    dnf' e@(Binary Or e1 e2) = if (Set.null $ unknownsOf e1) && (Set.null $ unknownsOf e2)
                                then return e
                                else dnf' e1 ++ dnf' e2
    dnf' (Binary And e1 e2) = do
                                lClause <- dnf' e1
                                rClause <- dnf' e2
                                return $ lClause |&| rClause
    dnf' fml = [fml]

atomsOf fml = atomsOf' (negationNF fml)
  where
    atomsOf' (Binary And l r) = atomsOf' l `Set.union` atomsOf' r
    -- atomsOf' fml@(Binary Or l r) = Set.insert fml (atomsOf' l `Set.union` atomsOf' r)
    atomsOf' (Binary Or l r) = atomsOf' l `Set.union` atomsOf' r
    atomsOf' fml = Set.singleton fml

splitByPredicate :: Set Id -> Formula -> [Formula] -> Maybe (Map Id (Set Formula))
splitByPredicate preds arg fmls = foldM (\m fml -> checkFml fml m fml) Map.empty fmls
  where
    checkFml _ _ fml | fml == arg   = Nothing
    checkFml whole m fml = case fml of
      Pred _ name args ->
        if name `Set.member` preds && length args == 1 && head args == arg
          then return $ Map.insertWith Set.union name (Set.singleton whole) m
          else foldM (checkFml whole) m args
      SetLit _ args -> foldM (checkFml whole) m args
      Unary _ f -> checkFml whole m f
      Binary _ l r -> foldM (checkFml whole) m [l, r]
      Ite c t e -> foldM (checkFml whole) m [c, t, e]
      Cons _ _ args -> foldM (checkFml whole) m args
      _ -> return m


-- | 'setToPredicate' @x s@: predicate equivalent to @x in s@, which does not contain comprehensions
setToPredicate :: Formula -> Formula -> Formula
setToPredicate x (Binary Union sl sr) = Binary Or (setToPredicate x sl) (setToPredicate x sr)
setToPredicate x (Binary Intersect sl sr) = Binary And (setToPredicate x sl) (setToPredicate x sr)
setToPredicate x (Binary Diff sl sr) = Binary And (setToPredicate x sl) (Unary Not (setToPredicate x sr))
setToPredicate x (Ite c t e) = Ite c (setToPredicate x t) (setToPredicate x e)
setToPredicate x s = Binary Member x s

{- Qualifiers -}

-- | Search space for valuations of a single unknown
data QSpace = QSpace {
    _qualifiers :: [Formula],         -- ^ Qualifiers
    _maxCount :: Int                  -- ^ Maximum number of qualifiers in a valuation
  } deriving (Show, Eq, Ord, Read)

instance Encode QSpace where
  encode (QSpace xs _) = encode $ foldr (Binary And) (BoolLit True) xs

makeLenses ''QSpace

emptyQSpace = QSpace [] 0

toSpace mbN quals = let quals' = nub quals in
  case mbN of
    Nothing -> QSpace quals' (length quals')
    Just n -> QSpace quals' n

-- | Mapping from unknowns to their search spaces
type QMap = Map Id QSpace

-- | 'lookupQuals' @qmap g u@: get @g@ component of the search space for unknown @u@ in @qmap@
lookupQuals :: QMap -> Getter QSpace a -> Formula -> a
lookupQuals qmap g (Unknown _ u) = case Map.lookup u qmap of
  Just qs -> view g qs
  Nothing -> error $ unwords ["lookupQuals: missing qualifiers for unknown", u]

lookupQualsSubst :: QMap -> Formula -> [Formula]
lookupQualsSubst qmap u@(Unknown s _) = concatMap go $ map (substitute s) (lookupQuals qmap qualifiers u)
  where
    go u@(Unknown _ _) = lookupQualsSubst qmap u
    go fml = [fml]

type ExtractAssumptions = Formula -> Set Formula

{- Solutions -}

-- | Valuation of a predicate unknown as a set of qualifiers
type Valuation = Set Formula

-- | Mapping from predicate unknowns to their valuations
type Solution = Map Id Valuation

-- | 'topSolution' @qmap@ : top of the solution lattice (maps every unknown in the domain of @qmap@ to the empty set of qualifiers)
topSolution :: QMap -> Solution
topSolution qmap = constMap (Map.keysSet qmap) Set.empty

-- | 'botSolution' @qmap@ : bottom of the solution lattice (maps every unknown in the domain of @qmap@ to all its qualifiers)
botSolution :: QMap -> Solution
botSolution qmap = Map.map (\(QSpace quals _) -> Set.fromList quals) qmap

-- | 'valuation' @sol u@ : valuation of @u@ in @sol@
valuation :: Solution -> Formula -> Valuation
valuation sol (Unknown s u) = case Map.lookup u sol of
  Just quals -> Set.map (substitute s) quals
  Nothing -> error $ unwords ["valuation: no value for unknown", u]

-- | 'applySolution' @sol fml@ : Substitute solutions from sol for all predicate variables in fml
applySolution :: Solution -> Formula -> Formula
applySolution sol fml = case fml of
  Unknown s ident -> case Map.lookup ident sol of
    Just quals -> substitute s $ conjunction quals
    Nothing -> fml
  Unary op e -> Unary op (applySolution sol e)
  Binary op e1 e2 -> Binary op (applySolution sol e1) (applySolution sol e2)
  Ite e0 e1 e2 -> Ite (applySolution sol e0) (applySolution sol e1) (applySolution sol e2)
  All x e -> All x (applySolution sol e)
  otherwise -> fml

-- | 'merge' @sol sol'@ : element-wise conjunction of @sol@ and @sol'@
merge :: Solution -> Solution -> Solution
merge sol sol' = Map.unionWith Set.union sol sol'

{- Solution Candidates -}

-- | Solution candidate
data Candidate = Candidate {
    solution :: Solution,
    validConstraints :: Set Formula,
    invalidConstraints :: Set Formula,
    label :: String
  } deriving (Show)

initialCandidate = Candidate Map.empty Set.empty Set.empty "0"

instance Eq Candidate where
  (==) c1 c2 = Map.filter (not . Set.null) (solution c1) == Map.filter (not . Set.null) (solution c2) &&
               validConstraints c1 == validConstraints c2 &&
               invalidConstraints c1 == invalidConstraints c2

instance Ord Candidate where
  (<=) c1 c2 = Map.filter (not . Set.null) (solution c1) <= Map.filter (not . Set.null) (solution c2) &&
               validConstraints c1 <= validConstraints c2 &&
               invalidConstraints c1 <= invalidConstraints c2
