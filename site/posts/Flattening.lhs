---
title: "Flattening -- How to Get from A Tree to A Flat Shape And Back Again"
date: Feb 2, 2022
teaser: >
  The flattening has begun.
tags:
  items: [haskell, recursion]
---

\begin{code}
  {-# LANGUAGE TypeApplications #-}
  {-# LANGUAGE GADTs #-}
  {-# LANGUAGE RecordWildCards #-}
  {-# LANGUAGE DeriveTraversable #-}
  {-# LANGUAGE TemplateHaskell #-}
  {-# LANGUAGE TypeFamilies #-}
  {-# LANGUAGE ScopedTypeVariables #-}
  {-# LANGUAGE BangPatterns #-}
  {-# LANGUAGE FlexibleContexts #-}
  {-# LANGUAGE MultiParamTypeClasses #-}
  {-# LANGUAGE DefaultSignatures #-}
  {-# LANGUAGE FlexibleInstances #-}
  {-# LANGUAGE TypeOperators #-}
  {-# LANGUAGE UndecidableInstances #-}
  {-# LANGUAGE DeriveGeneric #-}
  {-# LANGUAGE DerivingVia #-}
  {-# LANGUAGE GeneralizedNewtypeDeriving #-}
  {-# LANGUAGE StandaloneDeriving #-}
  {-# LANGUAGE LambdaCase #-}
\end{code}

\begin{code}
  module Flattening where

  import Control.Applicative (Alternative (empty, (<|>)))
  import Control.Lens
    ( Cons (_Cons),
      Prism,
      cons,
      prism,
      uncons,
      withPrism,
      zoom,
      _1,
      _2,
    )
  import Control.Monad (MonadPlus, mfilter)
  import Control.Monad.State
    ( MonadState (get, put),
      StateT (runStateT),
      evalStateT,
    )
  import Control.Monad.Trans (MonadTrans (lift))
  import Control.Monad.Writer (Writer, execWriter, tell)
  import Data.Coerce (coerce)
  import Data.Functor.Foldable (Base, Corecursive (embed), Recursive (cata))
  import Data.Functor.Foldable.TH (makeBaseFunctor)
  import Data.Kind (Type)
  import Data.Maybe (fromJust)
  import Data.Monoid (Sum (..))
  import Data.Vector (Vector)
  import GHC.Generics
    ( Generic (Rep, from, to),
      K1 (K1, unK1),
      M1 (M1, unM1),
      U1 (U1),
      V1,
      type (:*:) ((:*:)),
      type (:+:) (L1, R1),
    )
  import Unrecurse (Continue (Break, Continue), Stack, Tree, exampleTree, pop, push, while)
  import Prelude hiding (even, odd)
\end{code}

The Flattening of A Tree
------------------------

Previously, we have seen how one can remove recursive calls from a function.
We learned about continuations, defunctionalization, and monadic `State` effects.
We used these techniques to reimplement two simple recursive functions,
`printTree` and `accumTree`, using only iteration.
These functions are both examples of a fold that deconstructs a binary tree,
`Tree a`, bit by bit to produce an effect or a value, respectively.
The `Tree` type is a recursive data type, and we did not dare to try to remove recursion from it.
This time, we are more ambitious. By the end of this article,
we will not only know how to remove recursive calls from a function,
we will also know how to remove recursion from a data type.

We need a few ingredients:

* `Token`s, which are a set of values that are going to represent different pieces of a `Tree`.
* A `Tape`, which is a linear data structure that can be written to and read from and that can be used to represent a whole `Tree`.
* A linearizer that can convert a `Tree` to a `Tape` of `Token`s.
* A parser that can convert a `Tape` of `Token`s to a `Tree`.
* A pattern functor for the `Tree` type, which can be used to construct or deconstruct a tree iteratively.
* Lots and lots of boilerplate with Haskell `Generic` code.

The high-level idea of all of this is that we are going to store our tree
in a linear data structure, a tape, in a fashion that allows us to [tbd]

\begin{code}
  data Token
    = Rec Int
    | L
    | R
    | I Int
    deriving stock (Eq, Show)
\end{code}

\begin{code}
  newtype Tape t a = Tape {unTape :: t a}
    deriving stock (Eq, Show)
    deriving newtype (Functor, Applicative, Monad, Alternative, Foldable)

  instance Cons (t a) (t b) a b => Cons (Tape t a) (Tape t b) a b where
    _Cons = withPrism cons' $ \review' preview' ->
      prism (coerce review') (coerce preview')
      where
        cons' :: Prism (t a) (t b) (a, t a) (b, t b)
        cons' = _Cons

  type TTape t = Tape t Token
\end{code}

\begin{code}
  type To t a = a -> TTape t

  type From b t a = StateT (TTape t) b a
\end{code}

\begin{code}
  token ::
    forall b t.
    ( MonadFail b,
      Cons (TTape t) (TTape t) Token Token
    ) =>
    From b t Token
  token = do
    t <- get
    case uncons t of
      Nothing -> fail "unexpected end of input"
      Just (x, xs) -> put xs >> pure x
\end{code}

\begin{code}
  isToken ::
    forall b t.
    ( MonadFail b,
      MonadPlus b,
      Cons (TTape t) (TTape t) Token Token
    ) =>
    Token ->
    From b t Token
  isToken t = mfilter (== t) token
\end{code}

\begin{code}
  class ToTokens (t :: Type -> Type) (a :: Type) where
    linearize :: To t a
    default linearize :: (Recursive a, ToTokensStep t (Base a)) => To t a
    linearize = cata linearizeStep
\end{code}

\begin{code}
  instance (Alternative t) => ToTokens t Int where
    linearize i = pure (I i)

  instance (Alternative t, Foldable t) => ToTokens t (TTape t) where
    linearize tape = pure (Rec $ length tape) <|> tape

  class ToTokensStep (t :: Type -> Type) (base :: Type -> Type) where
    linearizeStep :: To t (base (TTape t))
    default linearizeStep ::
      ( Alternative t,
        Foldable t,
        Generic (base (TTape t)),
        GToTokensStep t (Rep (base (TTape t)))
      ) =>
      To t (base (TTape t))
    linearizeStep = gLinearizeStep . GHC.Generics.from

  class GToTokensStep (t :: Type -> Type) (f :: Type -> Type) where
    gLinearizeStep :: forall a. To t (f a)

  instance GToTokensStep t V1 where
    gLinearizeStep v = v `seq` error "GToTokensStep.V1"

  instance Alternative t => GToTokensStep t U1 where
    gLinearizeStep _ = Tape empty

  instance
    ( Applicative t,
      Alternative t,
      Foldable t,
      GToTokensStep t f,
      GToTokensStep t g
    ) =>
    GToTokensStep t (f :+: g)
    where
    gLinearizeStep (L1 x) = pure L <|> gLinearizeStep x
    gLinearizeStep (R1 x) = pure R <|> gLinearizeStep x

  instance
    ( Alternative t,
      Foldable t,
      GToTokensStep t f,
      GToTokensStep t g
    ) =>
    GToTokensStep t (f :*: g)
    where
    gLinearizeStep (x :*: y) = gLinearizeStep x <|> gLinearizeStep y

  instance ToTokens t c => GToTokensStep t (K1 i c) where
    gLinearizeStep = linearize . unK1

  instance GToTokensStep t f => GToTokensStep t (M1 i c f) where
    gLinearizeStep = gLinearizeStep . unM1

  class FromTokensStep (b :: Type -> Type) (t :: Type -> Type) (base :: Type -> Type) where
    parseStep :: From b t (base (TTape t))
    default parseStep ::
      ( Functor b,
        Generic (base (TTape t)),
        GFromTokensStep b t (Rep (base (TTape t)))
      ) =>
      From b t (base (TTape t))
    parseStep = to <$> gParseStep

  class GFromTokensStep (b :: Type -> Type) (t :: Type -> Type) (f :: Type -> Type) where
    gParseStep :: forall a. From b t (f a)

  instance MonadFail b => GFromTokensStep b t V1 where
    gParseStep = fail "GFromTokensStep.V1"

  instance Monad b => GFromTokensStep b t U1 where
    gParseStep = pure U1

  instance
    ( MonadFail b,
      MonadPlus b,
      Cons (TTape t) (TTape t) Token Token,
      GFromTokensStep b t f,
      GFromTokensStep b t g
    ) =>
    GFromTokensStep b t (f :+: g)
    where
    gParseStep = (isToken L >> L1 <$> gParseStep) <|> (isToken R >> R1 <$> gParseStep)

  instance
    ( MonadFail b,
      MonadPlus b,
      Cons (TTape t) (TTape t) Token Token,
      GFromTokensStep b t f,
      GFromTokensStep b t g
    ) =>
    GFromTokensStep b t (f :*: g)
    where
    gParseStep = (:*:) <$> gParseStep <*> gParseStep

  instance (Monad b, FromTokens b t c) => GFromTokensStep b t (K1 i c) where
    gParseStep = K1 <$> parse

  instance (Functor b, GFromTokensStep b t f) => GFromTokensStep b t (M1 i c f) where
    gParseStep = M1 <$> gParseStep
\end{code}

\begin{code}
  resetParse :: Monad b => From b t a -> TTape t -> From b t a
  resetParse m = lift . evalStateT m

  class FromTokens (b :: Type -> Type) (t :: Type -> Type) (a :: Type) where
    parse :: From b t a
    default parse ::
      ( Corecursive a,
        Monad b,
        Traversable (Base a),
        FromTokensStep b t (Base a)
      ) =>
      From b t a
    parse = go where go = fmap embed $ parseStep >>= traverse (resetParse go)
\end{code}

`makeBaseFunctor` is a template haskell function that generates the base functor, `TreeF`, for the `Tree` type:

\begin{code}
  makeBaseFunctor ''Tree
\end{code}

With this, we can auto-derive a bunch of instances for the `TreeF` and `Tree` types:

\begin{code}
  deriving stock instance (Show a, Show r) => Show (TreeF a r)

  deriving stock instance Generic (TreeF a r)

  instance (Alternative t, Foldable t, ToTokens t a) => ToTokensStep t (TreeF a)

  instance (ToTokensStep t (TreeF a)) => ToTokens t (Tree a)

  instance
    ( MonadFail b,
      MonadPlus b,
      Cons (t Token) (t Token) Token Token,
      Alternative t,
      FromTokens b t a
    ) =>
    FromTokensStep b t (TreeF a)

  instance (Monad b, FromTokensStep b t (TreeF a)) => FromTokens b t (Tree a)
\end{code}

\begin{code}

  instance
    ( MonadFail b,
      Cons (TTape t) (TTape t) Token Token
    ) =>
    FromTokens b t Int
    where
    parse =
      token >>= \case
        I i -> pure i
        _ -> fail "expected Int"

  instance
    ( MonadFail b,
      Alternative t,
      Cons (TTape t) (TTape t) Token Token
    ) =>
    FromTokens b t (TTape t)
    where
    parse =
      token >>= \case
        Rec n -> go n
          where
            go :: Int -> From b t (TTape t)
            go 0 = pure empty
            go n' = cons <$> token <*> go (n' - 1)
        _ -> fail "expected Rec"
\end{code}

```
evalStateT parse (linearize @[] exampleTree) == Just exampleTree
```

\begin{code}
  data NextF a r = FirstF r | SecondF a | ThirdF r
\end{code}

\begin{code}
  accumTree'''''''' ::
    forall t a.
    ( Alternative t,
      Foldable t,
      Monoid a,
      ToTokens t a,
      FromTokens Maybe t a,
      Cons (t Token) (t Token) Token Token
    ) =>
    StateT (TTape t, Stack (NextF a (TTape t))) (Writer a) ()
  accumTree'''''''' =
    while $ do
      treeF <- fromJust . evalStateT parseStep <$> zoom _1 get
      case treeF of
        NilF -> do
          c <- zoom _2 pop
          case c of
            Just (FirstF leftF) -> do
              zoom _1 $ put leftF
              pure Continue
            Just (SecondF contentF) -> do
              lift (tell contentF)
              zoom _1 $ put (linearizeStep $ NilF @a)
              pure Continue
            Just (ThirdF rightF) -> do
              zoom _1 $ put rightF
              pure Continue
            Nothing -> pure Break
        NodeF {..} -> do
          zoom _2 $ push (ThirdF rightF)
          zoom _2 $ push (SecondF contentF)
          zoom _2 $ push (FirstF leftF)
          zoom _1 $ put (linearizeStep $ NilF @a)
          pure Continue
\end{code}

\begin{code}
  makeBaseFunctor ''Sum

  deriving stock instance (Show a, Show r) => Show (SumF a r)

  deriving stock instance Generic (SumF a r)

  instance
    ( Alternative t,
      Foldable t,
      ToTokens t a
    ) =>
    ToTokensStep t (SumF a)

  instance
    ( Alternative t,
      Foldable t,
      ToTokens t a
    ) =>
    ToTokens t (Sum a)

  instance
    ( Monad b,
      Alternative t,
      Foldable t,
      FromTokens b t a
    ) =>
    FromTokensStep b t (SumF a)

  instance
    ( Monad b,
      Alternative t,
      Foldable t,
      FromTokens b t a
    ) =>
    FromTokens b t (Sum a)
\end{code}

\begin{code}
  -- | Run stuff
  -- >>> r
  -- Sum {getSum = 28}
  r :: Sum Int
  r = execWriter $ runStateT (accumTree'''''''' @Vector) (linearize $ Sum <$> exampleTree, [])
\end{code}

Mutual Recursion
----------------

\begin{code}
  even :: Int -> Bool
  even 0 = True
  even n = odd (n - 1)

  odd :: Int -> Bool
  odd 0 = False
  odd n = even (n - 1)
\end{code}

<https://www.haskellforall.com/2012/06/you-could-have-invented-free-monads.html>
<https://www.tweag.io/blog/2018-02-05-free-monads/>
<https://gist.github.com/eamelink/4466932a11d8d92a6b76e80364062250>

The trampoline is the Free monad.

\begin{code}
  data Trampoline f r
    = Trampoline {bounce :: f (Trampoline f r)}
    | Done {result :: r}

  instance Functor f => Functor (Trampoline f) where
    fmap f (Trampoline m) = Trampoline (fmap (fmap f) m)
    fmap f (Done r) = Done (f r)

  instance Functor f => Applicative (Trampoline f) where
    pure = Done
    Done f <*> Done x = Done $ f x
    Done f <*> Trampoline mx = Trampoline $ fmap f <$> mx
    Trampoline mf <*> x = Trampoline $ fmap (<*> x) mf

  instance Functor f => Monad (Trampoline f) where
    return = Done
    Done x >>= f = f x
    Trampoline mx >>= f = Trampoline (fmap (>>= f) mx)

  liftF :: Functor f => f r -> Trampoline f r
  liftF m = Trampoline (fmap Done m)
\end{code}

The DSL for the even/odd problem.

\begin{code}
  data EvenOddF next
    = Even Int (Bool -> next)
    | Odd Int (Bool -> next)
    deriving stock (Functor)

  -- instance Functor EvenOddF where
  --   fmap f (Even n k) = Even n (f . k)
  --   fmap f (Odd n k) = Odd n (f . k)

  type EvenOdd = Trampoline EvenOddF
\end{code}

Rewritten in terms of the DSL.

\begin{code}
  even' :: Int -> EvenOdd Bool
  even' 0 = Done True
  even' n = liftF (Odd (n - 1) id)

  odd' :: Int -> EvenOdd Bool
  odd' 0 = Done False
  odd' n = liftF (Even (n - 1) id)

  evenOddHandler :: EvenOddF (EvenOdd r) -> EvenOdd r
  evenOddHandler (Even n k) = even' n >>= k
  evenOddHandler (Odd n k) = odd' n >>= k
\end{code}

Reduce a trampoline to a value.

\begin{code}
  iterTrampoline ::
    Functor f =>
    (f (Trampoline f r) -> Trampoline f r) ->
    Trampoline f r ->
    r
  iterTrampoline h = go
    where
      go Done {..} = result
      go Trampoline {..} = go (h bounce)
\end{code}

Run the trampoline to completion with the even/odd handler.

\begin{code}
  runEvenOdd :: EvenOdd r -> r
  runEvenOdd = iterTrampoline evenOddHandler
\end{code}

<https://stackoverflow.com/questions/57733363/how-to-adapt-trampolines-to-continuation-passing-style>

Fibonacci numbers.

\begin{code}
  fib :: Int -> Int
  fib n = go n 0 1
    where go !n !a !b | n == 0    = a
                      | otherwise = go (n - 1) b (a + b)

  data FibF next =
      FibF Int Int Int (Int -> next)
    deriving stock Functor

  type Fib = Trampoline FibF

  fib' :: Int -> Fib Int
  fib' n = liftF (FibF n 0 1 id)

  fibHandler :: FibF (Fib r) -> Fib r
  fibHandler (FibF 0 a _ k) = Done a >>= k
  fibHandler (FibF n a b k) = liftF (FibF (n - 1) b (a + b) id) >>= k

  runFib :: Fib Int -> Int
  runFib = iterTrampoline fibHandler
\end{code}