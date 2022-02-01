---
title: "Unrecurse"
date: Jan 20, 2022
teaser: Let's play with recursion!
tags:
  items: [haskell]
---

A few language extensions are required.

\begin{code}
  {-# LANGUAGE RankNTypes #-}
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
  {-# OPTIONS_GHC -Wno-incomplete-patterns #-}
\end{code}

These are the imports we need.

\begin{code}
  module Unrecurse where

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
  import Control.Monad.Cont (ContT (runContT), MonadTrans (lift))
  import Control.Monad.State
    ( MonadState (get, put),
      StateT (runStateT),
      evalStateT,
      modify,
    )
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
  import Prelude hiding (even, odd)
\end{code}
  
Mutual recursion.

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

The best refactoring we have never heard of
-------------------------------------------

The first part of this literal Haskell essay is based on a
[2019 post and talk](https://www.pathsensitive.com/2019/07/the-best-refactoring-youve-never-heard.html)
by [James Koppel](https://www.jameskoppel.com)
about the equivalence of recursion and iteration.

James' talk is a great introduction to the topic,
but it left out a few details and did not explain
how to generalize the example from the talk to more complicated cases.
The original talk used Java and a little bit of Haskell,
but I'm going to use Haskell exclusively here.
This will make it clearer for people familiar with Haskell.

Data type for a binary tree.

\begin{code}
  data Tree a
    = Nil
    | Node {left :: Tree a, content :: a, right :: Tree a}
    deriving stock (Eq, Show, Functor, Generic)
\end{code}

We will use the following integer example tree in the following.

\begin{code}
  exampleTree :: Tree Int
  exampleTree =
    Node
      ( Node
          (Node Nil 1 Nil)
          2
          (Node Nil 3 Nil)
      )
      4
      ( Node
          (Node Nil 5 Nil)
          6
          (Node Nil 7 Nil)
      )
\end{code}

Any tree will do, but we will use this one.
We can print the content of the tree using the `Show` instance of `a`.

\begin{code}
  -- | Print the content of a tree.
  -- >>> printTree exampleTree
  -- 1
  -- 2
  -- 3
  -- 4
  -- 5
  -- 6
  -- 7
  printTree :: forall a. Show a => Tree a -> IO ()
  printTree Nil = pure ()
  printTree Node {..} = do
    printTree left
    print content
    printTree right
\end{code}

We have two recursive calls to `printTree` here.

https://hexagoxel.de/postsforpublish/posts/2018-09-09-cont-part-one.html

To eliminate the recursive calls, we need to perform a number of transformations.

First, we rewrite the `printTree` function to use continuations.
In Haskell, this is usually done by using the `ContT` monad transformer.

\begin{code}
  printTree' :: forall a r. Show a => Tree a -> ContT r IO ()
  printTree' Nil = pure ()
  printTree' Node {..} = do
    printTree' left
    lift $ print content
    printTree' right
\end{code}

The code appears mostly the same, except for a few subtleties.
We have changed the return type and added a new type variable, `r`,
which is the type of the result of the continuation.
We can run the `ContT r m a` monad transformer
using the `runContT :: ContT r m a -> (a -> m r) -> m r` function
to get the result:

\begin{code}
  -- | Run the `ContT` computation for `printTree'`.
  -- >>> runPrintTree' exampleTree
  -- 1
  -- 2
  -- 3
  -- 4
  -- 5
  -- 6
  -- 7
  runPrintTree' :: Show a => Tree a -> IO ()
  runPrintTree' tree = runContT (printTree' tree) pure
\end{code}

The `ContT` monad transformer is a great convenience
and allows us to deal with continuations in the familiar monadic style.
However, it also hides what is actually happening behind the scenes.
Below is the same code as before, but with the `ContT` monad transformer
inlined into the `printTree'` function.

\begin{code}
  printTree'' :: forall a r. Show a => Tree a -> (() -> IO r) -> IO r
  printTree'' Nil = \c -> c ()
  printTree'' Node {..} =
    let first = \c -> printTree'' left c
        second = \c -> print content >>= c
        third = \c -> printTree'' right c
        inner = \c -> second (\x -> (\() -> third) x c)
        outer = \c -> first (\x -> (\() -> inner) x c)
     in outer
\end{code}

We can see now that the continuation, `c`, is a function
that takes a value of type `()` and returns a value of type `IO r`.
The `do` notation is desugared into nested continuations.
`inner` chains the `second` and `third` functions,
and `outer` chains the `first` and `inner` functions.
We can convince ourselves that the `printTree''` function is computing the
same result as the `printTree'` function.

\begin{code}
  -- | Run the CPS computation for `printTree''`.
  -- >>> runPrintTree'' exampleTree
  -- 1
  -- 2
  -- 3
  -- 4
  -- 5
  -- 6
  -- 7
  runPrintTree'' :: Show a => Tree a -> IO ()
  runPrintTree'' tree = printTree'' tree (\() -> pure ())
\end{code}

At this point, we still have not eliminated any recursive calls.
In order to make progress, we need to convert the
higher-order `printTree''` function to a first-order function.
This process is called "defunctionalization".
We defunctionalize the `printTree''` function by replacing all
the continuation, `c :: () -> IO r`, with a value of the new data type
`Kont (Next a)`, where

\begin{code}
  data Kont a = Finished | More a (Kont a)

  data Next a = First (Tree a) | Second a | Third (Tree a)
\end{code}

`Next` describes the next step in the computation.
Each constructor corresponds to a different action.
The `First` constructor is named after the `first` function in `printTree''`.
It takes a left subtree as argument.
The `Second` constructor is named after the `second` function in `printTree''`.
It's argument is the content of the current node.
The `Third` constructor is named after the `third` function in `printTree''`,
and it takes a right subtree as argument.
We need a new function, `apply`, that interprets a `Kont (Next a)` value and
executes the corresponding action:

\begin{code}
  apply :: forall a. Show a => Kont (Next a) -> IO ()
  apply (More (First left) c) = printTree''' left c
  apply (More (Second content) c) = print content >> apply c
  apply (More (Third right) c) = printTree''' right c
  apply Finished = pure ()
\end{code}

We can see here how different `Next` values correspond to different actions.
When the computation is finished, `apply` returns `()`.
When there is more work to do,
`apply` either calls the `printTree'''` function with the next subtree
and the continuation value, `c`, or it calls the `print` function
on the content of the node and then itself to continue the computation.
The purpose of the `printTree'''` function is to build continuation values
and then call `apply` to execute the corresponding actions:

\begin{code}
  printTree''' :: forall a. Show a => Tree a -> Kont (Next a) -> IO ()
  printTree''' Nil c = apply c
  printTree''' Node {..} c =
    apply
      ( More
          (First left)
          ( More
              (Second content)
              (More (Third right) c)
          )
      )
\end{code}

How do we know that `apply` is working correctly?
We can run it:

\begin{code}
  -- | Run the defunctionalized CPS computation for `printTree'''`.
  -- >>> runPrintTree''' exampleTree
  -- 1
  -- 2
  -- 3
  -- 4
  -- 5
  -- 6
  -- 7
  runPrintTree''' :: Show a => Tree a -> IO ()
  runPrintTree''' tree = printTree''' tree Finished
\end{code}

Great, we have successfully defunctionalized `printTree''`
and turned it into a first-order function.
We are not done yet, though.
We still have to eliminate the recursive calls.
It only appears as if `printTree'''` doesn't call itself anymore.
In fact, it still does which becomes more apparent
when we inline `apply` into `printTree'''`:

\begin{code}
  printTree'''' :: forall a. Show a => Tree a -> Kont (Next a) -> IO ()
  printTree'''' Nil (More (First left) c) =
    printTree'''' left c
  printTree'''' Nil (More (Second content) c) =
    print content >> printTree'''' Nil c
  printTree'''' Nil (More (Third right) c) =
    printTree'''' right c
  printTree'''' Nil Finished = pure ()
  printTree'''' Node {..} c =
    printTree''''
      Nil
      ( More
          (First left)
          ( More
              (Second content)
              (More (Third right) c)
          )
      )
\end{code}

At first glance, this doesn't look better than what we had before.
We went from two recursive calls up to four.
There is something interesting going on here, though.
The recursive call to `printTree'''` always appears in the tail position.
And this means that we should be able to replace the calls with a loop.
However, we also notice that `printTree'''`
is called with different arguments in all four cases.
We can't replace these calls with a loop without
removing those arguments first, and we can't remove the arguments
until we have a way to keep track of them.
Or do we? Haskell has a special type, called `State`,
that allows for just that.
Rather than passing arguments, we update the `State` with their values.

In preparation for this, let us have a closer look at `Kont`.
You may have already noticed, but our `Kont` is isomorphic to `[]`:
`Finished` is the empty list, and `More` is simply the list constructor.
Let's acknowledge that fact by using the `[]` constructor directly
and by giving `Kont` a better name: `Stack`.

\begin{code}
  type Stack a = [a]
\end{code}

The only operations on `Stack` we will need are
adding and removing single elements from its end.
These operations are commonly called `push` and `pop`.
They can be implemented as follows:

\begin{code}
  push ::
    forall a m.
    MonadState (Stack a) m =>
    a ->
    m ()
  push x = modify (x :)

  pop ::
    forall a m.
    MonadState (Stack a) m =>
    m (Maybe a)
  pop =
    get >>= \case
      [] -> pure Nothing
      (x : xs) -> put xs >> pure (Just x)
\end{code}

Notice here how we use `get` and `put` to update the `Stack` state.
With these, `printTree''''` becomes:

\begin{code}
  printTree''''' ::
    forall a.
    Show a =>
    Tree a ->
    StateT (Stack (Next a)) IO ()
  printTree''''' Nil = do
    next <- pop
    case next of
      Just (First left) -> printTree''''' left
      Just (Second content) -> lift (print content) >> printTree''''' Nil
      Just (Third right) -> printTree''''' right
      Nothing -> pure ()
  printTree''''' Node {..} = do
    push (Third right)
    push (Second content)
    push (First left)
    printTree''''' Nil
\end{code}

One thing to note here is that we use `StateT`, not `State`, to
represent the state of our computation.
This is because `StateT` is a monad transformer,
and we already have effects in `IO` that we need to lift into it.

Rather than building a value of type `Stack (Next a)`
that we pass to `printTree''''`,
we use `push` and `pop` to update the `Stack` state.
We need to confirm that this is doing the right thing:

\begin{code}
  -- | Run the defunctionalized CPS computation for `printTree'''''`.
  -- >>> runPrintTree''''' exampleTree
  -- 1
  -- 2
  -- 3
  -- 4
  -- 5
  -- 6
  -- 7
  runPrintTree''''' :: Show a => Tree a -> IO ()
  runPrintTree''''' tree = evalStateT (printTree''''' tree) []
\end{code}

The result is as expected.
This takes care of the `Stack` argument,
but we still need to eliminate the `Tree a` argument.
We can use the same trick again and add `Tree a` to the state:

\begin{code}
  printTree'''''' ::
    forall a.
    Show a =>
    StateT (Tree a, Stack (Next a)) IO ()
  printTree'''''' = do
    tree <- zoom _1 get
    case tree of
      Nil -> do
        c <- zoom _2 pop
        case c of
          Just (First left) -> do
            zoom _1 $ put left
            printTree''''''
          Just (Second content) -> do
            lift (print content)
            zoom _1 $ put Nil
            printTree''''''
          Just (Third right) -> do
            zoom _1 $ put right
            printTree''''''
          Nothing -> pure ()
      Node {..} -> do
        zoom _2 $ push (Third right)
        zoom _2 $ push (Second content)
        zoom _2 $ push (First left)
        zoom _1 $ put Nil
        printTree''''''
\end{code}

In this new version,
`printTree''''''` appears in the tail position
and does not have any arguments.
We are now ready for the final magic step,
which is to eliminate all recursive calls.

Let us introduce a little helper, `while`:

\begin{code}
  data Continue = Continue | Break

  while ::
    forall m.
    Monad m =>
    m Continue ->
    m ()
  while m = m >>= \case
    Continue -> while m
    Break -> pure ()
\end{code}

This function just runs the given monadic computation
again and again until it returns `Break`.
This is as close to a loop as we can get in Haskell.
With this, we finally have:

\begin{code}
  printTree''''''' ::
    forall a.
    Show a =>
    StateT (Tree a, Stack (Next a)) IO ()
  printTree''''''' =
    while $ do
      tree <- zoom _1 get
      case tree of
        Nil -> do
          c <- zoom _2 pop
          case c of
            Just (First left) -> do
              zoom _1 $ put left
              pure Continue
            Just (Second content) -> do
              lift (print content)
              zoom _1 $ put Nil
              pure Continue
            Just (Third right) -> do
              zoom _1 $ put right
              pure Continue
            Nothing -> pure Break
        Node {..} -> do
          zoom _2 $ push (Third right)
          zoom _2 $ push (Second content)
          zoom _2 $ push (First left)
          zoom _1 $ put Nil
          pure Continue
\end{code}

```haskell
runStateT printTree''''''' (exampleTree, []) :: IO ()
```

\begin{code}
  accumTree :: forall a. Monoid a => Tree a -> a
  accumTree Nil = mempty
  accumTree Node {..} =
    let lacc = accumTree left
        racc = accumTree right
     in lacc <> content <> racc
\end{code}

\begin{code}
  accumTree' :: forall a. Monoid a => Tree a -> Writer a ()
  accumTree' Nil = pure ()
  accumTree' Node {..} = do
    accumTree' left
    tell content
    accumTree' right
\end{code}

```haskell
execWriter $ accumTree' (Sum <$> exampleTree)
```

\begin{code}
  accumTree''''''' ::
    forall a.
    Monoid a =>
    StateT (Tree a, Stack (Next a)) (Writer a) ()
  accumTree''''''' =
    while $ do
      tree <- zoom _1 get
      case tree of
        Nil -> do
          c <- zoom _2 pop
          case c of
            Just (First left) -> do
              zoom _1 $ put left
              pure Continue
            Just (Second content) -> do
              lift (tell content)
              zoom _1 $ put Nil
              pure Continue
            Just (Third right) -> do
              zoom _1 $ put right
              pure Continue
            Nothing -> pure Break
        Node {..} -> do
          zoom _2 $ push (Third right)
          zoom _2 $ push (Second content)
          zoom _2 $ push (First left)
          zoom _1 $ put Nil
          pure Continue
\end{code}

```haskell
execWriter $ runStateT accumTree''''''' (Sum <$> exampleTree, [])
```

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
