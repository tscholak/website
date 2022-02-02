---
title: "Unrecurse -- A Recursive Function That Doesn't Recurse"
date: Jan 20, 2022
teaser: >
  Have you ever wanted to write a recursive function and wondered
  what would happen if someone took away recursion from Haskell?
  Say goodbye to recursive function calls, say goodbye to recursive data types.
  How sad Haskell would be without them!
  I'm sure that thought must have occured to you.
  (If not, what are you even doing here?)
  Well, this article has you covered should that day ever come.
  After reading it, you will know how to write
  a recursive function that doesn't recurse.
tags:
  items: [haskell]
image: goodbye.gif
---

Preliminaries
-------------

For this literal Haskell essay, a few language extensions are required.
Nothing extraordinarily fancy, just the usually fancy Haskell flavour.

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

A bunch of these are part of
[GHC2021](https://ghc.gitlab.haskell.org/ghc/doc/users_guide/exts/control.html#extension-GHC2021)
that was introduced in GHC 9.2,
but this blog is boring and still running on GHC 8.10.

Unsurprisingly, we will also work with code that is not in the `Prelude`:

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
  import Control.Monad.Cont (ContT (runContT))
  import Control.Monad.State
    ( MonadState (get, put),
      StateT (runStateT),
      evalStateT,
      modify,
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
  import Prelude hiding (even, odd)
\end{code}

Now, since this is settled, we can get on with the article.

The Best Refactoring You Have Never Done
----------------------------------------

The first part of this article is based on a
[2019 post and talk](https://www.pathsensitive.com/2019/07/the-best-refactoring-youve-never-heard.html)
by [James Koppel](https://www.jameskoppel.com)
about the equivalence of recursion and iteration.

James' talk is a great introduction to the topic,
but it left out a few important details and did not explain
how to generalize the example from the talk to more complicated cases.
Also, the original talk used Java and only a little bit of Haskell,
but I'm going to exclusively use Haskell here.
This will make it clearer for people familiar with Haskell but not with Java,
like me.

James' example is a recursive function that prints the content of a binary tree.
The problem he chose to focus on is
to convert that recursive function to an iterative one.
We will reproduce the conversion process in Haskell code,
and the first step is going to be to define
an abstract data type for binary trees:

\begin{code}
  data Tree a
    = Nil
    | Node
        { left :: Tree a,
          content :: a,
          right :: Tree a
        }
    deriving stock (Eq, Show, Functor, Generic)
\end{code}

For illustration purposes,
we will use the following balanced tree
that carries consecutive integers at its seven leaves:

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

Really, any tree will do, but we will use this one.
We can print the contents of our tree using the `Show` instance of integers.

\begin{code}
  -- | Print the content of a `Tree a`.
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

This is the function we want to convert to an iterative one.
In its current form, it contains two recursive calls to itself.
To eliminate the recursive calls,
we need to perform a number of transformations.
Each transformation is potentially headache inducing,
and I am not going to promise pretty code.
In fact, the code is going to get worse and worse.

With your expectations properly lowered,
let's start with the first transformation.
We need to rewrite the `printTree` function to use continuations,
using something called the "continuation passing style" or CPS.
I'm not going to explain CPS in all its glory,
only that it is a style of programming in which functions do
not return values, but instead repeatedly pass control to
a function called a continuation that decides what to do next.
The mind-bending implications of this style of programming
are introduced and discussed in
[this article on wikibooks.org](https://en.wikibooks.org/wiki/Haskell/Continuation_passing_style).
Have a look at that article if you are interested in the fine details.
It's not necessary to understand the details of CPS to
understand the rest of this article, though.

Haskell is particularly good at handling the CPS style,
rewriting to it is easy and mechanical.
The tool we will use is called the `ContT` monad transformer.
It is found in the [transformers](https://hackage.haskell.org/package/transformers) package.
`ContT` will wrap the `IO` monad,
and we will use the `>>=` operator to chain continuations.
`IO` values need to be lifted to `ContT` values.
This gives us:

\begin{code}
  printTree' ::
    forall a r.
    Show a =>
    Tree a ->
    ContT r IO ()
  printTree' Nil = pure ()
  printTree' Node {..} = do
    printTree' left
    lift $ print content
    printTree' right
\end{code}

The code appears mostly the same, except for a few subtleties.
We have changed the return type and added a new type variable, `r`,
that we cannot touch since it is quantified over.
It is the type of the result of the continuation.
`r` will only be of interest when we run the `ContT` monad transformer.
This is done using the `runContT :: ContT r m a -> (a -> m r) -> m r` function.
It runs the CPS computation encoded in `ContT r m a`
and gets the result, but only if we seed it with one final continuation
of the form `a -> m r`:

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
  runPrintTree' tree =
    runContT (printTree' tree) pure
\end{code}

The final continuation is `pure`,
and it decides that the result of the continuation is `r ~ ()`.
Everything still works as expected, phew.

The `ContT` monad transformer is a great convenience
and allows us to deal with continuations in the familiar monadic style.
What's great for Haskell and its users is not so great
for us in this article, though.
Remember, we want less of that good, idiomatic Haskell, not more of it.
The goodbye to recursion must hurt.
`ContT` is very effective at hiding what is actually happening behind the scenes.
We need to pull that curtain up and see what is going on.
Below is the same code as before, but with the `ContT` monad transformer
inlined into the `printTree'` function:

\begin{code}
  printTree'' ::
    forall a r.
    Show a =>
    Tree a ->
    (() -> IO r) ->
    IO r
  printTree'' Nil = \c -> c ()
  printTree'' Node {..} =
    let first = \c -> printTree'' left c
        second = \c -> print content >>= c
        third = \c -> printTree'' right c
        inner = \c -> second (\x -> (\() -> third) x c)
        outer = \c -> first (\x -> (\() -> inner) x c)
     in outer
\end{code}

This is starting to look nasty.
Don't look at me, I warned you.
`printTree''` is now a higher-order function that takes a continuation
function, `c`, as its second argument.
Notwithstanding, we can also clearly see now that `c`
takes a value of type `()` and returns a value of type `IO r`.
The `()` type of the argument is a consequence of the fact that
the original `printTree` function returned a value of type `IO ()`.

Let's take a look at the `Node` case of `printTree''`.
The `do` notation is [desugared](https://en.wikipedia.org/wiki/Syntactic_sugar)
into nested continuations.
`inner` chains the `second` and `third` functions,
and `outer` chains the `first` and `inner` functions.
`first` happens first, then `second`, and then `third`.
We can convince ourselves that this painfully obfuscated `printTree''` function
is still computing the same result as the `printTree` function.

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
  runPrintTree'' tree =
    printTree'' tree (\() -> pure ())
\end{code}

This sure is nice,
but we still have not eliminated any recursive calls.
In order to make progress, we need to convert the
higher-order `printTree''` function to a first-order function.
This process is called "defunctionalization".
Once again, there is a [wiki article](https://en.wikipedia.org/wiki/Defunctionalization)
on the subject if you are interested.
It's one of those rare and wonderous articles on Wikipedia
that exclusively use Haskell to explain things.
Not that there should be more of them, but I digress.

Concretely, we defunctionalize the `printTree''` function by replacing all
the continuations, `c :: () -> IO r`, with a value of a new data type
`Kont (Next a)`. `Kont` and `Next` look like this:

\begin{code}
  data Kont next
    = Finished
    | More next (Kont next)

  data Next a
    = First (Tree a)
    | Second a
    | Third (Tree a)
\end{code}

`Kont` is a recursive data type with two constructors,
`Finished` and `More`. We use `Finished` to terminate the computation,
and `More` to indicate that we need to continue.
When that happens, we use `Next` to describe
the details of the next step in the computation.
Each constructor of `Next` corresponds to a different action that needs to be taken.
The `First` constructor is named after the `first` function in `printTree''`.
It takes a left subtree as argument.
The `Second` constructor is named after the `second` function in `printTree''`.
Its argument is the content of the current node that needs to be printed.
The `Third` constructor is named after the `third` function in `printTree''`,
and it takes a right subtree as argument.
We need a new function, `apply`, that interprets a `Kont (Next a)` value and
executes the corresponding action:

\begin{code}
  apply ::
    forall a.
    Show a =>
    Kont (Next a) ->
    IO ()
  apply (More (First left) c) =
    printTree''' left c
  apply (More (Second content) c) =
    print content >> apply c
  apply (More (Third right) c) =
    printTree''' right c
  apply Finished =
    pure ()
\end{code}

We can see here how different `Next` values correspond to different actions.
When the computation is finished, `apply` returns `()`.
When there is more work to do,
`apply` either calls the yet-to-be-defined `printTree'''` function
with the next subtree and the continuation value, `c :: Kont (Next a)`,
or it calls the `print` function on the content of the node
and then itself to continue the computation.
The purpose of the `printTree'''` function is to build continuation values
and then call `apply` to execute the corresponding actions:

\begin{code}
  printTree''' ::
    forall a.
    Show a =>
    Tree a ->
    Kont (Next a) ->
    IO ()
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

How do we know that this is all working correctly?
We can run it:

\begin{code}
  -- | Run the defunctionalized CPS computation
  -- for `printTree'''`.
  -- >>> runPrintTree''' exampleTree
  -- 1
  -- 2
  -- 3
  -- 4
  -- 5
  -- 6
  -- 7
  runPrintTree''' :: Show a => Tree a -> IO ()
  runPrintTree''' tree =
    printTree''' tree Finished
\end{code}

Great, we have successfully defunctionalized `printTree''`
and turned it into a first-order function.
This has been a major step, but we are not done yet.
We still have to eliminate the recursive calls.
It only appears as if `printTree'''` doesn't call itself anymore.
In fact, it still does, just indirectly through mutual recursion.
This becomes more apparent
once we inline `apply` into `printTree'''`:

\begin{code}
  printTree'''' ::
    forall a.
    Show a =>
    Tree a ->
    Kont (Next a) ->
    IO ()
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
We went from two recursive calls up to now four.
There is something interesting going on here, though.
The recursive call to `printTree'''` always appears in the tail position.
And this means that we should be able to replace the calls with a loop.
Wikipedia also has [an article on that](https://en.wikipedia.org/wiki/Tail_call).
However, we also notice that `printTree'''`
is called with different arguments in all four cases.
We can't replace these calls with a loop without
removing those arguments first, and we can't remove the arguments
until we have a way to keep track of them.
Or do we? Haskell has a special type, called `State`,
that allows for just that.
On hackage, `State` is available as
[`Control.Monad.Trans.State.Lazy`](https://hackage.haskell.org/package/transformers-0.6.0.2/docs/Control-Monad-Trans-State-Lazy.html).
Rather than passing arguments,
we are going to update the `State` with the arguments' values.
In preparation for this, let us have a closer look at `Kont`.
You may have already noticed, but our `Kont` is isomorphic to `[]`:
`Finished` is the empty list, and `More` is simply the list constructor.
Let's acknowledge that fact by using the `[]` data type directly
and by giving `Kont` a better name: `Stack`.

\begin{code}
  type Stack a = [a]
\end{code}

The only operations on `Stack` we will need in the following are
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

Notice here how we use `get`, `put`, and `modify` to update the `Stack` state.
With `push` and `pop`, `printTree''''` becomes:

\begin{code}
  printTree''''' ::
    forall a.
    Show a =>
    Tree a ->
    StateT (Stack (Next a)) IO ()
  printTree''''' Nil = do
    next <- pop
    case next of
      Just (First left) ->
        printTree''''' left
      Just (Second content) ->
        lift (print content)
          >> printTree''''' Nil
      Just (Third right) ->
        printTree''''' right
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
As always, we need to confirm that this is doing the right thing:

\begin{code}
  -- | Run the defunctionalized CPS computation
  -- for `printTree'''''`.
  -- >>> runPrintTree''''' exampleTree
  -- 1
  -- 2
  -- 3
  -- 4
  -- 5
  -- 6
  -- 7
  runPrintTree''''' :: Show a => Tree a -> IO ()
  runPrintTree''''' tree =
    evalStateT (printTree''''' tree) []
\end{code}

The result is as expected.
This takes care of the `Kont (Next a)` argument,
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

This function just runs the given monadic computation, `m :: m Continue`,
again and again until it returns `Break`.
This is as close to a "while" loop as we can get in Haskell,
we can't remove the recursion here.
Let us pretend the hypothetical recursion-free Haskell
has `while` as a primitive.

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

There are no recursive calls left.
Well, almost, because we are still using the recursive `Tree` type.
We will worry about that later.

This marvel of a function still computes the same result,

\begin{code}
  -- | Run the unrolled `printTree'''''''` program.
  -- >>> runPrintTree''''''' exampleTree
  -- 1
  -- 2
  -- 3
  -- 4
  -- 5
  -- 6
  -- 7 
  runPrintTree''''''' :: Show a => Tree a -> IO ()
  runPrintTree''''''' tree =
    evalStateT printTree''''''' (tree, [])
\end{code}

Fantastic!

Accumulations
-------------

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
  accumTree' Node {..} =
    accumTree' left
      *> tell content
      *> accumTree' right
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

Flattening of A Tree
--------------------

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