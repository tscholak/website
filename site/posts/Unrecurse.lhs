---
title: "Unrecurse -- A Recursive Function That Doesn't Recurse"
date: Jan 20, 2022
teaser: >
  Have you ever wanted to write a recursive function and wondered
  what would happen if someone took away recursion from Haskell?
  Say goodbye to recursive function calls, say goodbye to recursive data types.
  How sad Haskell would be without them!
  I'm sure that thought must have occured to you
  -- if not, what are you even doing here?!
  Well, this article has you covered should that day ever come.
  After reading it, you will know how to write
  a recursive function that doesn't recurse.
tags:
  items: [haskell, recursion]
image: goodbye.gif
---

Preliminaries
-------------

For this literal Haskell essay, a few language extensions are required.
Nothing extraordinarily fancy, just the usually fancy Haskell flavour.

\begin{code}
  {-# LANGUAGE GADTs #-}
  {-# LANGUAGE RecordWildCards #-}
  {-# LANGUAGE DeriveTraversable #-}
  {-# LANGUAGE TypeFamilies #-}
  {-# LANGUAGE ScopedTypeVariables #-}
  {-# LANGUAGE FlexibleContexts #-}
  {-# LANGUAGE MultiParamTypeClasses #-}
  {-# LANGUAGE FlexibleInstances #-}
  {-# LANGUAGE UndecidableInstances #-}
  {-# LANGUAGE DeriveGeneric #-}
  {-# LANGUAGE DerivingVia #-}
  {-# LANGUAGE LambdaCase #-}
\end{code}

A bunch of these are part of
[GHC2021](https://ghc.gitlab.haskell.org/ghc/doc/users_guide/exts/control.html#extension-GHC2021)
that was introduced in GHC 9.2,
but this blog is boring and still running on GHC 8.10.

Unsurprisingly, we will also work with code that is not in the `Prelude`:

\begin{code}
  module Unrecurse where

  import Control.Lens
    ( zoom,
      _1,
      _2,
    )
  import Control.Monad.Cont (ContT (runContT))
  import Control.Monad.State
    ( MonadState (get, put),
      StateT (runStateT),
      evalStateT,
      modify,
    )
  import Control.Monad.Trans (MonadTrans (lift))
  import Control.Monad.Writer (Writer, execWriter, tell)
  import Data.Monoid (Sum (..))
  import GHC.Generics (Generic)
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
    deriving stock (Eq, Show, Functor, Foldable, Generic)
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
We will worry about that in a follow-up piece.

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

Fantastic! What's next? Accumulations.

The `printTree` example from James' 2019 talk is a bit too simple,
because we run `printTree` only for its effects.
We don't care about the result since it is just `()` and
will be the same for all trees we pass to `printTree`.
What if this was different?
In many real applications, we might want to reduce the tree to a value,
say, accumulate a result, like a sum.
This common pattern is captured by the function below:

\begin{code}
  accumTree :: forall a. Monoid a => Tree a -> a
  accumTree Nil = mempty
  accumTree Node {..} =
    let lacc = accumTree left
        racc = accumTree right
     in lacc <> content <> racc
\end{code}

Using `accumTree`, summing up all the content values of our example tree
is a simple matter:

\begin{code}
  -- | Sum the content values of a tree.
  -- >>> sumTree exampleTree
  -- 28
  sumTree :: Num a => Tree a -> a
  sumTree tree = getSum $ accumTree (Sum <$> tree)
\end{code}

`Sum` is a newtype wrapper whose only function is
to select the `Monoid` instance for summation,
where `mempty` is zero and `mappend` is addition.
This is necessary since Haskell doesn't have named class instances.

By the way, since `Tree` has a `Foldable` instance,
we could have achieved the same by using `foldMap` from `Data.Foldable`:

\begin{code}
  -- | Sum the content values of a tree.
  -- >>> sumTree' exampleTree
  -- 28
  sumTree' :: (Num a, Foldable f) => f a -> a
  sumTree' tree = getSum $ foldMap Sum tree
\end{code}

This certainly is convenient. Haskellers love `foldMap`,
because they can use it on any data type that has a `Foldable` instance, not just `Tree`.
But this is supposed to be a tutorial about inconvenient, recursion-free Haskell,
and for that we need to understand how we can remove recursion from `accumTree`.
The tool we are going to use here is the `Writer` monad.

\begin{code}
  accumTree'' ::
    forall a.
    Monoid a =>
    Tree a ->
    Writer a ()
  accumTree'' Nil = pure ()
  accumTree'' Node {..} = do
    accumTree'' left
    tell content
    accumTree'' right
\end{code}

This version of `accumTree` is a bit more complicated.
Accumulation has been moved to the `Writer` monad,
and we need to use `tell` to accumulate the result.
We also are working with a monadic effect, `Writer`,
whereas `accumTree` was pure.
We can extract the result by using `execWriter`:

\begin{code}
  -- | Sum the content values of a tree.
  -- >>> sumTree'' exampleTree
  -- 28
  sumTree'' :: Num a => Tree a -> a
  sumTree'' tree =
    getSum $ execWriter $ accumTree'' (Sum <$> tree)
\end{code}

Great, this works.
Now, sharp eyes will notice that
`accumTree''` and `printTree` from before are structurally equivalent.
The only difference is that `printTree` uses `IO`, and
`accumTree''` uses `Writer a`. That's all.
Based on what we have seen so far,
we can therefore immediately see how `accumTree''` can be made iterative:
 
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

Does this look familiar?
This is the exact same as `printTree'''''''` from above,
except `print content` is replaced by `tell content`.
We run this function using its companion function `sumTree'''''''`
that executes the `Writer` and `State` effects:

\begin{code}
  -- | Run the unrolled `accumTree'''''''` program
  -- and calculate the sum of the content values of the tree.
  -- >>> runAccumTree''''''' exampleTree
  -- 28
  sumTree''''''' :: Num a => Tree a -> a
  sumTree''''''' tree =
    getSum $ execWriter $ runStateT accumTree''''''' (Sum <$> tree, [])
\end{code}

This settles the issue.
Now you know how to write a recursive function that doesn't recurse.

Next time, we will take a closer look at the `Tree` type,
and make sure it doesn't recurse either.
I can't wait for that.
