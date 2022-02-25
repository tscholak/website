---
title: "Flattening -- How to Get from A Tree to A Flat Shape And Back Again"
date: Feb 2, 2022
teaser: >
tags:
  items: [haskell, recursion, generics, parsing]
image: tape.gif
---

Preliminaries
-------------

This is a literate Haskell essay.
Every line of program code in this article has been checked by the Haskell compiler.
Every example and property in the Haddock comments has been tested by the doctest tool.

To make this a proper Haskell file, it needs a header.
There are several language extensions we need to enable:

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
  {-# LANGUAGE InstanceSigs #-}
\end{code}

Nice, this is more looking like your typical Haskell file now.
We will also need to import some libraries, functions, and types:

\begin{code}
  module Flattening where

  import Control.Applicative (Alternative (empty, (<|>)))
  import Control.Lens
    ( Cons (_Cons),
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
  import Data.Functor.Foldable (Base, Corecursive (embed), Recursive (cata, project))
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
  import Unrecurse (Continue (..), Kont (..), Stack, Tree, exampleTree, pop, push, while)
  import Prelude hiding (even, odd)
\end{code}

Ok, enough beating around the bush.
Now we can start with the actual content of the essay.

The Flattening of A Tree
------------------------

[Last time](/posts/Unrecurse.html) on this channel,
we have seen how one can remove recursive calls from a function.
We learned about continuations, defunctionalization, and monadic `State` effects.
We used these techniques to reimplement two simple recursive functions,
`printTree` and `accumTree`, using only iteration.
These functions are both specific examples of a `fold`.
They consume a value of type `Tree`,
a data type for binary trees with two constructors, `Nil` and `Node`.
`printTree` reduces the tree bit by bit in depth-first left-to-right order
to an effect, that is, printing each leaf value to stdout as it is encountered.
And `accumTree` reduces the tree to value, that is, the sum of all leaf values.

Even though we worked very hard to remove all recursion from these functions,
we still have a problem.
The definition of the `Tree` type was and remains self-referential.
Its `Node` constructor takes two `Tree` values as arguments,
and that makes `Tree` a *recursive data type*.
We did not dare to remove this kind of recursion.
This time, we are more ambitious.
By the end of this article,
we will know how to remove recursion from a data type.

The high-level idea is that
we are going to store our `Tree` in a linear data structure we call a `Tape`.
This will be done in a fashion that
allows us to zoom in on subtrees by slicing the `Tape`.

As usual, we need a few ingredients:

* `Token`s, which are a set of values that are going to represent different pieces of a `Tree`.
* The `Tape`, which is a linear data structure that can be written to and read from and that can be used to represent a whole `Tree` or parts of it.
* A linearizer that can convert a `Tree` to a `Tape` of `Token`s.
* A parser that can convert a `Tape` of `Token`s to a `Tree`.
* A base functor for the `Tree` type that can be used to construct or deconstruct a tree iteratively.
* Lots and lots of boilerplaty Haskell `Generic` code.

We will cover these ingredients in detail in the following sections.
It will take some time to go through all of them.
The slow pace will help you to can get a feel for all this stuff.
We shall now start by defining the `Token` and `Tape` types.
Chocks away!

Token Tapes
-----------

We define our tape as a `newtype` wrapper
around an underlying type constructor, `t :: Type -> Type`:

\begin{code}
  newtype Tape t a = Tape {unTape :: t a}
    deriving stock (Eq, Show)
    deriving newtype
      ( Semigroup,
        Monoid,
        Functor,
        Applicative,
        Monad,
        Alternative,
        Foldable
      )
\end{code}

The type `t` could be `[]`, `Seq`, `Vector`, `Deque`, etc.
It doesn't matter, we won't make a choice at this point.
The only requirement is that there is a way to attach or detach elements
on the left side of `t`.
The `Cons` data class provides a way to formalize this requirement,
and the following code propagates this requirement to the `Tape` type
by means of coercion:

\begin{code}
  instance
    Cons (t a) (t b) a b =>
    Cons (Tape t a) (Tape t b) a b
    where
    _Cons =
      withPrism _Cons $
        \(review' :: (b, t b) -> t b)
         (preview' :: t a -> Either (t b) (a, t a)) ->
            prism (coerce review') (coerce preview')
\end{code}

This class instance gives us a
[prism](https://hackage.haskell.org/package/lens-5.1/docs/Control-Lens-Prism.html#t:Prism)
that can be used to build or deconstruct a `Tape`
via the `cons` and `uncons` functions from
[Control.Lens.Cons](https://hackage.haskell.org/package/lens-5.1/docs/Control-Lens-Cons.html).
They basically work like `(:)` and `uncons` from `Data.List`,
but they are polymorphic in the type `t`
and thus can be used with any `t` that satisfies the `Cons` requirement.

Let's now talk about what we are going to put on the tape.
Our tapes will be made up entirely of `Token`s, to be defined momentarily.
Because of that homogeneity,
it is a good idea to save us some keystrokes and forge a handy type synonym:

\begin{code}
  -- | A tape of tokens.
  type TTape t = Tape t Token
\end{code}

Each `Token` will be used to represent
a piece of information about a particular `Tree`.
For trees with integer leaf nodes, i.e. `Tree Int`,
we will only ever need four `Token`s:

\begin{code}
  data Token
    = -- | Represent a recursive call to an abstract data type.
      Rec Int
    | -- | Represent a left choice between two constructors.
      L
    | -- | Represent a right choice between two constructors.
      R
    | -- | Represent an integer leaf node.
      I Int
    deriving stock (Eq, Show)
\end{code}

I will explain each of these tokens in more detail in a bit.
Their function will become clear as we go along.

Linearization
-------------

Now, how do we turn a `Tree` into a `TTape`?

In general,
we want a function -- let's call it `linearize` --
that turns a value of some type `a` into a tape of tokens without losing any information.
`a` could be any type,
but we explicitly want this to work for `a ~ Tree Int` in the end.

Let's give `linearize` a type signature:

\begin{code}
  type To t a = a -> TTape t
\end{code}

And, because we like to keep things formal, a formal definition:

\begin{code}
  class
    ToTokens
      (t :: Type -> Type)
      (a :: Type)
    where
    linearize :: To t a
\end{code}

This is Haskell.
And, in case you haven't noticed,
the way of Haskell is to make things as general as possible,
sometimes until it hurts.
For that reason,
this class is parameterized not only by 
the type of the values we are going to encode, `a`,
but also by the tape's type parameter, `t`.

To annoy you further,
I will give `linearize` an arcane `default` implementation:

\begin{code}
    default linearize ::
      ( Recursive a,
        ToTokensStep t (Base a)
      ) =>
      To t a
    linearize = cata linearizeStep
\end{code}

This definition uses the accurately named yet mysterious `Recursive` class.
`Recursive` gives us `cata`.
Both of these are defined in
[Data.Functor.Foldable](https://hackage.haskell.org/package/recursion-schemes-5.2.2.2/docs/Data-Functor-Foldable.html).
The `cata` function is a generalization of `fold` and takes two arguments:

* A function that performs one step of a recursive computation. For us, that function is `linearizeStep` which is doing the actual work. It has the type `Base a (TTape t) -> TTape t`.
* The value that needs to be worked on. That value has the type `a`.

With these, `cata` is recursively chewing up the value `a` and turning it into a `TTape t`.
I admit, this machinery is a wee opaque.
I will try my best to explain what is going on.
Stay with me.

Base Functors
-------------

Let's first zoom in on the cryptic type of `linearizeStep`.
This is a function that takes a value of type `Base a (TTape t)`
and gives us back a value of type `TTape t`.
I guess it's clear what comes out of this function (a tape of tokens),
but what in tarnation are we passing here?
What's `Base`, and why is it parameterized by both `a` and our trusty token tape type?

`Base :: Type -> (Type -> Type)`,
as it turns out,
is also coming from
[Data.Functor.Foldable](https://hackage.haskell.org/package/recursion-schemes-5.2.2.2/docs/Data-Functor-Foldable.html).
It is an open type family
and can be thought of as a type-level registry of so-called "base functors".
A registered base functor, `Base a r`, is a non-recursive data type
that is derived for a specific recursive data type, `a`.
The type parameter `r` is used to represent recursion in `a`. How?
Think of it in the following way:
`Base a r` is structurally equal to `a`
except that `r` takes the place of all recursive occurrences of `a` in `a`.

For instance,
the base functor of our `Kont` type from
the [previous installment](/posts/Unrecurse.html) of this series is:

\begin{code}
  -- | A base functor for `Kont`.
  data KontF next r
    = -- | Terminate a computation
      FinishedF
    | -- | Continue a computation with `next`
      MoreF next r
    deriving stock (Eq, Show, Functor)
\end{code}

The `r` type parameter appears exactly
where `Kont next` appears in the original `More` constructor of `Kont next`.
Go back to the definition of `Kont next` and check for yourself
if you don't believe me. Off you pop.

Quick side node on naming.
It is customary to name the base functor and its constructors
after the recursive data type they are associated with (in this case, `Kont`)
except for appending the letter `F` for "functor".
Like the name suggests,
a base functor is always a functor in the type parameter `r`,
and Haskell can derive that instance for us. Neat.

Now, with `KontF` in hand,
we can write the following type family instance:

\begin{code}
  type instance
    Base (Kont next) =
      KontF next
\end{code}

This tells Haskell that the base functor of `Kont` is `KontF`.

How is all this going to help us?

Like we said before,
the argument of `linearizeStep` is of type `Base a r` with `r ~ TTape t`.
If `a` were `Kont next`,
then `Base a r` would be `KontF next (TTape t)`.
And, likewise, if `a` were `Tree Int`,
then `Base a r` would be `TreeF Int (TTape t)`.
That means that `linearizeStep` always works on
a version of `a` where
recursive constructors are replaced with token tapes,
`r ~ TTape t`.

We now understand that `linearizeStep` takes
a special non-recursive version of `a` and that
it is supposed to produce a token tape.
But how should this transformation look like?

Linearization Example
---------------------

Let's dive into a concrete example
and try to understand how things should play out for `a ~ Kont Int`.
This is a bit easier than reaching immediately for trees.

First, consider the base case.
For a finished continuation, `FinishedF`,
our encoding should look like this:

\begin{code}
  -- | A linearized finished continuation.
  -- >>> linearizedFinished
  -- Tape {unTape = [L]}
  linearizedFinished :: TTape []
  linearizedFinished =
    let finished :: KontF Int (TTape []) =
          FinishedF
     in linearizeStep finished
\end{code}

This base case is particularly easy to deal with
since the `FinishedF` constructor has no arguments.
The only information we need to encode is the constructor itself.
I use the token `L` (for "left") to represent `FinishedF`,
because it appears on the left side in the sum type `KontF`.
Thus, the `linearizedFinished` tape should have one element: the token `L`.

Now, let's take a look at the recursive case:
For a continuation with one more step, `MoreF`,
the situation is more complicated, but only slightly so.
I propose the following encoding:

\begin{code}
  -- | A linearized continuation with one more step.
  -- >>> linearizedMore linearizedFinished
  -- Tape {unTape = [R,I 0,Rec 1,L]}
  -- >>> linearizedMore (linearizedMore linearizedFinished)
  -- Tape {unTape = [R,I 0,Rec 4,R,I 0,Rec 1,L]}
  -- >>> linearizedMore (linearizedMore (linearizedMore linearizedFinished))
  -- Tape {unTape = [R,I 0,Rec 7,R,I 0,Rec 4,R,I 0,Rec 1,L]}
  linearizedMore :: TTape [] -> TTape []
  linearizedMore previousTape =
    let more :: KontF Int (TTape []) =
          MoreF 0 previousTape
     in linearizeStep more
\end{code}

I hope the examples make it clear enough that in this encoding:

1. `R` (for "right") is the token for `MoreF`.
2. `I 0` (for "integer") is the token for the first argument of `MoreF`, which is always `0 :: Int` in this contrived example.
3. `Rec _` is the token for the recursive case. Its argument counts the number of tokens needed to encode it. Effectively, this just measures the length of the previous tape we pass to the `linearizedMore` function.

Note how, in the above examples,
calls to `linearizedMore` are nested to create a tape
that encodes progressively more recursive calls to the `MoreF` constructor.
What I have done here manually will in the end be done for us automatically by `linearize`
thanks to the `Recursive` type class and `cata`:

\begin{code}
  -- >>> linearize (Finished :: Kont Int) :: TTape []
  -- Tape {unTape = [L]}
  -- >>> linearize (More 0 $ Finished :: Kont Int) :: TTape []
  -- Tape {unTape = [R,I 0,Rec 1,L]}
  -- >>> linearize (More 0 $ More 0 $ Finished :: Kont Int) :: TTape []
  -- Tape {unTape = [R,I 0,Rec 4,R,I 0,Rec 1,L]}
  -- >>> linearize (More 0 $ More 0 $ More 0 $ Finished :: Kont Int) :: TTape []
  -- Tape {unTape = [R,I 0,Rec 7,R,I 0,Rec 4,R,I 0,Rec 1,L]}
\end{code}

If we had a working implementation of `linearizeStep` already,
then the only thing we would need to do to get this behaviour is to
define an instance of the `Recursive` type class for `Kont next`, like so:

\begin{code}
  instance Recursive (Kont next) where
    project ::
      Kont next ->
      KontF next (Kont next)
    project (More n k) = MoreF n k
    project Finished = FinishedF
\end{code}

This implementation of `project`
tells Haskell how a single layer of a `Kont next` value is unrolled
into a `KontF next (Kont next)` value.
The rest is taken care of by the `cata` function.
I can recommend you to read the newly revised
[documentation](https://hackage.haskell.org/package/recursion-schemes-5.2.2.2#readme-container)
of the recursion schemes package to get an even better understanding
of the principles behind this approach.

Good, we have a more or less clear picture of
how `linearizeStep` is supposed to work.
What's missing is an implementation.
Next up: an implementation.

Generic Stepwise Linearization
------------------------------

We can formally introduce `linearizeStep` like this:

\begin{code}
  class
    ToTokensStep
      (t :: Type -> Type)
      (base :: Type -> Type)
    where
    linearizeStep :: To t (base (TTape t))
\end{code}

Like `ToTokens`, the `ToTokensStep` type class is parameterized by the type of the token tape, `t`.
But instead of the `a` type, we've got another parameter, `base`, for its base functor.

I promised oodles of boilerplate code,
and I am happy to announce that the waiting is over.
We will use
[datatype-generic programming](https://downloads.haskell.org/ghc/latest/docs/html/users_guide/exts/generics.html)
to implement this class!

Have a look at the following `default` implementation:

\begin{code}
    default linearizeStep ::
      ( Alternative t,
        Foldable t,
        Generic (base (TTape t)),
        GToTokensStep t (Rep (base (TTape t)))
      ) =>
      To t (base (TTape t))
    linearizeStep =
      gLinearizeStep
        . GHC.Generics.from
\end{code}

Of course, that's just a wrapper around `gLinearizeStep`,
defined below:

\begin{code}
  class
    GToTokensStep
      (t :: Type -> Type)
      (rep :: Type -> Type)
    where
    gLinearizeStep :: forall a. To t (rep a)
\end{code}

This follows
[the](https://wiki.haskell.org/GHC.Generics#More_general_default_methods)
[usual](https://hackage.haskell.org/package/base-4.16.0.0/docs/GHC-Generics.html#g:13)
[pattern](https://hackage.haskell.org/package/binary-0.8.9.0/docs/Data-Binary.html#t:Binary)
for datatype-generic programming in Haskell.
In particular, this says that,
if our base functor has a `Generic` instance with generic representation
`Rep (base r)`,
then we can obtain a `ToTokensStep` instance 
(and thus `linearizeStep`) for free.
Free is very cheap.

`GHC.Generics.from` will convert a `base r` value into a `Rep (base r)` value.
The latter represents `base r` using only generic primitive types.
These types are defined in the `GHC.Generics` module and are:

* `V1` for impossible values (`Void`). This is used for types that have no constructors. We can't represent `Void` in our token tape.
* `U1` for constructors without arguments like `()` or `Finished`.
* `K1` for constants like `True` or `1`. This is used for constructor arguments. These could be recursive values.
* `M1` for meta data. This is a wrapper and used to encode constructor or data type names.
* `(:*:)` for product types. This is used to separate constructor arguments.
* `(:+:)` for sum types. This is used to encode a choice between two constructors.

If you have never seen these types before,
you may want to read some of the
[documentation](https://hackage.haskell.org/package/base-4.16.0.0/docs/GHC-Generics.html)
in the `GHC.Generics` module.
There are some examples that will help you understand the types
better than I can in this tutorial.

We only need to specify once what should happen for the six generic types.
For `V1`, we can't do anything:

\begin{code}
  instance GToTokensStep t V1 where
    gLinearizeStep v =
      v `seq` error "GToTokensStep.V1"
\end{code}

For `U1`, we can just ignore it and return an empty token tape:

\begin{code}
  instance
    Alternative t =>
    GToTokensStep t U1
    where
    gLinearizeStep _ = Tape empty
\end{code}

For `K1`, we can just delegate to `linearize`:

\begin{code}
  instance
    ToTokens t c =>
    GToTokensStep t (K1 i c)
    where
    gLinearizeStep = linearize . unK1
\end{code}

When specialized to `K1 i Int`,
this instance is used to convert an `Int` constant
appearing in `KontF Int r` into a tape of a single `I` token:

\begin{code}
  instance
    Alternative t =>
    ToTokens t Int
    where
    linearize i = pure (I i)
\end{code}

Moreover,
when specialized to `K1 i (TTape t)`,
the `K1` instance defines
what should happen for the `TTape t` constants
in `KontF next (TTape t)`.
This is the trick that allows us to deal with recursive constructor arguments:

\begin{code}
  instance
    (Alternative t, Foldable t) =>
    ToTokens t (TTape t)
    where
    linearize tape =
      pure (Rec $ length tape) <|> tape
\end{code}

Here we use `length` to measure the length of the tape.
We store that length in a `Rec` token
that we prepend to the tape using `(<|>)`.
This length information will be helpful later
when we want to decode the tape back into a value.

For `M1`, we can just unwrap the constructor:

\begin{code}
  instance
    GToTokensStep t f =>
    GToTokensStep t (M1 i c f)
    where
    gLinearizeStep = gLinearizeStep . unM1
\end{code}

For the product `(f :*: g)`,
we can delegate to the `GToTokensStep` instances of `f` and `g`:

\begin{code}
  instance
    ( Alternative t,
      Foldable t,
      GToTokensStep t f,
      GToTokensStep t g
    ) =>
    GToTokensStep t (f :*: g)
    where
    gLinearizeStep (x :*: y) =
      gLinearizeStep x <|> gLinearizeStep y
\end{code}

The tapes of the two `x :: f a` and `y :: g a` values are concatenated using `(<|>)`.

Finally, we can define an instance for the sum `(f :+: g)`:

\begin{code}
  instance
    ( Applicative t,
      Alternative t,
      Foldable t,
      GToTokensStep t f,
      GToTokensStep t g
    ) =>
    GToTokensStep t (f :+: g)
    where
    gLinearizeStep (L1 x) =
      pure L <|> gLinearizeStep x
    gLinearizeStep (R1 x) =
      pure R <|> gLinearizeStep x
\end{code}

We use `pure L` and `pure R` to encode the left and right constructor.

This concludes the definition of `GToTokensStep`
and the boilerplaty datatype-generic programming exercise for `ToTokensStep`.
Wasn't that fun?
There is more to come.

Auto-Generating `ToTokens` Instances
------------------------------------

Perhaps this was lost in the noise,
but we can now automatically generate `ToTokens` instances!

For the `Kont` data type, this is done in three steps:

Step 1: Ask Haskell to generate a `Generic` instance for `Kont`'s base functor, `KontF`.

\begin{code}
  deriving stock instance Generic (KontF next r)
\end{code}

Step 2: Obtain a `ToTokensStep` instance from the default implementation.

\begin{code}
  instance
    (Alternative t, Foldable t, ToTokens t next) =>
    ToTokensStep t (KontF next)
\end{code}

Step 3: Earn a `ToTokens` instance.

\begin{code}
  instance
    ToTokensStep t (KontF next) =>
    ToTokens t (Kont next)
\end{code}

With these we can convert a `Kont next` value into a `TTape t` value
(if we also happen to have a `ToTokens` instance for `next`).
And we know that this is true because this is a literate Haskell article,
and all previously seen examples were in fact already working.
Surprise!

Originally,
we were interested in values of type `Tree Int`.
Perhaps you remember.
Are we any closer to linearizing those, too?
We are. We can automagically generate now everything we need.

We defined the base functor `KontF` for the `Kont` data type manually.
This was a bit tedious,
but it helped us understand base functor types.
Now, rather than going through the trouble of writing our own base functor for `Tree`
(or any other data type `a`),
we can use `makeBaseFunctor` to do this for us.
`makeBaseFunctor` is
a [Template Haskell](https://en.wikipedia.org/wiki/Template_Haskell) function
that generates the base functor for the `Tree` type and calls it `TreeF`.

\begin{code}
  makeBaseFunctor ''Tree
\end{code}

This little trick also generates `Base` and `Recursive` instances for `Tree`,
among a few other things that we don't need to worry about right now.

However, we don't get a `Show` or `Generic` instance for `TreeF`,
so let's quickly add those:

\begin{code}
  deriving stock instance (Show a, Show r) => Show (TreeF a r)

  deriving stock instance Generic (TreeF a r)
\end{code}

The `Generic` instance opens up the possibility of auto-generating
the `ToTokens` instance for `Tree`:

\begin{code}
  instance
    (Alternative t, Foldable t, ToTokens t a) =>
    ToTokensStep t (TreeF a)

  instance
    ToTokensStep t (TreeF a) =>
    ToTokens t (Tree a)
\end{code}

And that's it!
Let's see what we can do with this:

\begin{code}
  -- >>> linearize (Nil :: Tree Int) :: TTape []
  -- Tape {unTape = [L]}
  -- >>> linearize (Node Nil 0 Nil :: Tree Int) :: TTape []
  -- Tape {unTape = [R,Rec 1,L,I 0,Rec 1,L]}
  -- >>> linearize (Node (Node Nil 0 Nil) 1 (Node Nil 2 Nil) :: Tree Int) :: TTape []
  -- Tape {unTape = [R,Rec 6,R,Rec 1,L,I 0,Rec 1,L,I 1,Rec 6,R,Rec 1,L,I 2,Rec 1,L]}
  -- >>> linearize exampleTree :: TTape []
  -- Tape {unTape = [R,Rec 16,R,Rec 6,R,Rec 1,L,I 1,Rec 1,L,I 2,Rec 6,R,Rec 1,L,I 3,Rec 1,L,I 4,Rec 16,R,Rec 6,R,Rec 1,L,I 5,Rec 1,L,I 6,Rec 6,R,Rec 1,L,I 7,Rec 1,L]}
\end{code}

There you have it,
we can flatten binary trees and store them in tapes of tokens.
Cool stuff!

Parsing Tapes of Tokens
-----------------------

How can we go back from a `TTape t` value to a `Tree` value?

The answer is *parsing*.
[Many](https://hackage.haskell.org/package/parsec)
[parsing](https://hackage.haskell.org/package/megaparsec)
[libraries](https://hackage.haskell.org/package/attoparsec)
[exist](https://hackage.haskell.org/package/trifecta)
for Haskell,
but we will use none of them,
because we need a lot less than what they offer.
Instead, we will use a minimal approach to parsing
based on the good old state monad transformer, `StateT`.
We know it well from the [previous article](/posts/Unrecurse.html).

It is a little-known fact that `StateT`
already provides all that we need to implement a
[monadic parser](http://www.cs.nott.ac.uk/~pszgmh/pearl.pdf).
It even supports backtracking.
This may be surprising,
since `StateT s b a` is just a `newtype` wrapper around
`s -> b (a, s)`,
where `s` is the state's type,
and `b` is the type of some inner monad.
Why should this matter for parsing?
Well, that's because, at its most fundamental level,
*a parser for things `a` is a function
from strings `s` to lists `b ~ []` of pairs `(a, s)` of things and strings*.
That's a little Seussian rhyme I borrowed from 
[Fritz Ruehr](http://www.willamette.edu/~fruehr/haskell/seuss.html).
It means that if we have a string `s` and a parser `StateT s b a` with `b ~ []`,
then running the parser on `s` will return:

* an empty list if there is no way to create an `a` from any prefix of the input string `s` (including the empty string) or
* a non-empty list of full and/or partial parses of `s`, where each pair in the list belongs to one alternative parse of `s`. The first part of a pair is the parsing result, `a`, and the second part is the unconsumed remainder of the input string.

There may be a very long list of alternatives, but for `b ~ []` those are lazily evaluated.
This is why we can think of `StateT s [] a` as a parser with backtracking.
If we don't want backtracking, we can use `StateT s Maybe a` instead.
Then we will only ever get zero or one parse. If we get `Nothing`, the parse failed. If we get `Just`, the parse succeeded.
For `b ~ Maybe`, we can never explore more than one alternative.
We are greedily parsing, and committing to the first alternative that succeeds is a final decision.
`b` (for "backtracking") should always be a monad with a `MonadPlus` instance for supporting choice (`mplus`) and failure (`mzero`).
`[]`, `Maybe`, and `LogicT` from [Control.Monad.Logic](https://hackage.haskell.org/package/logict)
fulfil this requirement, but there are many monads that do not.

In Haskell, a string is a list of characters.
Here, we have a tape of tokens.
If we want to parse a tape of tokens,
then we should be able to do that with this state monad transformer:

\begin{code}
  type From b t a = StateT (TTape t) b a
\end{code}

This is the counterpart to `To t a` that we have been using to flatten trees into tapes of tokens.
To go the other way, we need to define a value of type `From b t a`.
It will need to be made such that it is compatible with how we defined `To t a` above
and undoes the flattening we engineered there.
We will build this value from the ground up starting with simplest parser we can write down:

\begin{code}
  -- | A parser that consumes a single token from the tape and returns it.
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

This parser just tries to take the first token from the tape and yields it,
no matter what the token is.
If there are no tokens left, it fails.
The `MonadFail` constraint is needed for the `fail` function,
and the `Cons` constraint is needed for the `uncons` function.

The second most simple parser we can write is one that consumes a single token and
returns it only if it matches a given predicate:

\begin{code}
  -- | A parser that matches a given token and returns it.
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

The `mfilter` function is a monadic version of `filter` and provided by the `MonadPlus` requirement.

These two parsers, `token` and `isToken`, will turn out to be everything we need.
We will use *combinator functions* to compose them again and again
until we get to the final parser that solves our problem.
The combinators will mostly be provided by the `Alternative` and `MonadPlus` instances for `From b t`.
This will become much clearer in the next section.
It's all about the combinators from here.
There is [documentation](https://en.wikibooks.org/wiki/Haskell/Alternative_and_MonadPlus) on the subject
for those who are interested, but it should not be necessary to read this to understand the rest of this article.

There And Back Again
--------------------








\begin{code}
  class
    FromTokensStep
      (b :: Type -> Type)
      (t :: Type -> Type)
      (base :: Type -> Type)
    where
    parseStep :: From b t (base (TTape t))
    default parseStep ::
      ( Functor b,
        Generic (base (TTape t)),
        GFromTokensStep b t (Rep (base (TTape t)))
      ) =>
      From b t (base (TTape t))
    parseStep = to <$> gParseStep

  class
    GFromTokensStep
      (b :: Type -> Type)
      (t :: Type -> Type)
      (rep :: Type -> Type)
    where
    gParseStep :: forall a. From b t (rep a)

  instance
    MonadFail b =>
    GFromTokensStep b t V1
    where
    gParseStep = fail "GFromTokensStep.V1"

  instance
    Monad b =>
    GFromTokensStep b t U1
    where
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
    gParseStep =
      (isToken L >> L1 <$> gParseStep)
        <|> (isToken R >> R1 <$> gParseStep)

  instance
    ( MonadFail b,
      MonadPlus b,
      Cons (TTape t) (TTape t) Token Token,
      GFromTokensStep b t f,
      GFromTokensStep b t g
    ) =>
    GFromTokensStep b t (f :*: g)
    where
    gParseStep =
      (:*:)
        <$> gParseStep
        <*> gParseStep

  instance
    (Monad b, FromTokens b t c) =>
    GFromTokensStep b t (K1 i c)
    where
    gParseStep = K1 <$> parse

  instance
    (Functor b, GFromTokensStep b t f) =>
    GFromTokensStep b t (M1 i c f)
    where
    gParseStep = M1 <$> gParseStep
\end{code}

\begin{code}
  resetParse :: Monad b => 
    From b t a -> TTape t -> From b t a
  resetParse m = lift . evalStateT m

  class
    FromTokens
      (b :: Type -> Type)
      (t :: Type -> Type)
      (a :: Type)
    where
    parse :: From b t a
    default parse ::
      ( Corecursive a,
        Monad b,
        Traversable (Base a),
        FromTokensStep b t (Base a)
      ) =>
      From b t a
    parse = go
      where
        go =
          fmap embed $
            parseStep
              >>= traverse (resetParse go)
\end{code}

\begin{code}
  instance
    ( MonadFail b,
      MonadPlus b,
      Cons (t Token) (t Token) Token Token,
      Alternative t,
      FromTokens b t a
    ) =>
    FromTokensStep b t (TreeF a)

  instance
    (Monad b, FromTokensStep b t (TreeF a)) =>
    FromTokens b t (Tree a)
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
