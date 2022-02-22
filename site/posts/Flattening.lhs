---
title: "Flattening -- How to Get from A Tree to A Flat Shape And Back Again"
date: Feb 2, 2022
teaser: >
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
  {-# LANGUAGE InstanceSigs #-}
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

The Flattening of A Tree
------------------------

[Previously](/posts/Unrecurse.html),
we have seen how one can remove recursive calls from a function.
We learned about continuations, defunctionalization, and monadic `State` effects.
We used these techniques to reimplement two simple recursive functions,
`printTree` and `accumTree`, using only iteration.
These functions are both specific examples of a `fold`.
They consume a value of type `Tree`,
a data type for binary trees with two constructors, `Nil` and `Node`.
The `Tree` data structure is reduced bit by bit
to an effect (in the case of `printTree`) or a value (in the case of `accumTree`).

Even though we worked very hard,
our attempts to remove all recursion from our program were incomplete:
the `Tree` type was and remains recursively defined.
Its `Node` constructor takes two `Tree` values as arguments
which makes `Tree` a recursive data type.
Last time, we did not dare to remove this kind of recursion.
This time, we are more ambitious. By the end of this article,
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

We shall start by defining the `Token` and `Tape` types.

Token Tapes
-----------

We define the tape as a `newtype` wrapper
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
We won't make a choice at this point.
The only requirement is that there is a way to attach or detach elements
on the left side of `t`.
The `Cons` data class provides a way to formalize this requirement,
and the following code propagates this requirement to the `Tape` type:

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

Let's now talk about what we are going to put on the tape.
Our tapes will be made up of `Token`s.
Thus, it is a good idea to save us some keystrokes and use a type synonym:

\begin{code}
  -- | A tape of tokens.
  type TTape t = Tape t Token
\end{code}

Each `Token` will be used to represent
a piece of information about a particular `Tree`.
For trees with integer leaf nodes, `Tree Int`,
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

I will explain each of these tokens in more detail later.

Linearization
-------------

How do we turn a `Tree` into a `Tape`?

We want a function, let's call it `linearize`,
that returns a lossless encoding
of values of some type `a` into a tape of tokens, `TTape t`.
`a` could be any type,
but we want this to work for `a ~ Tree Int` in the end.
Therefore, our `linearize` will have the type:

\begin{code}
  type To t a = a -> TTape t
\end{code}

We can formalize this by defining a type class:

\begin{code}
  class
    ToTokens
      (t :: Type -> Type)
      (a :: Type)
    where
    linearize :: To t a
\end{code}

To make this as general as possible,
this class is parameterized by the tape's type parameter, `t`,
and the type of the values we are going to encode, `a`.

`linearize` has a `default` implementation that uses
the accurately named yet mysterious `Recursive` class:

\begin{code}
    default linearize ::
      ( Recursive a,
        ToTokensStep t (Base a)
      ) =>
      To t a
    linearize = cata linearizeStep
\end{code}

`Recursive` gives us `cata`
which we use to recursively encode values of type `a` into `TTape t`.
Both, `Recursive` and `cata`, are defined in
[Data.Functor.Foldable](https://hackage.haskell.org/package/recursion-schemes-5.2.2.2/docs/Data-Functor-Foldable.html).
`cata` is a generalization of `fold` and takes two arguments:

* A function that performs one step of a recursive computation. Here, that function is `linearizeStep`. It has the type `Base a (TTape t) -> TTape t`.
* The value that needs to be worked on. That value has the type `a`.

With these, `cata` returns a value of type `TTape t`.
This machinery is a bit opaque.
I will try my best to explain what is going on.

Base Functors
-------------

Let's zoom in on the cryptic type of `linearizeStep`.
It is a function that takes a value of type `Base a (TTape t)`
and returns a value of type `TTape t`.
I guess it's clear what we get back, a tape of tokens.
But what exactly are we passing here?
`Base :: Type -> (Type -> Type)` is defined in
[Data.Functor.Foldable](https://hackage.haskell.org/package/recursion-schemes-5.2.2.2/docs/Data-Functor-Foldable.html).
It is an open type family
and can be thought of as a registry of so-called "base functors".
A base functor, `Base a r`, is a data type that is derived
for a specific recursive data type, `a`.
The type parameter `r` is used to represent recursion in `a`.

The data type `Base a r` is structurally equal to `a`
except that `r` takes the place of all recursive occurrences of `a` in `a`.

For instance,
the base functor of our `Kont` type from the [previous article](/posts/Unrecurse.html) is:

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
if you don't believe me.

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
`r ~ TTape t`:

Linearization Example
---------------------

To get an idea of what `linearizeStep` actually does,
let's look at how things play out for `a ~ Kont Int`.

First, the base case.
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

The base case is particularly easy to deal with
since the `FinishedF` constructor has no arguments.
The only information we need to encode is the constructor itself.
We use the token `L` (for "left") to represent `FinishedF`,
because it appears on the left side in the sum type `KontF`.
Thus, the `linearizedFinished` tape has one element: the token `L`.

Now, let's look at the recursive case:
For a continuation with one more step, `MoreF`,
the situation is more complicated, but only slightly so.

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

I hope the examples make it clear enough that:

1. `R` (for "right") is the token for `MoreF`.
2. `I 0` (for "integer") is the token for the first argument of `MoreF`, which is always `0 :: Int` in this contrived example.
3. `Rec _` is the token for the recursive case. Its argument counts the number of tokens needed to encode it. This just measures the length of the previous tape.

Note how, in the above examples,
calls to `linearizedMore` are nested to create a tape
that encodes progressively more recursive calls to the `MoreF` constructor.
What we have done here manually is done for us automatically by `linearize`
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

If we had `linearizeStep` already,
then the only thing we would need to do to get this behaviour is to
define an instance of the `Recursive` class for `Kont next`, like so:

\begin{code}
  instance Recursive (Kont next) where
    project ::
      Kont next ->
      KontF next (Kont next)
    project (More n k) = MoreF n k
    project Finished = FinishedF
\end{code}

`project` tells Haskell how a single layer of a `Kont next` value is unrolled
into a `KontF next (Kont next)` value.
The rest is taken care of by the `cata` function.

What's missing is an implementation of `linearizeStep`.

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

The `ToTokensStep` type class is parameterized by the type of the token tape, `t`,
and the base functor of the recursive data type, `base`.
We will use datatype-generic programming to implement this class:

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

This `default` implementation is just a wrapper around `gLinearizeStep`,
defined below:

\begin{code}
  class
    GToTokensStep
      (t :: Type -> Type)
      (rep :: Type -> Type)
    where
    gLinearizeStep :: forall a. To t (rep a)
\end{code}

This follows the usual pattern for datatype-generic programming in Haskell.
In particular, this says that,
if our base functor has a `Generic` instance with generic representation
`Rep (base r)`,
then we can obtain a `ToTokensStep` instance 
(and thus `linearizeStep`) for free.

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

For `U1`, we can just ignore it:

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
what should happen for a `TTape t` constant
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
that we prepend to the tape using `<|>`.
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
we can just delegate to the `GToTokensStep` instances of `f` and `g`:

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

The tapes of the two `x :: f a` and `y :: g a` values are concatenated using `<|>`.

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
and the datatype-generic programming exercise for `ToTokensStep`.

Auto-Generating `ToTokens` Instances
------------------------------------

We can now obtain `ToTokens` instances for all base functors.

Let's start with the base functor `KontF` for the `Kont` data type:

First, we need to ask Haskell to generate a `Generic` instance for `KontF`:

\begin{code}
  deriving stock instance Generic (KontF next r)
\end{code}

Then, we can obtain a `ToTokensStep` instance from the default implementation:

\begin{code}
  instance
    (Alternative t, Foldable t, ToTokens t next) =>
    ToTokensStep t (KontF next)
\end{code}

Finally, we can use this instance to obtain a `ToTokens` instance:

\begin{code}
  instance
    ToTokensStep t (KontF next) =>
    ToTokens t (Kont next)
\end{code}

Now we can linearize a `Kont next` value into a `TTape t` value
(if we have a `ToTokens` instance for `next`).

This is nice,
but we were originally interested in values of type `Tree Int`.
Are we any closer to linearizing those?
We are. We can automagically generate now everything we need.

We defined the base functor `KontF` for the `Kont` data type manually.
This was a bit tedious, but it helped us understand base functor types.
Now, rather than going through the trouble of writing our own base functor for `Tree`,
we can use `makeBaseFunctor` to do it for us.
`makeBaseFunctor` is a [Template Haskell](https://en.wikipedia.org/wiki/Template_Haskell) function
that generates the base functor for the `Tree` type and calls it `TreeF`.

\begin{code}
  makeBaseFunctor ''Tree
\end{code}

This also generates `Base` and `Recursive` instances for `Tree`,
among a few other things that we don't need to worry about right now.

However, this doesn't generate a `Show` or `Generic` instance for `TreeF`,
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

Parsing Tapes of Tokens
-----------------------

How can we go back from a `TTape t` value to a `Tree` value?

The answer is parsing.
[Many](https://hackage.haskell.org/package/parsec)
[parsing](https://hackage.haskell.org/package/megaparsec)
[libraries](https://hackage.haskell.org/package/attoparsec)
[exist](https://hackage.haskell.org/package/trifecta)
for Haskell,
but we will use none of them,
because we need much less than what they offer.
Instead, we will use a minimal approach based on the state monad transformer, `StateT`.

The counterpart to `To t a` is:

\begin{code}
  type From b t a = StateT (TTape t) b a
\end{code}



There And Back Again
--------------------





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


With this, we can auto-derive a bunch of instances for the `TreeF` and `Tree` types:

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
