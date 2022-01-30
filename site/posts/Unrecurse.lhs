---
title: "Unrecurse"
date: Jan 20, 2022
teaser: Let's play with recursion!
tags:
  items: [haskell]
---

> {-# LANGUAGE RankNTypes #-}
> {-# LANGUAGE TypeApplications #-}
> {-# LANGUAGE GADTs #-}
> {-# LANGUAGE RecordWildCards #-}
> {-# LANGUAGE DeriveTraversable #-}
> {-# LANGUAGE TemplateHaskell #-}
> {-# LANGUAGE TypeFamilies #-}
> {-# LANGUAGE ScopedTypeVariables #-}
> {-# LANGUAGE BangPatterns #-}
> {-# LANGUAGE FlexibleContexts #-}
> {-# LANGUAGE MultiParamTypeClasses #-}
> {-# LANGUAGE DefaultSignatures #-}
> {-# LANGUAGE FlexibleInstances #-}
> {-# LANGUAGE TypeOperators #-}
> {-# LANGUAGE UndecidableInstances #-}
> {-# LANGUAGE DeriveGeneric #-}
> {-# LANGUAGE DerivingVia #-}
> {-# LANGUAGE GeneralizedNewtypeDeriving #-}
> {-# LANGUAGE StandaloneDeriving #-}
> {-# LANGUAGE LambdaCase #-}
> {-# OPTIONS_GHC -Wno-incomplete-patterns #-}

> module Unrecurse where

> import Prelude hiding (even, odd)
> import Control.Applicative (Alternative((<|>), empty))
> import Control.Lens (zoom, _1, _2, uncons, Cons(_Cons), from, bimapping, prism, Prism, withPrism, cons)
> import Control.Monad.Cont (ContT, MonadTrans (lift))
> import Control.Monad.State (MonadState (get, put), modify, StateT (runStateT, StateT), evalStateT, execStateT)
> import Control.Monad (void, MonadPlus, mfilter)
> import Data.Functor.Identity (Identity)
> import Data.Kind (Type)
> import GHC.Generics (Generic (Rep, from, to), K1 (unK1, K1), M1 (unM1, M1), type (:+:) (L1, R1), type (:*:) ((:*:)), V1, U1 (U1), Generic1 (Rep1, to1), Par1, Rec1)
> import Control.Monad.Writer (Writer, tell)
> import Data.Functor.Foldable.TH (makeBaseFunctor)
> import Data.Functor.Foldable (Base, Recursive (cata), Corecursive (ana, embed))
> import Data.Coerce (coerce)
> import Control.Monad.Trans.Maybe (runMaybeT)
> import Data.Maybe (fromJust)
> import Data.Monoid (Sum (..))
  
Let's do some recursion.

> even :: Int -> Bool
> even 0 = True
> even n = odd (n - 1)

> odd :: Int -> Bool
> odd 0 = False
> odd n = even (n - 1)

<https://www.haskellforall.com/2012/06/you-could-have-invented-free-monads.html>
<https://www.tweag.io/blog/2018-02-05-free-monads/>
<https://gist.github.com/eamelink/4466932a11d8d92a6b76e80364062250>

> data Trampoline f r = 
>     Trampoline { bounce :: f (Trampoline f r) }
>   | Done { result :: r }

> instance Functor f => Functor (Trampoline f) where
>   fmap f (Trampoline m) = Trampoline (fmap (fmap f) m)
>   fmap f (Done r) = Done (f r)

> instance Functor f => Applicative (Trampoline f) where
>   pure = Done
>   Done f <*> Done x = Done $ f x
>   Done f <*> Trampoline mx = Trampoline $ fmap f <$> mx
>   Trampoline mf <*> x = Trampoline $ fmap (<*> x) mf

> instance Functor f => Monad (Trampoline f) where
>   return = Done
>   Done x >>= f = f x
>   Trampoline mx >>= f = Trampoline (fmap (>>= f) mx)

> liftF :: Functor f => f r -> Trampoline f r
> liftF m = Trampoline (fmap Done m)

> data EvenOddF next =
>     Even Int (Bool -> next)
>   | Odd Int (Bool -> next)
>   deriving stock Functor

> -- instance Functor EvenOddF where
> --   fmap f (Even n k) = Even n (f . k)
> --   fmap f (Odd n k) = Odd n (f . k)

> type EvenOdd = Trampoline EvenOddF

> even' :: Int -> EvenOdd Bool
> even' 0 = Done True
> even' n = liftF (Odd (n - 1) id)

> odd' :: Int -> EvenOdd Bool
> odd' 0 = Done False
> odd' n = liftF (Even (n - 1) id)

> evenOddHandler :: EvenOddF (EvenOdd r) -> EvenOdd r
> evenOddHandler (Even n k) = even' n >>= k
> evenOddHandler (Odd n k) = odd' n >>= k

> iterTrampoline :: Functor f => (f (Trampoline f r) -> Trampoline f r) -> Trampoline f r -> r
> iterTrampoline h = go
>   where go Done {..} = result
>         go Trampoline {..} = go (h bounce)

> runEvenOdd :: EvenOdd r -> r
> runEvenOdd = iterTrampoline evenOddHandler

<https://stackoverflow.com/questions/57733363/how-to-adapt-trampolines-to-continuation-passing-style>

> fib :: Int -> Int
> fib n = go n 0 1
>   where go !n !a !b | n == 0    = a
>                     | otherwise = go (n - 1) b (a + b)

> data FibF next =
>     FibF Int Int Int (Int -> next)
>   deriving stock Functor

> type Fib = Trampoline FibF

> fib' :: Int -> Fib Int
> fib' n = liftF (FibF n 0 1 id)

> fibHandler :: FibF (Fib r) -> Fib r
> fibHandler (FibF 0 a _ k) = Done a >>= k
> fibHandler (FibF n a b k) = liftF (FibF (n - 1) b (a + b) id) >>= k

> runFib :: Fib Int -> Int
> runFib = iterTrampoline fibHandler

<https://www.pathsensitive.com/2019/07/the-best-refactoring-youve-never-heard.html>

> data Tree a = Nil | Node {left :: Tree a, content :: a, right :: Tree a}
>   deriving stock (Eq, Show, Functor, Generic)

> makeBaseFunctor ''Tree

> deriving stock instance (Show a, Show r) => Show (TreeF a r)
> deriving stock instance Generic (TreeF a r)

> exampleTree :: Tree Int
> exampleTree = Node (Node (Node Nil 1 Nil) 2 (Node Nil 3 Nil)) 4 (Node (Node Nil 5 Nil) 6 (Node Nil 7 Nil))

> printTree :: forall a. Show a => Tree a -> IO ()
> printTree Nil = pure ()
> printTree Node {..} = do
>   printTree left
>   print content
>   printTree right

https://hexagoxel.de/postsforpublish/posts/2018-09-09-cont-part-one.html

> printTree' :: forall a r. Show a => Tree a -> ContT r IO ()
> printTree' Nil = pure ()
> printTree' Node {..} = do
>   printTree' left
>   lift $ print content
>   printTree' right

> printTree'' :: forall a r. Show a => Tree a -> (() -> IO r) -> IO r
> printTree'' Nil = \c -> c ()
> printTree'' Node {..} =
>    let first = \c -> printTree'' left c
>        second = \c -> print content >>= c
>        third = \c -> printTree'' right c
>        inner = \c -> second (\x -> (\() -> third) x c)
>        outer = \c -> first (\x -> (\() -> inner) x c)
>    in outer

```haskell
printTree'' exampleTree (\() -> pure ()) :: IO ()
```

> data Kont a = First (Tree a) (Kont a) | Second a (Kont a) | Third (Tree a) (Kont a) | Finished
>   deriving stock Functor

> apply :: forall a. Show a => Kont a -> IO ()
> apply (First left c) = printTree''' left c
> apply (Second content c) = print content >> apply c
> apply (Third right c) = printTree''' right c
> apply Finished = pure ()

> printTree''' :: forall a. Show a => Tree a -> Kont a -> IO ()
> printTree''' Nil c = apply c
> printTree''' Node {..} c = apply (First left (Second content (Third right c)))

```haskell
printTree''' exampleTree Finished :: IO ()
```

> printTree'''' :: forall a. Show a => Tree a -> Kont a -> IO ()
> printTree'''' Nil (First left c) = printTree'''' left c
> printTree'''' Nil (Second content c) = print content >> printTree'''' Nil c
> printTree'''' Nil (Third right c) = printTree'''' right c
> printTree'''' Nil Finished = pure ()
> printTree'''' Node {..} c = printTree'''' Nil (First left (Second content (Third right c)))

> data Next a = First' (Tree a) | Second' a | Third' (Tree a)
> type Stack a = [a]

> push :: forall a m. MonadState (Stack a) m => a -> m ()
> push x = modify (x:)

> pop :: forall a m. MonadState (Stack a) m => m (Maybe a)
> pop = do
>   stack <- get
>   case stack of
>     [] -> pure Nothing
>     (x:xs) -> do
>       put xs
>       pure (Just x)

> printTree''''' :: forall a. Show a => Tree a -> StateT (Stack (Next a)) IO ()
> printTree''''' Nil = do
>   c <- pop
>   case c of
>     Just (First' left) -> printTree''''' left
>     Just (Second' content) -> lift (print content) >> printTree''''' Nil
>     Just (Third' right) -> printTree''''' right
>     Nothing -> pure ()
> printTree''''' Node {..} = do
>   push (Third' right)
>   push (Second' content)
>   push (First' left)
>   printTree''''' Nil

```haskell
runStateT (printTree''''' exampleTree) [] :: IO ()
```

> printTree'''''' :: forall a. Show a => StateT (Tree a, Stack (Next a)) IO ()
> printTree'''''' = do
>   tree <- zoom _1 get
>   case tree of
>     Nil -> do
>       c <- zoom _2 pop
>       case c of
>         Just (First' left) -> do
>           zoom _1 $ put left 
>           printTree''''''
>         Just (Second' content) -> do
>           lift (print content)
>           zoom _1 $ put Nil
>           printTree''''''
>         Just (Third' right) -> do
>           zoom _1 $ put right
>           printTree''''''
>         Nothing -> pure ()
>     Node {..} -> do
>       zoom _2 $ push (Third' right)
>       zoom _2 $ push (Second' content)
>       zoom _2 $ push (First' left)
>       zoom _1 $ put Nil
>       printTree''''''

```haskell
runStateT printTree'''''' (exampleTree, []) :: IO ()
```

> data Continue = Continue | Break

> while :: forall m. Monad m => m Continue -> m ()
> while m = do
>   c <- m
>   case c of
>     Continue -> while m
>     Break -> pure ()

> printTree''''''' :: forall a. Show a => StateT (Tree a, Stack (Next a)) IO ()
> printTree''''''' =
>   while $ do
>     tree <- zoom _1 get
>     case tree of
>       Nil -> do
>         c <- zoom _2 pop
>         case c of
>           Just (First' left) -> do
>             zoom _1 $ put left
>             pure Continue
>           Just (Second' content) -> do
>             lift (print content)
>             zoom _1 $ put Nil
>             pure Continue
>           Just (Third' right) -> do
>             zoom _1 $ put right
>             pure Continue
>           Nothing -> pure Break
>       Node {..} -> do
>         zoom _2 $ push (Third' right)
>         zoom _2 $ push (Second' content)
>         zoom _2 $ push (First' left)
>         zoom _1 $ put Nil
>         pure Continue

```haskell
runStateT printTree''''''' (exampleTree, []) :: IO ()
```

> accumTree :: forall a. Monoid a => Tree a -> a
> accumTree Nil = mempty
> accumTree Node {..} =
>   let lacc = accumTree left
>       racc = accumTree right
>   in lacc <> content <> racc

> accumTree' :: forall a. Monoid a => Tree a -> Writer a ()
> accumTree' Nil = pure ()
> accumTree' Node {..} = do
>   accumTree' left
>   tell content
>   accumTree' right

```haskell
execWriter $ accumTree' (Sum <$> exampleTree)
```

> accumTree''''''' :: forall a. Monoid a => StateT (Tree a, Stack (Next a)) (Writer a) ()
> accumTree''''''' =
>   while $ do
>     tree <- zoom _1 get
>     case tree of
>       Nil -> do
>         c <- zoom _2 pop
>         case c of
>           Just (First' left) -> do
>             zoom _1 $ put left
>             pure Continue
>           Just (Second' content) -> do
>             lift (tell content)
>             zoom _1 $ put Nil
>             pure Continue
>           Just (Third' right) -> do
>             zoom _1 $ put right
>             pure Continue
>           Nothing -> pure Break
>       Node {..} -> do
>         zoom _2 $ push (Third' right)
>         zoom _2 $ push (Second' content)
>         zoom _2 $ push (First' left)
>         zoom _1 $ put Nil
>         pure Continue

```haskell
execWriter $ runStateT accumTree''''''' (Sum <$> exampleTree, [])
```

> data Token =
>   Rec Int | L | R | I Int
>   deriving stock (Eq, Show)

> newtype Tape t a = Tape { unTape :: t a }
>   deriving stock (Eq, Show)
>   deriving newtype (Functor, Applicative, Monad, Alternative, Foldable)

> instance Cons (t a) (t b) a b => Cons (Tape t a) (Tape t b) a b where
>   _Cons = withPrism cons' $ \review' preview' ->
>     prism (coerce review') (coerce preview')
>     where
>       cons' :: Prism (t a) (t b) (a, t a) (b, t b)
>       cons' = _Cons

> type TTape t = Tape t Token

> type To t a = a -> TTape t

> type From b t a = StateT (TTape t) b a

> token :: forall b t.
>   ( MonadFail b,
>     Cons (TTape t) (TTape t) Token Token
>   ) => From b t Token
> token = do
>   t <- get
>   case uncons t of
>     Nothing -> fail "unexpected end of input"
>     Just (x, xs) -> put xs >> pure x

> isToken :: forall b t.
>   ( MonadFail b,
>     MonadPlus b,
>     Cons (TTape t) (TTape t) Token Token
>   ) => Token -> From b t Token
> isToken t = mfilter (== t) token

> class ToTokens (t :: Type -> Type) (a :: Type) where
>   linearize :: To t a
>   default linearize :: (Recursive a, ToTokensStep t (Base a)) => To t a
>   linearize = cata linearizeStep

> instance (Alternative t) => ToTokens t Int where
>   linearize i = pure (I i)

> instance (Alternative t, Foldable t) => ToTokens t (TTape t) where
>   linearize tape = pure (Rec $ length tape) <|> tape

> class ToTokensStep (t :: Type -> Type) (base :: Type -> Type) where
>   linearizeStep :: To t (base (TTape t))
>   default linearizeStep :: (Alternative t, Foldable t, Generic (base (TTape t)), GToTokensStep t (Rep (base (TTape t)))) => To t (base (TTape t))
>   linearizeStep = gLinearizeStep . GHC.Generics.from

> instance (Alternative t, Foldable t, ToTokens t a) => ToTokensStep t (TreeF a)

> instance (ToTokensStep t (TreeF a)) => ToTokens t (Tree a)

> class GToTokensStep (t :: Type -> Type) (f :: Type -> Type) where
>   gLinearizeStep :: forall a. To t (f a)

> instance GToTokensStep t V1 where
>   gLinearizeStep v = v `seq` error "GToTokensStep.V1"

> instance Alternative t => GToTokensStep t U1 where
>   gLinearizeStep _ = Tape empty

> instance (Applicative t, Alternative t, Foldable t, GToTokensStep t f, GToTokensStep t g) => GToTokensStep t (f :+: g) where
>   gLinearizeStep (L1 x) = pure L <|> gLinearizeStep x
>   gLinearizeStep (R1 x) = pure R <|> gLinearizeStep x

> instance (Alternative t, Foldable t, GToTokensStep t f, GToTokensStep t g) => GToTokensStep t (f :*: g) where
>   gLinearizeStep (x :*: y) = gLinearizeStep x <|> gLinearizeStep y

> instance ToTokens t c => GToTokensStep t (K1 i c) where
>   gLinearizeStep = linearize . unK1

> instance GToTokensStep t f => GToTokensStep t (M1 i c f) where
>   gLinearizeStep = gLinearizeStep . unM1

> class FromTokensStep (b :: Type -> Type) (t :: Type -> Type) (base :: Type -> Type) where
>   parseStep :: From b t (base (TTape t))
>   default parseStep :: (Functor b, Generic (base (TTape t)), GFromTokensStep b t (Rep (base (TTape t)))) => From b t (base (TTape t))
>   parseStep = to <$> gParseStep

> class GFromTokensStep (b :: Type -> Type) (t :: Type -> Type) (f :: Type -> Type) where
>   gParseStep :: forall a. From b t (f a)

> instance MonadFail b => GFromTokensStep b t V1 where
>   gParseStep = fail "GFromTokensStep.V1"

> instance Monad b => GFromTokensStep b t U1 where
>   gParseStep = pure U1

> instance
>   ( MonadFail b,
>     MonadPlus b,
>     Cons (TTape t) (TTape t) Token Token,
>     GFromTokensStep b t f,
>     GFromTokensStep b t g
>   ) => GFromTokensStep b t (f :+: g) where
>   gParseStep = (isToken L >> L1 <$> gParseStep) <|> (isToken R >> R1 <$> gParseStep)

> instance
>   ( MonadFail b, 
>     MonadPlus b, 
>     Cons (TTape t) (TTape t) Token Token,
>     GFromTokensStep b t f,
>     GFromTokensStep b t g
>   ) => GFromTokensStep b t (f :*: g) where
>   gParseStep = (:*:) <$> gParseStep <*> gParseStep

> instance (Monad b, FromTokens b t c) => GFromTokensStep b t (K1 i c) where
>   gParseStep = K1 <$> parse

> instance (Functor b, GFromTokensStep b t f) => GFromTokensStep b t (M1 i c f) where
>   gParseStep = M1 <$> gParseStep

> instance (MonadFail b, MonadPlus b, Cons (t Token) (t Token) Token Token, Alternative t, FromTokens b t a) => FromTokensStep b t (TreeF a)

> resetParse :: Monad b => From b t a -> TTape t -> From b t a
> resetParse m = lift . evalStateT m 

> class FromTokens (b :: Type -> Type) (t :: Type -> Type) (a :: Type) where
>   parse :: From b t a
>   default parse :: (Corecursive a, Monad b, Traversable (Base a), FromTokensStep b t (Base a)) => From b t a
>   parse = go where go = fmap embed $ parseStep >>= traverse (resetParse go) 

> instance (MonadFail b, Cons (TTape t) (TTape t) Token Token) => FromTokens b t Int where
>   parse = token >>= \case
>       I i -> pure i
>       _ -> fail "expected Int"

> instance (MonadFail b, Alternative t, Cons (TTape t) (TTape t) Token Token) => FromTokens b t (TTape t) where
>   parse = token >>= \case
>     Rec n -> go n
>       where go :: Int -> From b t (TTape t)
>             go 0 = pure empty
>             go n' = cons <$> token <*> go (n' - 1)
>     _ -> fail "expected Rec"

> instance (Monad b, FromTokensStep b t (TreeF a)) => FromTokens b t (Tree a)

```
evalStateT parse (linearize @[] exampleTree) == Just exampleTree
```

> data NextF a r = FirstF r | SecondF a | ThirdF r

> accumTree'''''''' :: forall a. (Monoid a, ToTokens [] a, FromTokens Maybe [] a) => StateT (TTape [], Stack (NextF a (TTape []))) (Writer a) ()
> accumTree'''''''' =
>   while $ do
>     treeF <- fromJust . evalStateT parseStep <$> zoom _1 get
>     case treeF of
>       NilF -> do
>         c <- zoom _2 pop
>         case c of
>           Just (FirstF leftF) -> do
>             zoom _1 $ put leftF
>             pure Continue
>           Just (SecondF contentF) -> do
>             lift (tell contentF)
>             zoom _1 $ put (linearizeStep $ NilF @a)
>             pure Continue
>           Just (ThirdF rightF) -> do
>             zoom _1 $ put rightF
>             pure Continue
>           Nothing -> pure Break
>       NodeF {..} -> do
>         zoom _2 $ push (ThirdF rightF)
>         zoom _2 $ push (SecondF contentF)
>         zoom _2 $ push (FirstF leftF)
>         zoom _1 $ put (linearizeStep $ NilF @a)
>         pure Continue

> makeBaseFunctor ''Sum

> deriving stock instance (Show a, Show r) => Show (SumF a r)
> deriving stock instance Generic (SumF a r)

> instance (Alternative t, Foldable t, ToTokens t a) => ToTokensStep t (SumF a)

> instance (Alternative t, Foldable t, ToTokens t a) => ToTokens t (Sum a)

> instance (Monad b, Alternative t, Foldable t, FromTokens b t a) => FromTokensStep b t (SumF a)

> instance (Monad b, Alternative t, Foldable t, FromTokens b t a) => FromTokens b t (Sum a)

```
execWriter @(Sum Int) $ runStateT accumTree'''''''' (Sum <$> exampleTree, [])
```
