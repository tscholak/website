---
title: "Unrecurse"
date: Jan 20, 2022
teaser: Let's play with recursion!
tags:
  items: [haskell]
---

> {-# LANGUAGE RankNTypes #-}
> {-# LANGUAGE GADTs #-}
> {-# LANGUAGE RecordWildCards #-}
> {-# LANGUAGE DeriveFunctor #-}
> {-# LANGUAGE ScopedTypeVariables #-}
> {-# LANGUAGE DerivingStrategies #-}
> {-# LANGUAGE BangPatterns #-}
> {-# LANGUAGE FlexibleContexts #-}
> {-# LANGUAGE KindSignatures #-}
> {-# LANGUAGE MultiParamTypeClasses #-}
> {-# LANGUAGE DefaultSignatures #-}
> {-# LANGUAGE FlexibleInstances #-}
> {-# LANGUAGE TypeOperators #-}
> {-# LANGUAGE UndecidableInstances #-}
> {-# LANGUAGE DeriveGeneric #-}
> {-# OPTIONS_GHC -Wno-incomplete-patterns #-}

> module Unrecurse where

> import Prelude hiding (even, odd)
> import Control.Lens (zoom, _1, _2, uncons, Cons)
> import Control.Monad.Cont (ContT, MonadTrans (lift))
> import Control.Monad.State (MonadState (get, put), modify, StateT (runStateT), evalStateT)
> import Control.Monad (void, MonadPlus, mfilter)
> import Data.Kind (Type)
> import GHC.Generics (Generic (Rep, from, to), K1 (unK1, K1), M1 (unM1, M1), type (:+:) (L1, R1), type (:*:) ((:*:)), V1, U1 (U1))
> import Control.Applicative (Alternative((<|>), empty))
  
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

> exampleTree :: Tree Int
> exampleTree = Node (Node (Node Nil 1 Nil) 2 (Node Nil 3 Nil)) 4 (Node (Node Nil 5 Nil) 6 (Node Nil 7 Nil))

> printTree :: forall a. Show a => Tree a -> IO ()
> printTree Nil = pure ()
> printTree Node {..} = do
>   printTree left
>   print content
>   printTree right

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

> while :: forall a m. Monad m => (a -> m Bool) -> (a -> m a) -> a -> m a
> while p f x = p x >>= \b -> if b then f x >>= while p f else pure x

> data Continue = Continue | Break

> printTree''''''' :: forall a. Show a => StateT (Tree a, Stack (Next a)) IO ()
> printTree''''''' =
>   let p Continue = pure True
>       p Break = pure False
>   in void $ while p (\_ -> do
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
>     ) Continue

> data Token =
>   L | R | Grow | Reduce | I Int
>   deriving stock (Eq, Show)

> type To t a = a -> t Token

> type From b t a = StateT (t Token) b a

> token :: forall b t. (MonadFail b, Cons (t Token) (t Token) Token Token) => From b t Token
> token = do
>   t <- get
>   case uncons t of
>     Nothing -> fail "unexpected end of input"
>     Just (x, xs) -> put xs >> pure x

> isToken :: (MonadFail b, MonadPlus b, Eq Token, Cons (t Token) (t Token) Token Token) => Token -> From b t Token
> isToken t = mfilter (== t) token

> class ToTokens (t :: Type -> Type) (a :: Type) where
>   linearize :: To t a
>   default linearize :: (Generic a, GToTokens t (Rep a)) => To t a
>   linearize = gLinearize . from

> class GToTokens (t :: Type -> Type) (f :: Type -> Type) where
>   gLinearize :: forall a. To t (f a)

> class FromTokens (b :: Type -> Type) (t :: Type -> Type) (a :: Type) where
>   parse :: From b t a
>   default parse :: (Functor b, Generic a, GFromTokens b t (Rep a)) => From b t a
>   parse = to <$> gParse

> class GFromTokens (b :: Type -> Type) (t :: Type -> Type) (f :: Type -> Type) where
>   gParse :: forall a. From b t (f a)

> instance GToTokens t V1 where
>   gLinearize v = v `seq` error "GToTokens.V1"

> instance Alternative t => GToTokens t U1 where
>   gLinearize _ = empty

> instance (Applicative t, Alternative t, GToTokens t f, GToTokens t g) => GToTokens t (f :+: g) where
>   gLinearize (L1 x) = pure L <|> gLinearize x
>   gLinearize (R1 x) = pure R <|> gLinearize x

> instance (Alternative t, GToTokens t f, GToTokens t g) => GToTokens t (f :*: g) where
>   gLinearize (x :*: y) = gLinearize x <|> gLinearize y

> instance ToTokens t c => GToTokens t (K1 i c) where
>   gLinearize = linearize . unK1

> instance GToTokens t f => GToTokens t (M1 i c f) where
>   gLinearize = gLinearize . unM1

> instance MonadFail b => GFromTokens b t V1 where
>   gParse = fail "GFromTokens.V1"

> instance Monad b => GFromTokens b t U1 where
>   gParse = pure U1

> instance
>   ( MonadFail b,
>     MonadPlus b,
>     Cons (t Token) (t Token) Token Token,
>     GFromTokens b t f,
>     GFromTokens b t g
>   ) => GFromTokens b t (f :+: g) where
>   gParse = (isToken L >> L1 <$> gParse) <|> (isToken R >> R1 <$> gParse)

> instance (Monad b, GFromTokens b t f, GFromTokens b t g) => GFromTokens b t (f :*: g) where
>   gParse = (:*:) <$> gParse <*> gParse

> instance (Monad b, FromTokens b t c) => GFromTokens b t (K1 i c) where
>   gParse = K1 <$> parse

> instance (Functor b, GFromTokens b t f) => GFromTokens b t (M1 i c f) where
>   gParse = M1 <$> gParse

> instance (Alternative t) => ToTokens t Int where
>   linearize i = pure $ I i

> instance (Alternative t, ToTokens t a) => ToTokens t (Tree a)

> instance (MonadFail b, Cons (t Token) (t Token) Token Token) => FromTokens b t Int where
>   parse = do
>     t <- token
>     case t of
>       I i -> pure i
>       _ -> fail "expected Int"

> instance
>   ( MonadFail b,
>     MonadPlus b,
>     FromTokens b t a,
>     Cons (t Token) (t Token) Token Token
>   ) => FromTokens b t (Tree a)

```haskell
linearize @[] exampleTree
```

```haskell
runStateT (parse @[] @[] @(Tree Int)) (linearize @[] exampleTree)
```

> -- shiftParse :: (_ -> From b t a) -> From b t a

> -- resetParse :: From b t a -> From b t a
> -- resetParse = lift . evalStateT
