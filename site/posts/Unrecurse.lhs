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
> {-# OPTIONS_GHC -Wno-incomplete-patterns #-}

> module Unrecurse where

> import Prelude hiding (even, odd)

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
