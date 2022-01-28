> module Contact where

If you want to write me, feel free to send me an `email`, where

> email :: String
> email =
>   concat $
>     zipWith
>       (<>)
>       ( reverse
>           [ "com",
>             "mail",
>             "google",
>             "scholak",
>             "torsten"
>           ]
>       )
>       [".", "@", mempty, ".", mempty]

If you can't figure this out by looking at the Haskell code,
open [ghci](https://downloads.haskell.org/~ghc/latest/docs/html/users_guide/ghci.html?highlight=ghci)
and copy&paste the above code into it.
<https://tryhaskell.org/> is a good place to start to learn Haskell.

You can also find me on GitHub, Twitter, and other social media.
Click on the icons below to see what I'm up to.
