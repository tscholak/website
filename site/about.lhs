> module About where

If you want to write me, then you can send me an `email`, where

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
