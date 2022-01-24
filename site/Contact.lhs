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
