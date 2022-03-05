module Main where

import Build_doctests (Component (..), components)
import Data.Foldable (for_)
import System.Environment (lookupEnv)
import Test.DocTest (doctest)

main :: IO ()
main = do
  libDir <- lookupEnv "NIX_GHC_LIBDIR"
  for_ components $ \(Component name flags pkgs sources) -> do
    print name
    putStrLn "----------------------------------------"
    let args =
          concat
            [ flags,
              pkgs,
              maybe [] (\x -> ["-package-db " <> x <> "/package.conf.d"]) libDir,
              [ "-XOverloadedStrings",
                "-XScopedTypeVariables",
                "-XDataKinds",
                "-XTypeApplications"
              ],
              sources
            ]
    for_ args putStrLn
    doctest args
