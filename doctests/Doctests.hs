module Main where

import Build_doctests (flags_exe_website, module_sources_exe_website, pkgs_exe_website)
import Data.Foldable (traverse_)
import System.Environment (lookupEnv)
import Test.DocTest (doctest)

main :: IO ()
main = do
  libDir <- lookupEnv "NIX_GHC_LIBDIR"

  let args =
        concat
          [ flags_exe_website,
            pkgs_exe_website,
            maybe [] (\x -> ["-package-db " <> x <> "/package.conf.d"]) libDir,
            ["-XOverloadedStrings", "-XScopedTypeVariables", "-XDataKinds"],
            module_sources_exe_website
          ]

  traverse_ putStrLn args
  doctest args
