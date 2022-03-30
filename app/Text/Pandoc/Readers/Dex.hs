{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Text.Pandoc.Readers.Dex where

import Control.Monad.IO.Class (MonadIO (liftIO))
import Control.Monad.Reader (MonadReader (ask), ReaderT (runReaderT))
import Data.Foldable (Foldable (fold))
import qualified Data.Text as T
import qualified Err as Dex
import Network.URI.JSON ()
import qualified PPrint as Dex ()
import qualified Syntax as Dex (Output (HtmlOut, TextOut), Result (Result), SourceBlock (sbContents), SourceBlock' (..))
import qualified Text.Pandoc as Pandoc
import qualified Text.Pandoc.Builder as Pandoc
import qualified Text.Pandoc.Parsing as Pandoc
import qualified TopLevel as Dex

readDex ::
  (MonadIO m, Pandoc.PandocMonad m, Pandoc.ToSources a) =>
  Pandoc.ReaderOptions ->
  Dex.EvalConfig ->
  a ->
  m Pandoc.Pandoc
readDex pandocOpts dexOpts s = flip runReaderT pandocOpts $ do
  dexEnv <- liftIO Dex.loadCache
  let source :: String = T.unpack . Pandoc.sourcesToText . Pandoc.toSources $ s
  (results, dexEnv') <- liftIO $ Dex.runTopperM dexOpts dexEnv $ Dex.evalSourceText source
  Dex.storeCache dexEnv'
  fold
    <$> traverse
      ( \(sourceBlock, result) -> do
          Pandoc.Pandoc meta blocks <- toPandoc sourceBlock
          Pandoc.Pandoc meta' blocks' <- toPandoc result
          return $ Pandoc.Pandoc (meta <> meta') (blocks <> blocks')
      )
      results

class ToPandoc a where
  toPandoc :: (MonadReader Pandoc.ReaderOptions m, Pandoc.PandocMonad m) => a -> m Pandoc.Pandoc

instance ToPandoc Dex.Result where
  toPandoc (Dex.Result outs err) = do
    outsBlocks <- fold <$> traverse toPandoc outs
    errBlocks <- toPandoc err
    return $ outsBlocks <> errBlocks

instance ToPandoc (Dex.Except ()) where
  toPandoc err = case err of
    Dex.Failure er ->
      pure
        . Pandoc.Pandoc mempty
        . Pandoc.toList
        . Pandoc.codeBlock
        . T.pack
        $ Dex.pprint er
    Dex.Success _x0 -> pure mempty

instance ToPandoc Dex.Output where
  toPandoc out = case out of
    Dex.TextOut s ->
      pure
        . Pandoc.Pandoc mempty
        . Pandoc.toList
        . Pandoc.plain
        . Pandoc.str
        . T.pack
        $ s
    Dex.HtmlOut s -> do
      pandocOpts <- ask
      Pandoc.readHtml pandocOpts . T.pack $ s
    -- Dex.PassInfo pn s -> undefined
    -- Dex.EvalTime x ma -> undefined
    -- Dex.TotalTime x -> undefined
    -- Dex.BenchResult s x y ma -> undefined
    -- Dex.MiscLog s -> undefined
    _ ->
      pure
        . Pandoc.Pandoc mempty
        . Pandoc.toList
        . Pandoc.codeBlock
        . T.pack
        $ Dex.pprint out

instance ToPandoc Dex.SourceBlock where
  toPandoc sourceBlock = case Dex.sbContents sourceBlock of
    -- Dex.EvalUDecl ud -> undefined
    -- Dex.Command cn wse -> undefined
    -- Dex.DeclareForeign s uab -> undefined
    -- Dex.GetNameType s -> undefined
    -- Dex.ImportModule msn -> undefined
    -- Dex.QueryEnv eq -> undefined
    Dex.ProseBlock s -> do
      pandocOpts <- ask
      Pandoc.readCommonMark pandocOpts . T.pack $ s
    -- Dex.CommentLine -> undefined
    Dex.EmptyLines -> pure mempty
    -- Dex.UnParseable b s -> undefined
    _ ->
      pure
        . Pandoc.Pandoc mempty
        . Pandoc.toList
        . Pandoc.codeBlock
        . T.pack
        $ Dex.pprint sourceBlock
