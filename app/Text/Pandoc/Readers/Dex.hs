{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Text.Pandoc.Readers.Dex where

import Control.Applicative (Alternative ((<|>)), optional)
import Control.Monad.Error (MonadError (throwError))
import Control.Monad.IO.Class (MonadIO (liftIO))
import Control.Monad.Reader (MonadReader (ask), ReaderT (runReaderT))
import qualified Data.Aeson as A (Object, Result (Error, Success), Value (..), fromJSON)
import qualified Data.ByteString as BS
import Data.Foldable (Foldable (fold))
import Data.Map.Lazy (Map)
import qualified Data.Map.Lazy as Map
import qualified Data.Text as T
import qualified Data.Yaml as Yaml
import Debug.Trace (trace, traceShowId)
import qualified Err as Dex
import Network.URI.JSON ()
import qualified PPrint as Dex ()
import qualified Syntax as Dex (Output (HtmlOut, TextOut), Result (Result), SourceBlock (sbContents), SourceBlock' (..))
import qualified Text.Pandoc as Pandoc
import qualified Text.Pandoc.Builder as Pandoc
import qualified Text.Pandoc.Parsing as Pandoc
import qualified Text.Pandoc.Shared as Pandoc
import qualified Text.Pandoc.UTF8 as Pandoc
import qualified TopLevel as Dex

readDex ::
  (MonadIO m, Pandoc.PandocMonad m, Pandoc.ToSources a) =>
  Pandoc.ReaderOptions ->
  Dex.EvalConfig ->
  a ->
  m Pandoc.Pandoc
readDex pandocOpts dexOpts s = flip runReaderT dexOpts $ do
  dexEnv <- liftIO Dex.loadCache
  parsed <-
    Pandoc.readWithM
      ((,) <$> parseDex <*> Pandoc.getState)
      (ParserStateWithDexEnv Pandoc.def {Pandoc.stateOptions = pandocOpts} dexEnv)
      (Pandoc.ensureFinalNewlines 3 (Pandoc.toSources s))
  case parsed of
    Right (result, st) -> do
      Dex.storeCache $ _dexEnv st
      return result
    Left e -> throwError e

type PandocParser st m = Pandoc.ParserT Pandoc.Sources st m

data ParserStateWithDexEnv = ParserStateWithDexEnv
  { _parserState :: Pandoc.ParserState,
    _dexEnv :: Dex.TopStateEx
  }

parseDex ::
  (MonadIO m, MonadReader Dex.EvalConfig m, Pandoc.PandocMonad m) =>
  PandocParser ParserStateWithDexEnv m Pandoc.Pandoc
parseDex = do
  blocks <- parseDexBlocks
  st <- Pandoc.getState
  let parserState = _parserState st
      doc =
        Pandoc.runF
          ( do
              Pandoc.Pandoc meta' bs <- Pandoc.doc <$> blocks
              meta <- Pandoc.stateMeta' parserState
              trace ("[parseDex] " <> show meta') $ return $ Pandoc.Pandoc meta bs
          )
          parserState
  Pandoc.reportLogMessages
  return doc

yamlMetaBlock' ::
  (MonadIO m, MonadReader Dex.EvalConfig m, Pandoc.PandocMonad m) =>
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState Pandoc.Blocks)
yamlMetaBlock' = do
  Pandoc.guardEnabled Pandoc.Ext_yaml_metadata_block
  newMetaF <- trace "[newMetaF]" yamlMetaBlock
  let msg = Pandoc.tshow $ Pandoc.runF newMetaF Pandoc.defaultParserState
  trace ("[yamlMetaBlock'] " <> T.unpack msg) $ return ()
  Pandoc.updateState $ \(st :: ParserStateWithDexEnv) ->
    let parserState = _parserState st
     in st {_parserState = parserState {Pandoc.stateMeta' = Pandoc.stateMeta' parserState <> newMetaF}}
  return mempty

yamlMetaBlock ::
  Pandoc.PandocMonad m =>
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState Pandoc.Meta)
yamlMetaBlock = Pandoc.try $ do
  _ <- dexComment $ do
    _ <- Pandoc.string "---"
    _ <- Pandoc.blankline
    Pandoc.notFollowedBy Pandoc.blankline -- if --- is followed by a blank it's an HRULE
  rawYamlLines <- Pandoc.manyTill (dexComment Pandoc.anyLine) (dexComment stopLine)
  -- by including --- and ..., we allow yaml blocks with just comments:
  let rawYaml = T.unlines ("---" : (rawYamlLines ++ ["..."]))
  _ <- optional Pandoc.blanklines
  yamlBsToMeta $ Pandoc.fromText rawYaml

dexComment ::
  Pandoc.PandocMonad m =>
  PandocParser st m a ->
  PandocParser st m a
dexComment parser = do
  Pandoc.try $ do
    _ <- Pandoc.string "-- "
    parser

stopLine :: Monad m => PandocParser st m ()
stopLine = Pandoc.try $ (Pandoc.string "---" <|> Pandoc.string "...") >> Pandoc.blankline >> return ()

yamlBsToMeta ::
  Pandoc.PandocMonad m =>
  BS.ByteString ->
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState Pandoc.Meta)
yamlBsToMeta bstr =
  case Yaml.decodeAllEither' bstr of
    Right (A.Object o : _) -> fmap Pandoc.Meta <$> yamlMap o
    Right [] -> return . return $ mempty
    Right [A.Null] -> return . return $ mempty
    Right _ -> Prelude.fail "expected YAML object"
    Left err' -> do
      throwError $
        Pandoc.PandocParseError $
          T.pack $ Yaml.prettyPrintParseException err'

yamlMap ::
  Pandoc.PandocMonad m =>
  A.Object ->
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState (Map T.Text Pandoc.MetaValue))
yamlMap o = do
  case A.fromJSON (A.Object o) of
    A.Error err' -> throwError $ Pandoc.PandocParseError $ T.pack err'
    A.Success (m' :: Map T.Text A.Value) -> do
      let kvs = filter (not . ignorable . fst) $ Map.toList m'
      fmap Map.fromList . sequence <$> mapM toMeta kvs
  where
    ignorable t = "_" `T.isSuffixOf` t
    toMeta (k, v) = do
      fv <- yamlToMetaValue v
      return $ do
        v' <- fv
        return (k, v')

yamlToMetaValue ::
  Pandoc.PandocMonad m =>
  A.Value ->
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState Pandoc.MetaValue)
yamlToMetaValue v =
  case v of
    A.String t -> normalize t
    A.Bool b -> return . return $ Pandoc.MetaBool b
    A.Number d -> normalize $
      case A.fromJSON v of
        A.Success (i :: Int) -> Pandoc.tshow i
        _ -> Pandoc.tshow d
    A.Null -> return . return $ Pandoc.MetaString ""
    A.Array {} -> do
      case A.fromJSON v of
        A.Error err' -> throwError $ Pandoc.PandocParseError $ T.pack err'
        A.Success vs ->
          fmap Pandoc.MetaList . sequence
            <$> mapM yamlToMetaValue vs
    A.Object o -> fmap Pandoc.MetaMap <$> yamlMap o
  where
    normalize t = return . return $ Pandoc.MetaString t

parseDexBlocks ::
  (MonadIO m, MonadReader Dex.EvalConfig m, Pandoc.PandocMonad m) =>
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState Pandoc.Blocks)
parseDexBlocks = do
  mconcat <$> Pandoc.manyTill block Pandoc.eof
  where
    block = do
      res <-
        Pandoc.choice
          [ mempty <$ Pandoc.blanklines,
            yamlMetaBlock',
            dexBlock
          ]
      let msg = Pandoc.tshow $ Pandoc.toList $ Pandoc.runF res Pandoc.defaultParserState
      trace ("[parseDexBlocks] " <> T.unpack msg) $ return res

instance Pandoc.HasReaderOptions ParserStateWithDexEnv where
  extractReaderOptions = Pandoc.extractReaderOptions . _parserState

instance Pandoc.HasLastStrPosition ParserStateWithDexEnv where
  setLastStrPos msp st = st {_parserState = Pandoc.setLastStrPos msp (_parserState st)}
  getLastStrPos = Pandoc.getLastStrPos . _parserState

instance Pandoc.HasLogMessages ParserStateWithDexEnv where
  addLogMessage lm st = st {_parserState = Pandoc.addLogMessage lm (_parserState st)}
  getLogMessages = Pandoc.getLogMessages . _parserState

dexBlock ::
  (MonadIO m, MonadReader Dex.EvalConfig m, Pandoc.PandocMonad m) =>
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState Pandoc.Blocks)
dexBlock = Pandoc.try $ do
  raw <- Pandoc.many1 Pandoc.anyLine
  let dexLines = T.unlines raw
  trace ("[dexBlock] " <> T.unpack dexLines) $ return ()
  st <- Pandoc.getState
  opts <- ask
  (r, dexEnv) <- liftIO $ Dex.runTopperM opts (_dexEnv st) $ Dex.evalSourceText $ T.unpack dexLines
  Pandoc.updateState (\st' -> st' {_dexEnv = dexEnv})
  blocks <- traverse (\(sourceBlock, result) -> toPandocBlocks sourceBlock <> toPandocBlocks result) r
  return . return $ mconcat blocks

class ToPandocBlocks a where
  toPandocBlocks :: Pandoc.PandocMonad m => a -> m Pandoc.Blocks

instance ToPandocBlocks Dex.Result where
  toPandocBlocks (Dex.Result outs err) = do
    outsBlocks <- fold <$> traverse toPandocBlocks outs
    errBlocks <- toPandocBlocks err
    return $ outsBlocks <> errBlocks

instance ToPandocBlocks (Dex.Except ()) where
  toPandocBlocks err = case err of
    Dex.Failure er -> pure $ Pandoc.codeBlock . T.pack $ Dex.pprint er
    Dex.Success _x0 -> pure mempty

instance ToPandocBlocks Dex.Output where
  toPandocBlocks out = case out of
    Dex.TextOut s -> pure $ Pandoc.plain . Pandoc.str . T.pack $ s
    Dex.HtmlOut s -> do
      Pandoc.Pandoc meta blocks <- Pandoc.readHtml Pandoc.def . T.pack $ s
      if null . Pandoc.unMeta $ meta
        then pure $ Pandoc.fromList blocks
        else throwError $ Pandoc.PandocParseError "invalid meta for html output"
    -- Dex.PassInfo pn s -> undefined
    -- Dex.EvalTime x ma -> undefined
    -- Dex.TotalTime x -> undefined
    -- Dex.BenchResult s x y ma -> undefined
    -- Dex.MiscLog s -> undefined
    _ -> pure $ Pandoc.codeBlock . T.pack $ traceShowId $ Dex.pprint out

instance ToPandocBlocks Dex.SourceBlock where
  toPandocBlocks sourceBlock = case Dex.sbContents sourceBlock of
    -- Dex.EvalUDecl ud -> undefined
    -- Dex.Command cn wse -> undefined
    -- Dex.DeclareForeign s uab -> undefined
    -- Dex.GetNameType s -> undefined
    -- Dex.ImportModule msn -> undefined
    -- Dex.QueryEnv eq -> undefined
    Dex.ProseBlock s -> do
      Pandoc.Pandoc meta blocks <- Pandoc.readCommonMark Pandoc.def . T.pack $ s
      if null . Pandoc.unMeta $ meta
        then pure $ Pandoc.fromList blocks
        else throwError $ Pandoc.PandocParseError "invalid meta for prose block"
    -- Dex.CommentLine -> undefined
    Dex.EmptyLines -> pure mempty
    -- Dex.UnParseable b s -> undefined
    _ -> pure $ Pandoc.codeBlock . T.pack $ Dex.pprint sourceBlock
