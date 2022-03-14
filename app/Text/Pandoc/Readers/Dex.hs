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
import qualified Err as Dex
import Network.URI.JSON ()
import qualified PPrint as Dex ()
import qualified Parser as Dex
import qualified Syntax as Dex (Output (BenchResult, EvalTime, HtmlOut, MiscLog, PassInfo, TextOut, TotalTime), Result (Result), SourceBlock (sbContents), SourceBlock' (..))
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
              Pandoc.Pandoc _ bs <- Pandoc.doc <$> blocks
              meta <- Pandoc.stateMeta' parserState
              return $ Pandoc.Pandoc meta bs
          )
          parserState
  Pandoc.reportLogMessages
  return doc

yamlMetaBlock' ::
  (MonadIO m, MonadReader Dex.EvalConfig m, Pandoc.PandocMonad m) =>
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState Pandoc.Blocks)
yamlMetaBlock' = do
  Pandoc.guardEnabled Pandoc.Ext_yaml_metadata_block
  newMetaF <- yamlMetaBlock (fmap Pandoc.toMetaValue <$> parseDexBlocks)
  -- Since `<>` is left-biased, existing values are not touched:
  Pandoc.updateState $ \(st :: ParserStateWithDexEnv) ->
    let parserState = _parserState st
     in st {_parserState = parserState {Pandoc.stateMeta' = Pandoc.stateMeta' parserState <> newMetaF}}
  return mempty

yamlMetaBlock ::
  Pandoc.PandocMonad m =>
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState Pandoc.MetaValue) ->
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState Pandoc.Meta)
yamlMetaBlock parser = Pandoc.try $ do
  _ <- dexComment $ do
    _ <- Pandoc.string "---"
    _ <- Pandoc.blankline
    Pandoc.notFollowedBy Pandoc.blankline -- if --- is followed by a blank it's an HRULE
  rawYamlLines <- Pandoc.manyTill (dexComment Pandoc.anyLine) (dexComment stopLine)
  -- by including --- and ..., we allow yaml blocks with just comments:
  let rawYaml = T.unlines ("---" : (rawYamlLines ++ ["..."]))
  _ <- optional Pandoc.blanklines
  yamlBsToMeta parser $ Pandoc.fromText rawYaml

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
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState Pandoc.MetaValue) ->
  BS.ByteString ->
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState Pandoc.Meta)
yamlBsToMeta pMetaValue bstr = do
  case Yaml.decodeAllEither' bstr of
    Right (A.Object o : _) -> fmap Pandoc.Meta <$> yamlMap pMetaValue o
    Right [] -> return . return $ mempty
    Right [A.Null] -> return . return $ mempty
    Right _ -> Prelude.fail "expected YAML object"
    Left err' -> do
      throwError $
        Pandoc.PandocParseError $
          T.pack $ Yaml.prettyPrintParseException err'

yamlMap ::
  Pandoc.PandocMonad m =>
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState Pandoc.MetaValue) ->
  A.Object ->
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState (Map T.Text Pandoc.MetaValue))
yamlMap pMetaValue o = do
  case A.fromJSON (A.Object o) of
    A.Error err' -> throwError $ Pandoc.PandocParseError $ T.pack err'
    A.Success (m' :: Map T.Text A.Value) -> do
      let kvs = filter (not . ignorable . fst) $ Map.toList m'
      fmap Map.fromList . sequence <$> mapM toMeta kvs
  where
    ignorable t = "_" `T.isSuffixOf` t
    toMeta (k, v) = do
      fv <- yamlToMetaValue pMetaValue v
      return $ do
        v' <- fv
        return (k, v')

yamlToMetaValue ::
  Pandoc.PandocMonad m =>
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState Pandoc.MetaValue) ->
  A.Value ->
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState Pandoc.MetaValue)
yamlToMetaValue pMetaValue v =
  case v of
    A.String t -> normalize t
    A.Bool b -> return $ return $ Pandoc.MetaBool b
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
            <$> mapM (yamlToMetaValue pMetaValue) vs
    A.Object o -> fmap Pandoc.MetaMap <$> yamlMap pMetaValue o
  where
    normalize t =
      -- Note: a standard quoted or unquoted YAML value will
      -- not end in a newline, but a "block" set off with
      -- `|` or `>` will.
      if "\n" `T.isSuffixOf` T.dropWhileEnd isSpaceChar t -- see #6823
        then Pandoc.parseFromString' pMetaValue (t <> "\n")
        else Pandoc.parseFromString' asInlines t
    asInlines = fmap b2i <$> pMetaValue
    b2i (Pandoc.MetaBlocks [Pandoc.Plain ils]) = Pandoc.MetaInlines ils
    b2i (Pandoc.MetaBlocks [Pandoc.Para ils]) = Pandoc.MetaInlines ils
    b2i bs = bs
    isSpaceChar ' ' = True
    isSpaceChar '\t' = True
    isSpaceChar _ = False

parseDexBlocks ::
  (MonadIO m, MonadReader Dex.EvalConfig m, Pandoc.PandocMonad m) =>
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState Pandoc.Blocks)
parseDexBlocks = do
  ret <- evalDexBlock Dex.preludeImportBlock
  mconcat . (:) ret <$> Pandoc.manyTill block Pandoc.eof
  where
    block =
      Pandoc.choice
        [ mempty <$ Pandoc.blanklines,
          yamlMetaBlock',
          dexBlock
        ]

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
  rawDex <- Pandoc.many Pandoc.anyLine
  let dexSource = T.unlines rawDex
  case Dex.parseTopDeclRepl $ T.unpack dexSource of
    Just sourceBlock -> evalDexBlock sourceBlock
    Nothing -> fail "invalid dex block"

evalDexBlock ::
  (MonadIO m, MonadReader Dex.EvalConfig m, Pandoc.PandocMonad m) =>
  Dex.SourceBlock ->
  PandocParser ParserStateWithDexEnv m (Pandoc.Future Pandoc.ParserState Pandoc.Blocks)
evalDexBlock sourceBlock = do
  st <- Pandoc.getState
  opts <- ask
  (result, dexEnv) <- liftIO $ Dex.evalSourceBlockIO opts (_dexEnv st) sourceBlock
  Pandoc.updateState (\st' -> st' {_dexEnv = dexEnv})
  blocks <- toPandocBlocks sourceBlock <> toPandocBlocks result
  return . return $ blocks

class ToPandocBlocks a where
  toPandocBlocks :: Pandoc.PandocMonad m => a -> m Pandoc.Blocks

instance ToPandocBlocks Dex.Result where
  toPandocBlocks (Dex.Result outs err) = do
    outsBlocks <- fold <$> traverse toPandocBlocks outs
    errBlocks <- toPandocBlocks err
    return $ outsBlocks <> errBlocks

instance ToPandocBlocks (Dex.Except a) where
  toPandocBlocks err = case err of
    Dex.Failure er -> undefined
    Dex.Success x0 -> undefined

instance ToPandocBlocks Dex.Output where
  toPandocBlocks out = case out of
    Dex.TextOut s -> pure $ Pandoc.plain . Pandoc.str . T.pack $ s
    Dex.HtmlOut s -> do
      Pandoc.Pandoc meta blocks <- Pandoc.readHtml Pandoc.def . T.pack $ s
      if null . Pandoc.unMeta $ meta
        then pure $ Pandoc.fromList blocks
        else throwError $ Pandoc.PandocParseError "invalid meta for html output"
    Dex.PassInfo pn s -> undefined
    Dex.EvalTime x ma -> undefined
    Dex.TotalTime x -> undefined
    Dex.BenchResult s x y ma -> undefined
    Dex.MiscLog s -> undefined

instance ToPandocBlocks Dex.SourceBlock where
  toPandocBlocks sourceBlock = case Dex.sbContents sourceBlock of
    Dex.EvalUDecl ud -> undefined
    Dex.Command cn wse -> undefined
    Dex.DeclareForeign s uab -> undefined
    Dex.GetNameType s -> undefined
    Dex.ImportModule msn -> undefined
    Dex.QueryEnv eq -> undefined
    Dex.ProseBlock s -> do
      Pandoc.Pandoc meta blocks <- Pandoc.readCommonMark Pandoc.def . T.pack $ s
      if null . Pandoc.unMeta $ meta
        then pure $ Pandoc.fromList blocks
        else throwError $ Pandoc.PandocParseError "invalid meta for prose block"
    Dex.CommentLine -> undefined
    Dex.EmptyLines -> pure mempty
    Dex.UnParseable b s -> undefined
