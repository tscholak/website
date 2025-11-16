{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Main (main) where

import qualified Barbies as Barbie
import Control.Applicative (Alternative (empty, (<|>)), Const (Const), optional, (<**>))
import Control.Lens (at, ix, (?~), (^?))
import Control.Monad (void)
import Control.Monad.IO.Class (MonadIO (liftIO))
import Data.Aeson (toJSON)
import qualified Data.Aeson as A (Encoding, FromArgs, FromJSON (parseJSON), KeyValue ((.=)), Options (..), Result (Error, Success), SumEncoding (..), ToJSON (toJSON), Value (..), defaultOptions, fromJSON, genericParseJSON, genericToEncoding, genericToJSON, object, withObject, withText, (.:), (.:?))
import qualified Data.Aeson.KeyMap as KM (insert, lookup, union)
import qualified Data.Aeson.Lens as A (AsValue (_Object, _String), key, pattern Integer)
import qualified Data.Aeson.Parser.Internal as A (jsonEOF')
import qualified Data.Attoparsec.ByteString as Atto
import qualified Data.Binary.Get as Get
import qualified Data.Binary.Put as Put
import qualified Data.ByteString as BS
import qualified Data.Char
import Data.Either.Validation (Validation)
import qualified Data.Either.Validation as Validation
import Data.Functor.Compose (Compose (Compose))
import Data.Functor.Identity (Identity (..))
import Data.List (isInfixOf, sortOn)
import qualified Data.List as List
import Data.List.NonEmpty (NonEmpty)
import qualified Data.List.NonEmpty as NonEmpty
import Data.Map.Lazy (Map)
import qualified Data.Map.Lazy as Map
import Data.Maybe (fromMaybe, mapMaybe)
import Data.Ord (Down (Down))
import qualified Data.Text as T
import Data.Time (UTCTime, defaultTimeLocale, getCurrentTime, parseTimeM, parseTimeOrError)
import Data.Time.Format.ISO8601 (iso8601Show)
import qualified Data.Yaml as Yaml
import Development.Shake (Action, Exit (..), ShakeOptions (..), Stderr (..), Stdout (..), cmd, copyFileChanged, forP, getDirectoryFiles, putNormal, readFile', shakeOptions, writeFile', pattern Chatty)
import Development.Shake.Classes (Binary (..))
import Development.Shake.FilePath (dropDirectory1, takeExtension, (-<.>), (</>))
import Development.Shake.Forward (cacheAction, shakeForward)
import GHC.Generics (Generic)
import Network.URI (URI (uriPath), parseURI)
import qualified Options.Applicative as Options
import qualified Slick (compileTemplate', convert, substitute)
import qualified Slick.Pandoc as Slick (defaultHtml5Options, loadUsingMeta, markdownToHTMLWithOpts)
import System.Exit (ExitCode (..))
import Text.Casing (fromHumps, toQuietSnake)
import qualified Text.Pandoc as Pandoc

-- | data type for configuration
data Config f = Config
  { inputFolder :: !(f FilePath),
    outputFolder :: !(f FilePath),
    siteMeta :: !(SiteMeta f)
  }
  deriving stock (Generic)
  deriving anyclass (Barbie.FunctorB, Barbie.TraversableB, Barbie.ApplicativeB, Barbie.ConstraintsB)

deriving stock instance (Barbie.AllBF Show f Config) => Show (Config f)

deriving stock instance (Barbie.AllBF Eq f Config) => Eq (Config f)

instance (Alternative f) => Semigroup (Config f) where
  (<>) :: (Alternative f) => Config f -> Config f -> Config f
  (<>) = Barbie.bzipWith (<|>)

instance (Alternative f) => Monoid (Config f) where
  mempty :: (Alternative f) => Config f
  mempty = Barbie.bpure empty

configCustomJSONOptions :: A.Options
configCustomJSONOptions = A.defaultOptions {A.fieldLabelModifier = toQuietSnake . fromHumps}

instance (Barbie.AllBF A.FromJSON f Config) => A.FromJSON (Config f) where
  parseJSON :: (Barbie.AllBF Yaml.FromJSON f Config) => Yaml.Value -> Yaml.Parser (Config f)
  parseJSON = A.genericParseJSON configCustomJSONOptions

instance (Barbie.AllBF A.ToJSON f Config) => A.ToJSON (Config f) where
  toJSON :: (Barbie.AllBF Yaml.ToJSON f Config) => Config f -> Yaml.Value
  toJSON = A.genericToJSON configCustomJSONOptions
  toEncoding :: (Barbie.AllBF Yaml.ToJSON f Config) => Config f -> A.Encoding
  toEncoding = A.genericToEncoding configCustomJSONOptions

-- | markdown options
markdownOptions :: Pandoc.ReaderOptions
markdownOptions =
  Pandoc.def {Pandoc.readerExtensions = pandocExtensions}
  where
    pandocExtensions =
      Pandoc.disableExtension Pandoc.Ext_autolink_bare_uris $
        mconcat
          [ Pandoc.extensionsFromList
              [ Pandoc.Ext_yaml_metadata_block,
                Pandoc.Ext_fenced_code_attributes,
                Pandoc.Ext_auto_identifiers,
                Pandoc.Ext_fenced_divs,
                Pandoc.Ext_link_attributes
              ],
            Pandoc.githubMarkdownExtensions
          ]

-- | convert markdown to html
markdownToHTML :: T.Text -> Action A.Value
markdownToHTML = Slick.markdownToHTMLWithOpts markdownOptions Slick.defaultHtml5Options

-- | Haskell code options
codeOptions :: Pandoc.ReaderOptions
codeOptions =
  Pandoc.def {Pandoc.readerExtensions = pandocExtensions}
  where
    pandocExtensions =
      mconcat
        [ Pandoc.extensionsFromList
            [ Pandoc.Ext_literate_haskell,
              Pandoc.Ext_yaml_metadata_block
            ],
          Pandoc.githubMarkdownExtensions
        ]

-- | convert literal Haskell code to html
codeToHTML :: T.Text -> Action A.Value
codeToHTML = Slick.markdownToHTMLWithOpts codeOptions Slick.defaultHtml5Options

-- | Reveal.js writer options
revealWriterOptions :: Pandoc.WriterOptions
revealWriterOptions = Pandoc.def
  { Pandoc.writerSlideLevel = Just 2
  , Pandoc.writerSectionDivs = True
  }

-- | convert markdown to reveal.js HTML
markdownToRevealHTML :: T.Text -> Action A.Value
markdownToRevealHTML = Slick.loadUsingMeta
  (Pandoc.readMarkdown markdownOptions)
  (Pandoc.writeRevealJs revealWriterOptions)
  (Pandoc.writeHtml5String Slick.defaultHtml5Options)

-- | compile and run code (if it has a main function)
compileAndRunCode :: FilePath -> Action ()
compileAndRunCode srcPath = do
  let cmdLine =
        unwords
          [ "stack",
            "exec",
            "--",
            "runhaskell",
            "-ilib", -- lib directory to import path
            "-isite/posts", -- posts directory for inter-module dependencies
            srcPath
          ]
  (Exit code, Stdout (_out :: String), Stderr (err :: String)) <- cmd cmdLine
  case code of
    ExitSuccess -> pure ()
    ExitFailure _ ->
      if any
        (`isInfixOf` err)
        [ "not in scope: ‘main’",
          "[GHC-76037]" -- present in recent GHCs for "not in scope"
        ]
        then putNormal $ "Skipping execution (no top-level main): " <> srcPath
        else fail $ "Failed to run " <> srcPath <> ":\n" <> err

-- | add site meta data to a JSON object
withSiteMeta :: Config Identity -> A.Value -> A.Value
withSiteMeta config (A.Object obj) = A.Object $ KM.union obj siteMetaObj
  where
    siteMetaObj = case A.toJSON . siteMeta $ config of
      A.Object obj' -> obj'
      _ -> error "siteMeta should serialize to an Object"
withSiteMeta _ v = error $ "only add site meta to objects, not " ++ show v

-- | Site meta data
data SiteMeta f = SiteMeta
  { siteBaseUrl :: !(f URI),
    siteTitle :: !(f String),
    siteAuthor :: !(f String),
    siteDescription :: !(f String),
    siteKeywords :: !(f String),
    siteTwitterHandle :: !(f (Maybe String)),
    siteTwitchHandle :: !(f (Maybe String)),
    siteYoutubeHandle :: !(f (Maybe String)),
    siteGithubUser :: !(f (Maybe String)),
    siteGoogleScholarHandle :: !(f (Maybe String)),
    siteGithubRepository :: !(f (Maybe URI))
  }
  deriving stock (Generic)
  deriving anyclass (Barbie.FunctorB, Barbie.TraversableB, Barbie.ApplicativeB, Barbie.ConstraintsB)

deriving stock instance (Barbie.AllBF Show f SiteMeta) => Show (SiteMeta f)

deriving stock instance (Barbie.AllBF Eq f SiteMeta) => Eq (SiteMeta f)

instance (Alternative f) => Semigroup (SiteMeta f) where
  (<>) :: (Alternative f) => SiteMeta f -> SiteMeta f -> SiteMeta f
  (<>) = Barbie.bzipWith (<|>)

instance (Alternative f) => Monoid (SiteMeta f) where
  mempty :: (Alternative f) => SiteMeta f
  mempty = Barbie.bpure empty

siteMetaCustomJSONOptions :: A.Options
siteMetaCustomJSONOptions = A.defaultOptions {A.fieldLabelModifier = toQuietSnake . fromHumps . drop 4}

instance (Barbie.AllBF A.FromJSON f SiteMeta) => A.FromJSON (SiteMeta f) where
  parseJSON :: (Barbie.AllBF Yaml.FromJSON f SiteMeta) => Yaml.Value -> Yaml.Parser (SiteMeta f)
  parseJSON = A.genericParseJSON siteMetaCustomJSONOptions

instance (Barbie.AllBF A.ToJSON f SiteMeta) => A.ToJSON (SiteMeta f) where
  toJSON :: (Barbie.AllBF Yaml.ToJSON f SiteMeta) => SiteMeta f -> Yaml.Value
  toJSON = A.genericToJSON siteMetaCustomJSONOptions
  toEncoding :: (Barbie.AllBF Yaml.ToJSON f SiteMeta) => SiteMeta f -> A.Encoding
  toEncoding = A.genericToEncoding siteMetaCustomJSONOptions

-- | option parser for config
configParser :: Config (Options.Parser `Compose` Maybe)
configParser = Barbie.bmap (Compose . optional) parser
  where
    parser =
      Config
        { inputFolder =
            Options.strOption $
              Options.long "input-folder"
                <> Options.short 'i'
                <> Options.metavar "INPUT-FOLDER"
                <> Options.help "The folder containing the site content",
          outputFolder =
            Options.strOption $
              Options.long "output-folder"
                <> Options.short 'o'
                <> Options.metavar "OUTPUT-FOLDER"
                <> Options.help "The folder where the site will be generated",
          siteMeta =
            SiteMeta
              { siteBaseUrl =
                  Options.option (Options.maybeReader parseURI) $
                    Options.long "base-url"
                      <> Options.short 'b'
                      <> Options.metavar "BASE-URL"
                      <> Options.help "The base url of the site",
                siteTitle =
                  Options.strOption $
                    Options.long "title"
                      <> Options.short 't'
                      <> Options.metavar "TITLE"
                      <> Options.help "The title of the site",
                siteAuthor =
                  Options.strOption $
                    Options.long "author"
                      <> Options.short 'a'
                      <> Options.metavar "AUTHOR"
                      <> Options.help "The author of the site",
                siteDescription =
                  Options.strOption $
                    Options.long "description"
                      <> Options.short 'd'
                      <> Options.metavar "DESCRIPTION"
                      <> Options.help "The description of the site",
                siteKeywords =
                  Options.strOption $
                    Options.long "keywords"
                      <> Options.short 'k'
                      <> Options.metavar "KEYWORDS"
                      <> Options.help "The keywords of the site",
                siteTwitterHandle =
                  Just
                    <$> Options.strOption
                      ( Options.long "twitter-handle"
                          <> Options.short 't'
                          <> Options.metavar "TWITTER-HANDLE"
                          <> Options.help "The twitter handle of the author of the site"
                      ),
                siteTwitchHandle =
                  Just
                    <$> Options.strOption
                      ( Options.long "twitch-handle"
                          <> Options.short 'c'
                          <> Options.metavar "TWITCH-HANDLE"
                          <> Options.help "The twitch handle of the author of the site"
                      ),
                siteYoutubeHandle =
                  Just
                    <$> Options.strOption
                      ( Options.long "youtube-handle"
                          <> Options.short 'y'
                          <> Options.metavar "YOUTUBE-HANDLE"
                          <> Options.help "The youtube handle of the author of the site"
                      ),
                siteGithubUser =
                  Just
                    <$> Options.strOption
                      ( Options.long "github-user"
                          <> Options.short 'g'
                          <> Options.metavar "GITHUB-USER"
                          <> Options.help "The github user of the author of the site"
                      ),
                siteGoogleScholarHandle =
                  Just
                    <$> Options.strOption
                      ( Options.long "google-scholar-handle"
                          <> Options.short 's'
                          <> Options.metavar "GOOGLE-SCHOLAR-HANDLE"
                          <> Options.help "The google scholar handle of the author of the site"
                      ),
                siteGithubRepository =
                  Just
                    <$> Options.option
                      (Options.maybeReader parseURI)
                      ( Options.long "github-repository"
                          <> Options.short 'r'
                          <> Options.metavar "GITHUB-REPOSITORY"
                          <> Options.help "The github repository of the site"
                      )
              }
        }

-- | error messages for missing site meta data
configErrors :: Config (Const String)
configErrors =
  Config
    { inputFolder = "input folder",
      outputFolder = "output folder ",
      siteMeta =
        SiteMeta
          { siteBaseUrl = "base url",
            siteTitle = "title",
            siteAuthor = "author",
            siteDescription = "site description",
            siteKeywords = "site keywords",
            siteTwitterHandle = "twitter handle",
            siteTwitchHandle = "twitch handle",
            siteYoutubeHandle = "youtube handle",
            siteGithubUser = "github user",
            siteGoogleScholarHandle = "google scholar handle",
            siteGithubRepository = "github repository"
          }
    }

newtype TagName = TagName String
  deriving stock (Generic, Eq, Ord, Show)
  deriving newtype (A.ToJSON, A.FromJSON, Binary)

-- | data type for article kinds
data ArticleKind = BlogPostKind | PublicationKind | SlideDeckKind
  deriving stock (Eq, Ord, Show, Generic)

-- | data type for publish status kinds
data PublishStatusKind = PublishedKind | DraftKind
  deriving stock (Eq, Ord, Show, Generic)

data PublicationField (s :: PublishStatusKind) where
  PubDate :: Date -> PublicationField 'PublishedKind
  PubDraft :: PublicationField 'DraftKind

deriving stock instance Show (PublicationField s)

deriving stock instance Eq (PublicationField s)

deriving stock instance Ord (PublicationField s)

instance Binary (PublicationField 'PublishedKind) where
  put :: PublicationField 'PublishedKind -> Put.Put
  put (PubDate d) = put d
  get :: Get.Get (PublicationField 'PublishedKind)
  get = PubDate <$> get

instance Binary (PublicationField 'DraftKind) where
  put :: PublicationField 'DraftKind -> Put.Put
  put PubDraft = pure ()
  get :: Get.Get (PublicationField 'DraftKind)
  get = pure PubDraft

instance A.FromJSON (PublicationField 'PublishedKind) where
  parseJSON :: Yaml.Value -> Yaml.Parser (PublicationField 'PublishedKind)
  parseJSON = A.withObject "PublicationField PublishedKind" $ \o -> do
    dateStr <- o A..: "date"
    date <- A.parseJSON dateStr
    pure (PubDate date)

instance A.FromJSON (PublicationField 'DraftKind) where
  parseJSON :: Yaml.Value -> Yaml.Parser (PublicationField 'DraftKind)
  parseJSON = A.withObject "PublicationField DraftKind" $ \_ -> pure PubDraft

instance A.ToJSON (PublicationField s) where
  toJSON :: PublicationField s -> Yaml.Value
  toJSON (PubDate date) =
    A.object
      [ "date" A..= date
      ]
  toJSON PubDraft =
    A.object
      []

-- | Date type that can be in human-readable format or UTC
data Date
  = HumanFormat String  -- "Oct 15, 2025"
  | UTC UTCTime         -- parsed timestamp for ISO8601 output
  deriving stock (Show, Generic)

-- | Equality based on the underlying time
instance Eq Date where
  (==) :: Date -> Date -> Bool
  a == b = dateToUTC a == dateToUTC b

-- | Ordering based on the underlying time
instance Ord Date where
  compare :: Date -> Date -> Ordering
  compare a b = compare (dateToUTC a) (dateToUTC b)

-- | Extract UTCTime from either variant
dateToUTC :: Date -> UTCTime
dateToUTC (UTC utc) = utc
dateToUTC (HumanFormat str) = parseTimeOrError True defaultTimeLocale "%b %e, %Y" str

-- | Convert any Date to UTC variant (for feeds)
toUTC :: Date -> Date
toUTC (HumanFormat str) = UTC (dateToUTC (HumanFormat str))
toUTC u@(UTC _) = u

-- | Convert an article's publication date to UTC format (for feeds)
articleToUTC :: Article kind 'PublishedKind -> Article kind 'PublishedKind
articleToUTC post@BlogPost{bpPublication = PubDate date} =
  post{bpPublication = PubDate (toUTC date)}
articleToUTC pub@Publication{pubPublication = PubDate date} =
  pub{pubPublication = PubDate (toUTC date)}
articleToUTC slideDeck@SlideDeck{sdPublication = PubDate date} =
  slideDeck{sdPublication = PubDate (toUTC date)}

-- | Convert a published article to UTC format (for feeds)
somePublishedArticleToUTC :: SomePublishedArticle -> SomePublishedArticle
somePublishedArticleToUTC (SomePublishedArticle article) =
  SomePublishedArticle (articleToUTC article)

instance A.FromJSON Date where
  parseJSON :: Yaml.Value -> Yaml.Parser Date
  parseJSON = A.withText "Date" $ \t -> do
    let str = T.unpack t
    -- Validate that it parses correctly
    case parseTimeM True defaultTimeLocale "%b %e, %Y" str of
      Nothing -> fail $ "Invalid date format: " ++ str ++ ". Expected format: 'Mon DD, YYYY' (e.g., 'Jan 16, 2022')"
      Just (_ :: UTCTime) -> pure (HumanFormat str)

instance A.ToJSON Date where
  toJSON :: Date -> Yaml.Value
  toJSON (HumanFormat str) = A.toJSON str
  toJSON (UTC utc) = A.toJSON (iso8601Show utc)

instance Binary Date where
  put :: Date -> Put.Put
  put (HumanFormat str) = Put.putWord8 0 >> put str
  put (UTC utc) = Put.putWord8 1 >> put (iso8601Show utc)
  get :: Get.Get Date
  get = do
    tag <- Get.getWord8
    case tag of
      0 -> HumanFormat <$> get
      1 -> do
        dateStr <- get
        case parseTimeM False defaultTimeLocale "%Y-%m-%dT%H:%M:%S%QZ" dateStr of
          Nothing -> fail $ "Failed to parse stored date: " ++ dateStr
          Just d -> pure (UTC d)
      _ -> fail $ "Unknown Date tag: " ++ show tag

-- | helper data type for non-empty lists
newtype Items a = Items {items :: NonEmpty a}
  deriving stock (Eq, Ord, Show, Generic)
  deriving anyclass (A.ToJSON, A.FromJSON, Binary)

-- | convert a non-empty list to a list
itemsToList :: Maybe (Items a) -> [a]
itemsToList = maybe [] (NonEmpty.toList . items)

-- | license info shown on a page
data LicenseInfo = LicenseInfo
  { name :: String,
    url :: Maybe String,
    note :: Maybe String
  }
  deriving stock (Generic, Eq, Ord, Show)

instance A.ToJSON LicenseInfo where toJSON = A.genericToJSON A.defaultOptions

instance A.FromJSON LicenseInfo where
  parseJSON (A.Object o) =
    LicenseInfo
      <$> o A..: "name"
      <*> o A..:? "url"
      <*> o A..:? "note"
  parseJSON (A.String s) =
    pure LicenseInfo {name = T.unpack s, url = Nothing, note = Nothing}
  parseJSON _ = fail "LicenseInfo must be a string or object"

instance Binary LicenseInfo where
  put LicenseInfo {..} = put name >> put url >> put note
  get = LicenseInfo <$> get <*> get <*> get

defaultCCBY :: LicenseInfo
defaultCCBY =
  LicenseInfo
    { name = "CC BY 4.0",
      url = Just "https://creativecommons.org/licenses/by/4.0/",
      note = Just "Please attribute \"Torsten Scholak\" with a link to the original."
    }

extractOrDefaultLicense :: (A.AsValue s) => FilePath -> s -> LicenseInfo
extractOrDefaultLicense src v =
  fromMaybe fallback (v ^? A.key "license" >>= asJSON)
  where
    asJSON x = case A.fromJSON x of
      A.Success a -> Just a
      _ -> Nothing
    lhsNote p
      | takeExtension p == ".lhs" = Just "Code blocks are BSD-3-Clause unless noted."
      | otherwise = Nothing
    fallback =
      defaultCCBY
        { note = case (note defaultCCBY, lhsNote src) of
            (Nothing, b) -> b
            (a, Nothing) -> a
            (Just a, Just b) -> Just (a <> " " <> b)
        }

-- | data type for articles
data Article (kind :: ArticleKind) (status :: PublishStatusKind) where
  BlogPost ::
    { bpTitle :: String,
      bpContent :: String,
      bpUrl :: String,
      bpPublication :: PublicationField status,
      bpTagNames :: Maybe (Items TagName),
      bpTeaser :: String,
      bpReadTime :: Int,
      bpGitHash :: String,
      bpImage :: Maybe String,
      bpLicense :: Maybe LicenseInfo,
      bpPrev :: Maybe (Article 'BlogPostKind status),
      bpNext :: Maybe (Article 'BlogPostKind status)
    } ->
    Article 'BlogPostKind status
  Publication ::
    { pubTitle :: String,
      pubAuthor :: [String],
      pubJournal :: String,
      pubContent :: String,
      pubUrl :: String,
      pubPublication :: PublicationField status,
      pubTagNames :: Maybe (Items TagName),
      pubTldr :: String,
      pubGitHash :: String,
      pubImage :: Maybe String,
      pubLink :: String,
      pubPDF :: Maybe String,
      pubCode :: Maybe String,
      pubTalk :: Maybe String,
      pubSlides :: Maybe String,
      pubPoster :: Maybe String,
      pubLicense :: Maybe LicenseInfo,
      pubPrev :: Maybe (Article 'PublicationKind status),
      pubNext :: Maybe (Article 'PublicationKind status)
    } ->
    Article 'PublicationKind status
  SlideDeck ::
    { sdTitle :: String,
      sdSubtitle :: Maybe String,
      sdAuthor :: Maybe String,
      sdInstitute :: Maybe String,
      sdDate :: Maybe String,
      sdContent :: String,
      sdUrl :: String,
      sdPublication :: PublicationField status,
      sdGitHash :: String,
      sdImage :: Maybe String,
      sdLicense :: Maybe LicenseInfo,
      sdPrev :: Maybe (Article 'SlideDeckKind status),
      sdNext :: Maybe (Article 'SlideDeckKind status)
    } ->
    Article 'SlideDeckKind status

deriving stock instance (Eq (PublicationField status)) => Eq (Article kind status)

deriving stock instance (Eq (PublicationField status), Ord (PublicationField status)) => Ord (Article kind status)

deriving stock instance (Show (PublicationField status)) => Show (Article kind status)

instance (Binary (PublicationField status), Binary (Maybe (Article 'BlogPostKind status))) => Binary (Article 'BlogPostKind status) where
  put :: Article 'BlogPostKind status -> Put.Put
  put BlogPost {..} = do
    put bpTitle
    put bpContent
    put bpUrl
    put bpPublication
    put bpTagNames
    put bpTeaser
    put bpReadTime
    put bpGitHash
    put bpImage
    put bpLicense
    put bpPrev
    put bpNext
  get :: Get.Get (Article 'BlogPostKind status)
  get =
    BlogPost
      <$> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get

instance (Binary (PublicationField status), Binary (Maybe (Article 'PublicationKind status))) => Binary (Article 'PublicationKind status) where
  put :: Article 'PublicationKind status -> Put.Put
  put Publication {..} = do
    put pubTitle
    put pubAuthor
    put pubJournal
    put pubContent
    put pubUrl
    put pubPublication
    put pubTagNames
    put pubTldr
    put pubGitHash
    put pubImage
    put pubLink
    put pubPDF
    put pubCode
    put pubTalk
    put pubSlides
    put pubPoster
    put pubLicense
    put pubPrev
    put pubNext
  get :: Get.Get (Article 'PublicationKind status)
  get =
    Publication
      <$> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get

instance (Binary (PublicationField status), Binary (Maybe (Article 'SlideDeckKind status))) => Binary (Article 'SlideDeckKind status) where
  put :: Article 'SlideDeckKind status -> Put.Put
  put SlideDeck {..} = do
    put sdTitle
    put sdSubtitle
    put sdAuthor
    put sdInstitute
    put sdDate
    put sdContent
    put sdUrl
    put sdPublication
    put sdGitHash
    put sdImage
    put sdLicense
    put sdPrev
    put sdNext
  get :: Get.Get (Article 'SlideDeckKind status)
  get =
    SlideDeck
      <$> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get
      <*> get

instance (A.ToJSON (PublicationField status), A.ToJSON (Maybe (Article 'BlogPostKind status))) => A.ToJSON (Article 'BlogPostKind status) where
  toJSON :: Article 'BlogPostKind status -> Yaml.Value
  toJSON BlogPost {..} =
    A.object
      [ "title" A..= bpTitle,
        "content" A..= bpContent,
        "url" A..= bpUrl,
        "publication" A..= bpPublication,
        "tags" A..= bpTagNames,
        "teaser" A..= bpTeaser,
        "readTime" A..= bpReadTime,
        "gitHash" A..= bpGitHash,
        "image" A..= bpImage,
        "license" A..= bpLicense,
        "prev" A..= bpPrev,
        "next" A..= bpNext
      ]

instance (A.ToJSON (PublicationField status), A.ToJSON (Maybe (Article 'PublicationKind status))) => A.ToJSON (Article 'PublicationKind status) where
  toJSON :: Article 'PublicationKind status -> Yaml.Value
  toJSON Publication {..} =
    A.object
      [ "title" A..= pubTitle,
        "author" A..= pubAuthor,
        "journal" A..= pubJournal,
        "content" A..= pubContent,
        "url" A..= pubUrl,
        "publication" A..= pubPublication,
        "tags" A..= pubTagNames,
        "tldr" A..= pubTldr,
        "gitHash" A..= pubGitHash,
        "image" A..= pubImage,
        "link" A..= pubLink,
        "pdf" A..= pubPDF,
        "code" A..= pubCode,
        "talk" A..= pubTalk,
        "slides" A..= pubSlides,
        "poster" A..= pubPoster,
        "license" A..= pubLicense,
        "prev" A..= pubPrev,
        "next" A..= pubNext
      ]

instance (A.ToJSON (PublicationField status), A.ToJSON (Maybe (Article 'SlideDeckKind status))) => A.ToJSON (Article 'SlideDeckKind status) where
  toJSON :: Article 'SlideDeckKind status -> Yaml.Value
  toJSON SlideDeck {..} =
    A.object
      [ "title" A..= sdTitle,
        "subtitle" A..= sdSubtitle,
        "author" A..= sdAuthor,
        "institute" A..= sdInstitute,
        "date" A..= sdDate,
        "content" A..= sdContent,
        "url" A..= sdUrl,
        "publication" A..= sdPublication,
        "gitHash" A..= sdGitHash,
        "image" A..= sdImage,
        "license" A..= sdLicense,
        "prev" A..= sdPrev,
        "next" A..= sdNext
      ]

instance (A.FromJSON (PublicationField status), A.FromJSON (Maybe (Article 'BlogPostKind status))) => A.FromJSON (Article 'BlogPostKind status) where
  parseJSON :: Yaml.Value -> Yaml.Parser (Article 'BlogPostKind status)
  parseJSON =
    A.withObject "Blog post" $ \o ->
      BlogPost
        <$> o A..: "title"
        <*> o A..: "content"
        <*> o A..: "url"
        <*> o A..: "publication"
        <*> o A..:? "tags"
        <*> o A..: "teaser"
        <*> o A..: "readTime"
        <*> o A..: "gitHash"
        <*> o A..:? "image"
        <*> o A..:? "license"
        <*> o A..:? "prev"
        <*> o A..:? "next"

instance (A.FromJSON (PublicationField status), A.FromJSON (Maybe (Article 'PublicationKind status))) => A.FromJSON (Article 'PublicationKind status) where
  parseJSON :: Yaml.Value -> Yaml.Parser (Article 'PublicationKind status)
  parseJSON =
    A.withObject "Publication" $ \o ->
      Publication
        <$> o A..: "title"
        <*> o A..: "author"
        <*> o A..: "journal"
        <*> o A..: "content"
        <*> o A..: "url"
        <*> o A..: "publication"
        <*> o A..:? "tags"
        <*> o A..: "tldr"
        <*> o A..: "gitHash"
        <*> o A..:? "image"
        <*> o A..: "link"
        <*> o A..:? "pdf"
        <*> o A..:? "code"
        <*> o A..:? "talk"
        <*> o A..:? "slides"
        <*> o A..:? "poster"
        <*> o A..:? "license"
        <*> o A..:? "prev"
        <*> o A..:? "next"

instance (A.FromJSON (PublicationField status), A.FromJSON (Maybe (Article 'SlideDeckKind status))) => A.FromJSON (Article 'SlideDeckKind status) where
  parseJSON :: Yaml.Value -> Yaml.Parser (Article 'SlideDeckKind status)
  parseJSON =
    A.withObject "SlideDeck" $ \o ->
      SlideDeck
        <$> o A..: "title"
        <*> o A..:? "subtitle"
        <*> o A..:? "author"
        <*> o A..:? "institute"
        <*> o A..:? "date"
        <*> o A..: "content"
        <*> o A..: "url"
        <*> o A..: "publication"
        <*> o A..: "gitHash"
        <*> o A..:? "image"
        <*> o A..:? "license"
        <*> o A..:? "prev"
        <*> o A..:? "next"

-- | assign next and previous articles based on the given order
assignAdjacentPublishedArticles :: forall kind. [Article kind 'PublishedKind] -> [Article kind 'PublishedKind]
assignAdjacentPublishedArticles posts =
  [ let prev = posts ^? ix (i + 1)
        next = posts ^? ix (i - 1)
        go Publication {} = cur {pubPrev = prev, pubNext = next}
        go BlogPost {} = cur {bpPrev = prev, bpNext = next}
        go SlideDeck {} = cur {sdPrev = prev, sdNext = next}
     in go cur
  | (cur, i) <- zip posts [0 :: Int ..]
  ]

-- | type-erased article with status hidden for a specific kind
data SomeStatusArticle kind where
  SomeStatusArticle :: Article kind status -> SomeStatusArticle kind

instance Show (SomeStatusArticle kind) where
  show :: SomeStatusArticle kind -> String
  show (SomeStatusArticle article) = case article of
    BlogPost {bpTitle, bpUrl} -> "SomeStatusArticle (BlogPost {bpTitle = " ++ show bpTitle ++ ", bpUrl = " ++ show bpUrl ++ ", ...})"
    Publication {pubTitle, pubUrl} -> "SomeStatusArticle (Publication {pubTitle = " ++ show pubTitle ++ ", pubUrl = " ++ show pubUrl ++ ", ...})"
    SlideDeck {sdTitle, sdUrl} -> "SomeStatusArticle (SlideDeck {sdTitle = " ++ show sdTitle ++ ", sdUrl = " ++ show sdUrl ++ ", ...})"

instance Binary (SomeStatusArticle 'BlogPostKind) where
  put :: SomeStatusArticle 'BlogPostKind -> Put.Put
  put (SomeStatusArticle article) = case article of
    BlogPost{bpPublication = PubDate _} -> do
      Put.putWord8 1  -- Tag for published
      put article
    BlogPost{bpPublication = PubDraft} -> do
      Put.putWord8 0  -- Tag for draft
      put article
  get :: Get.Get (SomeStatusArticle 'BlogPostKind)
  get = do
    tag <- Get.getWord8
    case tag of
      0 -> SomeStatusArticle <$> (get :: Get.Get (Article 'BlogPostKind 'DraftKind))
      1 -> SomeStatusArticle <$> (get :: Get.Get (Article 'BlogPostKind 'PublishedKind))
      _ -> fail $ "Unknown status tag: " ++ show tag

instance Binary (SomeStatusArticle 'PublicationKind) where
  put :: SomeStatusArticle 'PublicationKind -> Put.Put
  put (SomeStatusArticle article) = case article of
    Publication{pubPublication = PubDate _} -> do
      Put.putWord8 1  -- Tag for published
      put article
    Publication{pubPublication = PubDraft} -> do
      Put.putWord8 0  -- Tag for draft
      put article
  get :: Get.Get (SomeStatusArticle 'PublicationKind)
  get = do
    tag <- Get.getWord8
    case tag of
      0 -> SomeStatusArticle <$> (get :: Get.Get (Article 'PublicationKind 'DraftKind))
      1 -> SomeStatusArticle <$> (get :: Get.Get (Article 'PublicationKind 'PublishedKind))
      _ -> fail $ "Unknown status tag: " ++ show tag

instance Binary (SomeStatusArticle 'SlideDeckKind) where
  put :: SomeStatusArticle 'SlideDeckKind -> Put.Put
  put (SomeStatusArticle article) = case article of
    SlideDeck{sdPublication = PubDate _} -> do
      Put.putWord8 1  -- Tag for published
      put article
    SlideDeck{sdPublication = PubDraft} -> do
      Put.putWord8 0  -- Tag for draft
      put article
  get :: Get.Get (SomeStatusArticle 'SlideDeckKind)
  get = do
    tag <- Get.getWord8
    case tag of
      0 -> SomeStatusArticle <$> (get :: Get.Get (Article 'SlideDeckKind 'DraftKind))
      1 -> SomeStatusArticle <$> (get :: Get.Get (Article 'SlideDeckKind 'PublishedKind))
      _ -> fail $ "Unknown status tag: " ++ show tag

instance A.FromJSON (SomeStatusArticle 'BlogPostKind) where
  parseJSON :: Yaml.Value -> Yaml.Parser (SomeStatusArticle 'BlogPostKind)
  parseJSON v = A.withObject "Blog post" (\o -> do
    pub <- o A..: "publication"
    case pub of
      A.Object pubObj -> do
        status :: String <- pubObj A..: "status"
        case status of
          "published" -> do
            article <- A.parseJSON @(Article 'BlogPostKind 'PublishedKind) v
            return (SomeStatusArticle article)
          "draft" -> do
            article <- A.parseJSON @(Article 'BlogPostKind 'DraftKind) v
            return (SomeStatusArticle article)
          _ -> fail $ "Unknown publication status: " ++ status
      _ -> fail "Expected publication to be an object") v

instance A.FromJSON (SomeStatusArticle 'PublicationKind) where
  parseJSON :: Yaml.Value -> Yaml.Parser (SomeStatusArticle 'PublicationKind)
  parseJSON v = A.withObject "Publication" (\o -> do
    pub <- o A..: "publication"
    case pub of
      A.Object pubObj -> do
        status :: String <- pubObj A..: "status"
        case status of
          "published" -> do
            article <- A.parseJSON @(Article 'PublicationKind 'PublishedKind) v
            return (SomeStatusArticle article)
          "draft" -> do
            article <- A.parseJSON @(Article 'PublicationKind 'DraftKind) v
            return (SomeStatusArticle article)
          _ -> fail $ "Unknown publication status: " ++ status
      _ -> fail "Expected publication to be an object") v

instance A.FromJSON (SomeStatusArticle 'SlideDeckKind) where
  parseJSON :: Yaml.Value -> Yaml.Parser (SomeStatusArticle 'SlideDeckKind)
  parseJSON v = A.withObject "SlideDeck" (\o -> do
    pub <- o A..: "publication"
    case pub of
      A.Object pubObj -> do
        status :: String <- pubObj A..: "status"
        case status of
          "published" -> do
            article <- A.parseJSON @(Article 'SlideDeckKind 'PublishedKind) v
            return (SomeStatusArticle article)
          "draft" -> do
            article <- A.parseJSON @(Article 'SlideDeckKind 'DraftKind) v
            return (SomeStatusArticle article)
          _ -> fail $ "Unknown publication status: " ++ status
      _ -> fail "Expected publication to be an object") v

extractPublished ::
  SomeStatusArticle kind ->
  Maybe (Article kind 'PublishedKind, UTCTime)
extractPublished (SomeStatusArticle a) =
  case a of
    a'@BlogPost {bpPublication = PubDate date} -> Just (a', dateToUTC date)
    BlogPost {bpPublication = PubDraft} -> Nothing
    a'@Publication {pubPublication = PubDate date} -> Just (a', dateToUTC date)
    Publication {pubPublication = PubDraft} -> Nothing
    a'@SlideDeck {sdPublication = PubDate date} -> Just (a', dateToUTC date)
    SlideDeck {sdPublication = PubDraft} -> Nothing

-- | type-erased article with both kind and status hidden
-- Note: This only wraps published articles to ensure type safety
data SomePublishedArticle = forall kind. SomePublishedArticle (Article kind 'PublishedKind)

deriving stock instance Show SomePublishedArticle

withSomePublishedArticle :: forall a. SomePublishedArticle -> (forall kind. Article kind 'PublishedKind -> a) -> a
withSomePublishedArticle (SomePublishedArticle article) f = f article

instance Eq SomePublishedArticle where
  (==) :: SomePublishedArticle -> SomePublishedArticle -> Bool
  (SomePublishedArticle a@BlogPost {}) == (SomePublishedArticle b@BlogPost {}) = a == b
  (SomePublishedArticle a@Publication {}) == (SomePublishedArticle b@Publication {}) = a == b
  (SomePublishedArticle a@SlideDeck {}) == (SomePublishedArticle b@SlideDeck {}) = a == b
  _ == _ = False

instance Ord SomePublishedArticle where
  compare :: SomePublishedArticle -> SomePublishedArticle -> Ordering
  compare (SomePublishedArticle a@BlogPost {}) (SomePublishedArticle b@BlogPost {}) = compare a b
  compare (SomePublishedArticle BlogPost {}) (SomePublishedArticle Publication {}) = LT
  compare (SomePublishedArticle BlogPost {}) (SomePublishedArticle SlideDeck {}) = LT
  compare (SomePublishedArticle Publication {}) (SomePublishedArticle BlogPost {}) = GT
  compare (SomePublishedArticle a@Publication {}) (SomePublishedArticle b@Publication {}) = compare a b
  compare (SomePublishedArticle Publication {}) (SomePublishedArticle SlideDeck {}) = LT
  compare (SomePublishedArticle SlideDeck {}) (SomePublishedArticle BlogPost {}) = GT
  compare (SomePublishedArticle SlideDeck {}) (SomePublishedArticle Publication {}) = GT
  compare (SomePublishedArticle a@SlideDeck {}) (SomePublishedArticle b@SlideDeck {}) = compare a b

instance A.FromJSON SomePublishedArticle where
  parseJSON :: Yaml.Value -> Yaml.Parser SomePublishedArticle
  parseJSON =
    A.withObject "some article" $ \o -> do
      kind :: String <- o A..: "kind"
      case kind of
        "blog post" -> SomePublishedArticle <$> (A..:) @(Article 'BlogPostKind 'PublishedKind) o "article"
        "publication" -> SomePublishedArticle <$> (A..:) @(Article 'PublicationKind 'PublishedKind) o "article"
        "slide deck" -> SomePublishedArticle <$> (A..:) @(Article 'SlideDeckKind 'PublishedKind) o "article"
        _ -> fail "Expected blog post, publication, or slide deck"

instance A.ToJSON SomePublishedArticle where
  toJSON :: SomePublishedArticle -> Yaml.Value
  toJSON (SomePublishedArticle article) = case article of
    BlogPost {} -> A.object ["kind" A..= ("blog post" :: String), "article" A..= article]
    Publication {} -> A.object ["kind" A..= ("publication" :: String), "article" A..= article]
    SlideDeck {} -> A.object ["kind" A..= ("slide deck" :: String), "article" A..= article]

-- | helper data type for lists of articles of a given kind
newtype ArticlesInfo kind status = ArticlesInfo
  { articles :: [Article kind status]
  }
  deriving stock (Generic)

deriving stock instance (Eq (PublicationField status)) => Eq (ArticlesInfo kind status)

deriving stock instance (Eq (PublicationField status), Ord (PublicationField status)) => Ord (ArticlesInfo kind status)

deriving stock instance (Show (PublicationField status)) => Show (ArticlesInfo kind status)

instance (A.ToJSON (Article kind status)) => A.ToJSON (ArticlesInfo kind status) where
  toJSON :: (Yaml.ToJSON (Article kind status)) => ArticlesInfo kind status -> Yaml.Value
  toJSON ArticlesInfo {..} = A.object ["articles" A..= articles]

instance (A.FromJSON (Article kind status)) => A.FromJSON (ArticlesInfo kind status) where
  parseJSON :: (Yaml.FromJSON (Article kind status)) => Yaml.Value -> Yaml.Parser (ArticlesInfo kind status)
  parseJSON =
    A.withObject "ArticlesInfo" $ \o ->
      ArticlesInfo <$> o A..: "articles"

-- | data type for tags
data Tag = Tag
  { name :: TagName,
    articles :: [SomePublishedArticle],
    url :: String
  }
  deriving stock (Generic, Eq, Ord, Show)

tagJSONOptions :: A.Options
tagJSONOptions = A.defaultOptions {A.fieldLabelModifier = \case "name" -> "tag"; x -> x}

instance A.ToJSON Tag where
  toJSON :: Tag -> Yaml.Value
  toJSON = A.genericToJSON tagJSONOptions

instance A.FromJSON Tag where
  parseJSON :: Yaml.Value -> Yaml.Parser Tag
  parseJSON = A.genericParseJSON tagJSONOptions

newtype TagsInfo = TagsInfo
  { tags :: [Tag]
  }
  deriving stock (Generic, Eq, Ord, Show)
  deriving anyclass (A.ToJSON)

data FeedData = FeedData
  { title :: String,
    domain :: String,
    author :: String,
    articles :: [SomePublishedArticle],
    currentTime :: String,
    atomUrl :: String
  }
  deriving stock (Generic, Eq, Ord, Show)
  deriving anyclass (A.ToJSON)

data PageSpec = PageSpec
  { srcPath :: FilePath,
    templatePath :: FilePath,
    outPath :: FilePath,
    toHTML :: T.Text -> Action A.Value
  }

buildPage :: Config Identity -> PageSpec -> Action ()
buildPage config (PageSpec src tpl out xform) =
  cacheAction ("build" :: T.Text, src) $ do
    raw <- readFile' src
    body <- xform (T.pack raw)
    tplH <- Slick.compileTemplate' tpl
    gitHash <- getGitHash src >>= prettyGitHash config
    let withGitHash = A._Object . at "gitHash" ?~ A.String (T.pack gitHash)
        full = withSiteMeta config . withGitHash $ body
        html = T.unpack $ Slick.substitute tplH full
    writeFile' (runIdentity (outputFolder config) </> out) html

buildIndex :: Config Identity -> Action ()
buildIndex cfg =
  buildPage cfg $
    PageSpec
      { srcPath = "site/home.md",
        templatePath = "site/templates/index.html",
        outPath = "index.html",
        toHTML = markdownToHTML
      }

buildTerms :: Config Identity -> Action ()
buildTerms cfg =
  buildPage cfg $
    PageSpec
      { srcPath = "site/terms.md",
        templatePath = "site/templates/terms.html",
        outPath = "terms.html",
        toHTML = markdownToHTML
      }

buildContact :: Config Identity -> Action ()
buildContact cfg =
  buildPage cfg $
    PageSpec
      { srcPath = "site/Contact.lhs",
        templatePath = "site/templates/contact.html",
        outPath = "contact.html",
        toHTML = codeToHTML
      }

-- | find and build all blog posts
buildBlogPostList :: Config Identity -> Action [Article 'BlogPostKind 'PublishedKind]
buildBlogPostList config = do
  blogPostPaths <- getDirectoryFiles "." ["site/posts//*.md", "site/posts//*.lhs"]
  blogPosts <- forP blogPostPaths (buildBlogPost config)
  let published = mapMaybe extractPublished blogPosts
      blogPosts' = assignAdjacentPublishedArticles . map fst . sortOn (Down . snd) $ published
  _ <- forP blogPosts (writeBlogPost config) -- Write ALL posts, including drafts
  return blogPosts' -- But only return published ones for lists/tags/feeds

-- | build blog posts page
buildBlogPosts :: Config Identity -> [Article 'BlogPostKind 'PublishedKind] -> Action ()
buildBlogPosts config articles = do
  blogPostsTemplate <- Slick.compileTemplate' "site/templates/posts.html"
  let blogPostsInfo = ArticlesInfo {articles}
      blogPostsHTML = T.unpack $ Slick.substitute blogPostsTemplate (withSiteMeta config $ A.toJSON blogPostsInfo)
  writeFile' (runIdentity (outputFolder config) </> "posts.html") blogPostsHTML

-- | build a single blog post
buildBlogPost :: Config Identity -> FilePath -> Action (SomeStatusArticle 'BlogPostKind)
buildBlogPost config postSrcPath = cacheAction ("build" :: T.Text, postSrcPath) $ do
  postContent <- readFile' postSrcPath
  postData <- case takeExtension postSrcPath of
    ".md" -> markdownToHTML . T.pack $ postContent
    ".lhs" -> do
      compileAndRunCode postSrcPath
      codeToHTML . T.pack $ postContent
    _ -> fail "Expected .md or .lhs"
  gitHash <- getGitHash postSrcPath >>= prettyGitHash config
  let postUrl = T.pack . dropDirectory1 $ postSrcPath -<.> "html"
      withPostUrl = A._Object . at "url" ?~ A.String postUrl
      content = T.unpack $ fromMaybe mempty $ postData ^? A.key "content" . A._String
      withReadTime = A._Object . at "readTime" ?~ A.Integer (calcReadTime content)
      withGitHash = A._Object . at "gitHash" ?~ A.String (T.pack gitHash)
      withLicense = A._Object . at "license" ?~ A.toJSON (extractOrDefaultLicense postSrcPath postData)
      fullPostData = withReadTime . withGitHash . withPostUrl . withLicense $ postData
  Slick.convert fullPostData

-- | write blog post to file
writeBlogPost :: Config Identity -> SomeStatusArticle 'BlogPostKind -> Action ()
writeBlogPost config (SomeStatusArticle post@BlogPost {..}) = do
  postTemplate <- Slick.compileTemplate' "site/templates/post.html"
  writeFile' (runIdentity (outputFolder config) </> bpUrl)
    . T.unpack
    . Slick.substitute postTemplate
    . withSiteMeta config
    . A.toJSON
    $ post

-- | find and build all publications
buildPublicationList :: Config Identity -> Action [Article 'PublicationKind 'PublishedKind]
buildPublicationList config = do
  publicationPaths <- getDirectoryFiles "." ["site/publications//*.md"]
  publications <- forP publicationPaths (buildPublication config)
  let published = mapMaybe extractPublished publications
      publications' = assignAdjacentPublishedArticles . map fst . sortOn (Down . snd) $ published
  _ <- forP publications (writePublication config) -- Write ALL publications, including drafts
  return publications' -- But only return published ones for lists/tags/feeds

-- | build publications page
buildPublications :: Config Identity -> [Article 'PublicationKind 'PublishedKind] -> Action ()
buildPublications config articles = do
  publicationsTemplate <- Slick.compileTemplate' "site/templates/publications.html"
  let publicationsInfo = ArticlesInfo {articles}
      publicationsHTML = T.unpack $ Slick.substitute publicationsTemplate (withSiteMeta config $ A.toJSON publicationsInfo)
  writeFile' (runIdentity (outputFolder config) </> "publications.html") publicationsHTML

-- | build a single publication
buildPublication :: Config Identity -> FilePath -> Action (SomeStatusArticle 'PublicationKind)
buildPublication config publicationSrcPath = cacheAction ("build" :: T.Text, publicationSrcPath) $ do
  publicationContent <- readFile' publicationSrcPath
  publicationData <- markdownToHTML . T.pack $ publicationContent
  gitHash <- getGitHash publicationSrcPath >>= prettyGitHash config
  let publicationUrl = T.pack . dropDirectory1 $ publicationSrcPath -<.> "html"
      withPublicationUrl = A._Object . at "url" ?~ A.String publicationUrl
      withGitHash = A._Object . at "gitHash" ?~ A.String (T.pack gitHash)
      withLicense = A._Object . at "license" ?~ A.toJSON (extractOrDefaultLicense publicationSrcPath publicationData)
      fullPublicationData = withPublicationUrl . withGitHash . withLicense $ publicationData
  Slick.convert fullPublicationData

-- | write publication to file
writePublication :: Config Identity -> SomeStatusArticle 'PublicationKind -> Action ()
writePublication config (SomeStatusArticle publication@Publication {..}) = do
  publicationTemplate <- Slick.compileTemplate' "site/templates/publication.html"
  writeFile' (runIdentity (outputFolder config) </> pubUrl)
    . T.unpack
    . Slick.substitute publicationTemplate
    . withSiteMeta config
    . A.toJSON
    $ publication

-- | find and build all slide decks
buildSlideDeckList :: Config Identity -> Action [Article 'SlideDeckKind 'PublishedKind]
buildSlideDeckList config = do
  slideDeckPaths <- getDirectoryFiles "." ["site/slide-decks//*.md"]
  slideDecks <- forP slideDeckPaths (buildSlideDeck config)
  let published = mapMaybe extractPublished slideDecks
      slideDecks' = assignAdjacentPublishedArticles . map fst . sortOn (Down . snd) $ published
  _ <- forP slideDecks (writeSlideDeck config) -- Write ALL slide decks, including drafts
  return slideDecks' -- But only return published ones for lists

-- | build slide decks page
buildSlideDecks :: Config Identity -> [Article 'SlideDeckKind 'PublishedKind] -> Action ()
buildSlideDecks config articles = do
  slideDecksTemplate <- Slick.compileTemplate' "site/templates/slide-decks.html"
  let slideDecksInfo = ArticlesInfo {articles}
      slideDecksHTML = T.unpack $ Slick.substitute slideDecksTemplate (withSiteMeta config $ A.toJSON slideDecksInfo)
  writeFile' (runIdentity (outputFolder config) </> "slide-decks.html") slideDecksHTML

-- | build a single slide deck
buildSlideDeck :: Config Identity -> FilePath -> Action (SomeStatusArticle 'SlideDeckKind)
buildSlideDeck config slideDeckSrcPath = cacheAction ("build" :: T.Text, slideDeckSrcPath) $ do
  slideDeckContent <- readFile' slideDeckSrcPath
  slideDeckData <- markdownToRevealHTML . T.pack $ slideDeckContent
  gitHash <- getGitHash slideDeckSrcPath >>= prettyGitHash config
  let slideDeckUrl = T.pack . dropDirectory1 $ slideDeckSrcPath -<.> "html"
      withSlideDeckUrl = A._Object . at "url" ?~ A.String slideDeckUrl
      withGitHash = A._Object . at "gitHash" ?~ A.String (T.pack gitHash)
      withLicense = A._Object . at "license" ?~ A.toJSON (extractOrDefaultLicense slideDeckSrcPath slideDeckData)
      fullSlideDeckData = withSlideDeckUrl . withGitHash . withLicense $ slideDeckData
  Slick.convert fullSlideDeckData

-- | write slide deck to file
writeSlideDeck :: Config Identity -> SomeStatusArticle 'SlideDeckKind -> Action ()
writeSlideDeck config (SomeStatusArticle slideDeck@SlideDeck {..}) = do
  slideDeckTemplate <- Slick.compileTemplate' "site/templates/slide-deck.html"
  writeFile' (runIdentity (outputFolder config) </> sdUrl)
    . T.unpack
    . Slick.substitute slideDeckTemplate
    . withSiteMeta config
    . A.toJSON
    $ slideDeck

-- | find all tags and build tag pages
buildTagList :: Config Identity -> [SomePublishedArticle] -> Action [Tag]
buildTagList config articles =
  forP (Map.toList tags) (buildTag config . mkTag)
  where
    tags = Map.unionsWith (<>) ((`withSomePublishedArticle` collectTags) <$> articles)
    collectTags :: forall kind. Article kind 'PublishedKind -> Map TagName [SomePublishedArticle]
    collectTags post@BlogPost {bpTagNames} = Map.fromList $ (,pure $ SomePublishedArticle post) <$> itemsToList bpTagNames
    collectTags publication@Publication {pubTagNames} = Map.fromList $ (,pure $ SomePublishedArticle publication) <$> itemsToList pubTagNames
    collectTags SlideDeck {} = Map.empty  -- Slide decks don't have tags
    mkTag (tagName@(TagName name), articles') =
      Tag
        { name = tagName,
          articles = sortOn (`withSomePublishedArticle` comp) articles',
          url = "tags/" <> name <> ".html"
        }
    comp :: forall kind. Article kind 'PublishedKind -> Down UTCTime
    comp Publication {pubPublication = PubDate date} = Down (dateToUTC date)
    comp BlogPost {bpPublication = PubDate date} = Down (dateToUTC date)
    comp SlideDeck {sdPublication = PubDate date} = Down (dateToUTC date)

-- | build tags page
buildTags :: Config Identity -> [Tag] -> Action ()
buildTags config tags = do
  tagsTemplate <- Slick.compileTemplate' "site/templates/tags.html"
  let tagsInfo = TagsInfo {tags}
      tagsHTML = T.unpack $ Slick.substitute tagsTemplate (withSiteMeta config $ A.toJSON tagsInfo)
  writeFile' (runIdentity (outputFolder config) </> "tags.html") tagsHTML

-- | build a single tag page
buildTag :: Config Identity -> Tag -> Action Tag
buildTag config tag@Tag {..} =
  do
    tagTemplate <- Slick.compileTemplate' "site/templates/tag.html"
    let tagData = withSiteMeta config $ A.toJSON tag
        tagHTML = T.unpack $ Slick.substitute tagTemplate tagData
    writeFile' (runIdentity (outputFolder config) </> url) tagHTML
    Slick.convert tagData

-- | calculate read time of a post based on the number of words
-- and the average reading speed of around 200 words per minute
calcReadTime :: String -> Integer
calcReadTime = fromIntegral . uncurry roundUp . flip divMod 200 . length . words
  where
    roundUp mins secs = mins + if secs == 0 then 0 else 1

-- | build resume page
buildResume :: Config Identity -> Action ()
buildResume config = cacheAction ("build" :: T.Text, resumeSrcPath) $ do
  resumeYamlBS <- liftIO $ BS.readFile resumeSrcPath
  resumeTemplate <- Slick.compileTemplate' "site/templates/resume.html"
  gitHash <- getGitHash resumeSrcPath >>= prettyGitHash config
  let resumeData = case Yaml.decodeEither' resumeYamlBS of
        Left err -> error $ "Failed to parse YAML: " ++ show err
        Right val -> addUrls val
  let withGitHash = A._Object . at "gitHash" ?~ A.String (T.pack gitHash)
      fullResumeData = withSiteMeta config . withGitHash $ resumeData
      resumeHTML = T.unpack $ Slick.substitute resumeTemplate fullResumeData
  writeFile' (runIdentity (outputFolder config) </> "resume.html") resumeHTML
  where
    resumeSrcPath :: FilePath
    resumeSrcPath = "site/resume.yaml"

    addUrls (A.Object obj) = case KM.lookup "cv" obj of
      Just (A.Object cv) -> case KM.lookup "social_networks" cv of
        Just (A.Array networks) ->
          let networks' = fmap addUrl networks
              cv' = KM.insert "social_networks" (A.Array networks') cv
           in A.Object $ KM.insert "cv" (A.Object cv') obj
        _ -> A.Object obj
      _ -> A.Object obj
    addUrls v = v

    addUrl (A.Object net) = case (KM.lookup "network" net, KM.lookup "username" net) of
      (Just (A.String "GitHub"), Just (A.String user)) ->
        A.Object $ KM.insert "url" (A.String $ "https://github.com/" <> user) net
      (Just (A.String "LinkedIn"), Just (A.String user)) ->
        A.Object $ KM.insert "url" (A.String $ "https://linkedin.com/in/" <> user) net
      (Just (A.String "X"), Just (A.String user)) ->
        A.Object $ KM.insert "url" (A.String $ "https://x.com/" <> user) net
      (Just (A.String "Google Scholar"), Just (A.String user)) ->
        A.Object $ KM.insert "url" (A.String $ "https://scholar.google.com/citations?user=" <> user) net
      _ -> A.Object net
    addUrl v = v

-- | copy all static files
copyStaticFiles :: Config Identity -> Action ()
copyStaticFiles config = do
  staticFilePaths <- getDirectoryFiles "." ["site/images//*", "site/css//*", "site/js//*", "site/fonts//*"]
  void $
    forP staticFilePaths $ \src -> do
      let dest = runIdentity (outputFolder config) </> dropDirectory1 src
      copyFileChanged src dest

-- | convert UTC time to RFC 3339 format
toIsoDate :: UTCTime -> String
toIsoDate = iso8601Show

-- | build feed
buildFeed :: Config Identity -> [SomePublishedArticle] -> Action ()
buildFeed config articles = do
  now <- liftIO getCurrentTime
  let feedData =
        FeedData
          { title = runIdentity . siteTitle . siteMeta $ config,
            domain = show . runIdentity . siteBaseUrl . siteMeta $ config,
            author = runIdentity . siteAuthor . siteMeta $ config,
            articles = somePublishedArticleToUTC <$> articles,  -- Convert dates to UTC for Atom feed
            currentTime = toIsoDate now,
            atomUrl = "/feed.xml"
          }
  feedTemplate <- Slick.compileTemplate' "site/templates/feed.xml"
  writeFile' (runIdentity (outputFolder config) </> "feed.xml") . T.unpack $ Slick.substitute feedTemplate (A.toJSON feedData)

-- | build site using all actions
buildRules :: Config Identity -> Action ()
buildRules config = do
  buildIndex config
  posts <- buildBlogPostList config
  buildBlogPosts config posts
  publications <- buildPublicationList config
  buildPublications config publications
  slideDecks <- buildSlideDeckList config
  buildSlideDecks config slideDecks
  let articles = (SomePublishedArticle <$> posts) <> (SomePublishedArticle <$> publications) <> (SomePublishedArticle <$> slideDecks)
  tags <- buildTagList config articles
  buildTags config tags
  buildContact config
  buildResume config
  buildTerms config
  copyStaticFiles config
  buildFeed config articles

-- | data type for git hash
data GitHash
  = Committed {gitHash :: String, gitDate :: String, gitAuthor :: String, gitMessage :: String}
  | Uncommitted
  deriving (Eq, Show)

-- | get git hash of last commit of a file
getGitHash :: FilePath -> Action GitHash
getGitHash path = do
  let cmdLine =
        unwords
          [ "git",
            "log",
            "--pretty=format:%h%n%ci%n%an%n%s",
            "-n",
            "1",
            "--",
            path
          ]
  Stdout (gitInfo :: String) <- cmd cmdLine
  case lines gitInfo of
    [hash, date, author, subject] -> pure $ Committed hash date author subject
    [] -> pure Uncommitted
    _ -> fail $ "Unexpected git log output for " ++ path

-- | pretty print git hash with link to GitHub commit
prettyGitHash :: Config Identity -> GitHash -> Action String
prettyGitHash config (Committed {..}) = do
  Just uri <- pure $ do
    repo <- runIdentity . siteGithubRepository . siteMeta $ config
    return $ repo {uriPath = uriPath repo <> "/commit/" <> gitHash}
  let link = "<a href=\"" <> show uri <> "\">" <> gitHash <> "</a>"
  return $ "commit " <> link <> " (" <> gitDate <> ") " <> gitAuthor <> ": " <> gitMessage
prettyGitHash _ Uncommitted = return "uncommitted changes"

-- | parser info for command line arguments
parserInfo ::
  forall b f.
  (Barbie.TraversableB b) =>
  b (Options.Parser `Compose` f) ->
  Options.ParserInfo (b f, Maybe FilePath)
parserInfo b =
  let parser =
        (,)
          <$> Barbie.bsequence b
          <*> optional
            ( Options.option Options.str $
                Options.long "config-file"
                  <> Options.short 'f'
                  <> Options.metavar "CONFIG_FILE"
                  <> Options.help "Config file name"
            )
   in Options.info (parser <**> Options.helper) $
        Options.fullDesc
          <> Options.progDesc "Build site"
          <> Options.header "website - a blog and website generator"

-- | validate config
validate ::
  forall b.
  (Barbie.TraversableB b, Barbie.ApplicativeB b) =>
  b (Const String) ->
  b Maybe ->
  Validation [String] (b Identity)
validate errorMessages mb =
  Barbie.bsequence' $ Barbie.bzipWith go mb errorMessages
  where
    go :: forall a. Maybe a -> Const String a -> Validation [String] a
    go (Just a) _ = Validation.Success a
    go Nothing (Const errorMessage) = Validation.Failure [errorMessage]

-- | read config from file
readConfigFile :: FilePath -> IO A.Value
readConfigFile filePath = do
  res <- Atto.parseOnly A.jsonEOF' <$> BS.readFile filePath
  case res of
    Left _err -> error "Failed to parse config file"
    Right val -> pure val

-- | decode config from JSON value
jsonConfig :: forall b. (Show (b Maybe), A.FromJSON (b Maybe), Barbie.ApplicativeB b) => A.Value -> b Maybe
jsonConfig = fromResult . A.fromJSON
  where
    fromResult (A.Success a) = a
    fromResult (A.Error _) = Barbie.bpure Nothing

-- | parse command line arguments
parseConfig :: IO (Validation [String] (Config Identity))
parseConfig = do
  (c, mConfigFile) <- Options.execParser $ parserInfo configParser
  validate configErrors
    <$> maybe
      (pure c)
      (fmap ((c <>) . jsonConfig) . readConfigFile)
      mConfigFile

-- | main program
main :: IO ()
main =
  parseConfig >>= \case
    Validation.Failure errorMessageList -> do
      error $
        "The following options are required but not provided: "
          <> List.intercalate ", " errorMessageList
    Validation.Success config -> do
      let options = shakeOptions {shakeVerbosity = Chatty, shakeLintInside = ["\\"]}
      shakeForward options (buildRules config)
