{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
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
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Main (main) where

import qualified Barbies as Barbie
import Control.Applicative (Alternative (empty, (<|>)), Const (Const), optional, (<**>))
import Control.Lens (at, ix, (?~), (^?))
import Control.Monad (void)
import Control.Monad.IO.Class (MonadIO (liftIO))
import Data.Aeson (toJSON)
import qualified Data.Aeson as A (FromJSON (parseJSON), KeyValue ((.=)), Options (..), Result (Error, Success), ToJSON (toEncoding, toJSON), Value (..), decode', defaultOptions, fromJSON, genericParseJSON, genericToEncoding, genericToJSON, object, withObject, (.:), (.:?))
import qualified Data.Aeson.Lens as A (AsPrimitive (_String), AsValue (_Object), key, pattern Integer)
import qualified Data.Aeson.Parser.Internal as A (jsonEOF')
import qualified Data.Attoparsec.ByteString as Atto
import qualified Data.ByteString as BS
import Data.Either.Validation (Validation)
import qualified Data.Either.Validation as Validation
import Data.Functor.Compose (Compose (Compose))
import Data.Functor.Identity (Identity (..))
import qualified Data.HashMap.Lazy as HML
import Data.List (sortOn)
import qualified Data.List as List
import Data.List.NonEmpty (NonEmpty)
import qualified Data.List.NonEmpty as NonEmpty
import Data.Map.Lazy (Map)
import qualified Data.Map.Lazy as Map
import Data.Maybe (fromMaybe)
import Data.Ord (Down (Down))
import qualified Data.Text as T
import qualified Data.Text.Lazy as TL
import qualified Data.Text.Lazy.Encoding as TL
import Data.Time (UTCTime, defaultTimeLocale, formatTime, getCurrentTime, iso8601DateFormat, parseTimeOrError)
import Data.Vector as V (mapMaybe)
import Development.Shake (Action, ShakeOptions (..), copyFileChanged, forP, getDirectoryFiles, readFile', shakeOptions, writeFile', pattern Chatty)
import Development.Shake.Classes (Binary (..))
import Development.Shake.FilePath (dropDirectory1, takeExtension, (-<.>), (</>))
import Development.Shake.Forward (cacheAction, shakeForward)
import GHC.Generics (Generic)
import Network.URI (URI (uriPath), parseURI)
import Network.URI.JSON ()
import qualified Options.Applicative as Options
import Slick (compileTemplate', convert, substitute)
import Slick.Pandoc (defaultHtml5Options, markdownToHTMLWithOpts)
import System.Process (readProcess)
import Text.Casing (fromHumps, toQuietSnake)
import qualified Text.Pandoc as P

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
  (<>) = Barbie.bzipWith (<|>)

instance (Alternative f) => Monoid (Config f) where
  mempty = Barbie.bpure empty

configCustomJSONOptions :: A.Options
configCustomJSONOptions = A.defaultOptions {A.fieldLabelModifier = toQuietSnake . fromHumps}

instance (Barbie.AllBF A.FromJSON f Config) => A.FromJSON (Config f) where
  parseJSON = A.genericParseJSON configCustomJSONOptions

instance (Barbie.AllBF A.ToJSON f Config) => A.ToJSON (Config f) where
  toJSON = A.genericToJSON configCustomJSONOptions
  toEncoding = A.genericToEncoding configCustomJSONOptions

-- | markdown options
markdownOptions :: P.ReaderOptions
markdownOptions =
  P.def {P.readerExtensions = pandocExtensions}
  where
    pandocExtensions =
      P.disableExtension P.Ext_autolink_bare_uris $
        mconcat
          [ P.extensionsFromList
              [ P.Ext_yaml_metadata_block,
                P.Ext_fenced_code_attributes,
                P.Ext_auto_identifiers
              ],
            P.githubMarkdownExtensions
          ]

-- | convert markdown to html
markdownToHTML :: T.Text -> Action A.Value
markdownToHTML = markdownToHTMLWithOpts markdownOptions defaultHtml5Options

-- | convert literal Haskell code to html
codeToHTML :: T.Text -> Action A.Value
codeToHTML = markdownToHTMLWithOpts opts defaultHtml5Options
  where
    opts = P.def {P.readerExtensions = pandocExtensions}
    pandocExtensions =
      P.extensionsFromList
        [ P.Ext_literate_haskell,
          P.Ext_yaml_metadata_block
        ]

-- | add site meta data to a JSON object
withSiteMeta :: Config Identity -> A.Value -> A.Value
withSiteMeta config (A.Object obj) = A.Object $ HML.union obj siteMetaObj
  where
    A.Object siteMetaObj = A.toJSON . siteMeta $ config
withSiteMeta _ v = error $ "only add site meta to objects, not " ++ show v

-- | Site meta data
data SiteMeta f = SiteMeta
  { siteBaseUrl :: !(f URI), -- e.g. https://example.ca
    siteTitle :: !(f String),
    siteAuthor :: !(f String),
    siteDescription :: !(f String),
    siteKeywords :: !(f String),
    siteTwitterHandle :: !(f (Maybe String)), -- Without @
    siteTwitchHandle :: !(f (Maybe String)),
    siteYoutubeHandle :: !(f (Maybe String)),
    siteGithubUser :: !(f (Maybe String)),
    siteGithubRepository :: !(f (Maybe URI))
  }
  deriving stock (Generic)
  deriving anyclass (Barbie.FunctorB, Barbie.TraversableB, Barbie.ApplicativeB, Barbie.ConstraintsB)

deriving stock instance (Barbie.AllBF Show f SiteMeta) => Show (SiteMeta f)

deriving stock instance (Barbie.AllBF Eq f SiteMeta) => Eq (SiteMeta f)

instance (Alternative f) => Semigroup (SiteMeta f) where
  (<>) = Barbie.bzipWith (<|>)

instance (Alternative f) => Monoid (SiteMeta f) where
  mempty = Barbie.bpure empty

siteMetaCustomJSONOptions :: A.Options
siteMetaCustomJSONOptions = A.defaultOptions {A.fieldLabelModifier = toQuietSnake . fromHumps . drop 4}

instance (Barbie.AllBF A.FromJSON f SiteMeta) => A.FromJSON (SiteMeta f) where
  parseJSON = A.genericParseJSON siteMetaCustomJSONOptions

instance (Barbie.AllBF A.ToJSON f SiteMeta) => A.ToJSON (SiteMeta f) where
  toJSON = A.genericToJSON siteMetaCustomJSONOptions
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
            siteGithubRepository = "github repository"
          }
    }

newtype TagName = TagName String
  deriving stock (Generic, Eq, Ord, Show)

instance A.FromJSON TagName where
  parseJSON v = TagName <$> A.parseJSON v

instance A.ToJSON TagName where
  toJSON (TagName tagName) = A.toJSON tagName

instance Binary TagName where
  put (TagName tagName) = put tagName
  get = TagName <$> get

-- | data type for article kinds
data ArticleKind = BlogPostKind | PublicationKind
  deriving stock (Eq, Ord, Show, Generic)

-- | extract the article kind from an article of a given kind
articleKind :: forall kind. Article kind -> ArticleKind
articleKind BlogPost {} = BlogPostKind
articleKind Publication {} = PublicationKind

-- | helper data type for non-empty lists
newtype Items a = Items {items :: NonEmpty a}
  deriving stock (Eq, Ord, Show, Generic)
  deriving anyclass (A.ToJSON, A.FromJSON, Binary)

-- | convert a non-empty list to a list
itemsToList :: Maybe (Items a) -> [a]
itemsToList = maybe [] (NonEmpty.toList . items)

-- | data type for articles
data Article kind where
  BlogPost ::
    { bpTitle :: String,
      bpContent :: String,
      bpUrl :: String,
      bpDate :: String,
      bpTagNames :: Maybe (Items TagName),
      bpTeaser :: String,
      bpReadTime :: Int,
      bpGitHash :: String,
      bpImage :: Maybe String,
      bpPrev :: Maybe (Article 'BlogPostKind),
      bpNext :: Maybe (Article 'BlogPostKind)
    } ->
    Article 'BlogPostKind
  Publication ::
    { pubTitle :: String,
      pubAuthor :: [String],
      pubJournal :: String,
      pubContent :: String,
      pubUrl :: String,
      pubDate :: String,
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
      pubPrev :: Maybe (Article 'PublicationKind),
      pubNext :: Maybe (Article 'PublicationKind)
    } ->
    Article 'PublicationKind

deriving stock instance Eq (Article kind)

deriving stock instance Ord (Article kind)

deriving stock instance Show (Article kind)

instance Binary (Article 'BlogPostKind) where
  put BlogPost {..} = do
    put bpTitle
    put bpContent
    put bpUrl
    put bpDate
    put bpTagNames
    put bpTeaser
    put bpReadTime
    put bpGitHash
    put bpImage
    put bpPrev
    put bpNext
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

instance Binary (Article 'PublicationKind) where
  put Publication {..} = do
    put pubTitle
    put pubAuthor
    put pubJournal
    put pubContent
    put pubUrl
    put pubDate
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
    put pubPrev
    put pubNext
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

instance A.ToJSON (Article 'BlogPostKind) where
  toJSON BlogPost {..} =
    A.object
      [ "title" A..= bpTitle,
        "content" A..= bpContent,
        "url" A..= bpUrl,
        "date" A..= bpDate,
        "tags" A..= bpTagNames,
        "teaser" A..= bpTeaser,
        "readTime" A..= bpReadTime,
        "gitHash" A..= bpGitHash,
        "image" A..= bpImage,
        "prev" A..= bpPrev,
        "next" A..= bpNext
      ]

instance A.ToJSON (Article 'PublicationKind) where
  toJSON Publication {..} =
    A.object
      [ "title" A..= pubTitle,
        "author" A..= pubAuthor,
        "journal" A..= pubJournal,
        "content" A..= pubContent,
        "url" A..= pubUrl,
        "date" A..= pubDate,
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
        "prev" A..= pubPrev,
        "next" A..= pubNext
      ]

instance A.FromJSON (Article 'BlogPostKind) where
  parseJSON =
    A.withObject "Blog post" $ \o ->
      BlogPost
        <$> o A..: "title"
        <*> o A..: "content"
        <*> o A..: "url"
        <*> o A..: "date"
        <*> o A..:? "tags"
        <*> o A..: "teaser"
        <*> o A..: "readTime"
        <*> o A..: "gitHash"
        <*> o A..:? "image"
        <*> o A..:? "prev"
        <*> o A..:? "next"

instance A.FromJSON (Article 'PublicationKind) where
  parseJSON =
    A.withObject "Publication" $ \o ->
      Publication
        <$> o A..: "title"
        <*> o A..: "author"
        <*> o A..: "journal"
        <*> o A..: "content"
        <*> o A..: "url"
        <*> o A..: "date"
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
        <*> o A..:? "prev"
        <*> o A..:? "next"

-- | assign next and previous articles based on the given order
assignAdjacentArticles :: forall kind. [Article kind] -> [Article kind]
assignAdjacentArticles posts =
  [ let prev = posts ^? ix (i + 1)
        next = posts ^? ix (i - 1)
        go Publication {} = cur {pubPrev = prev, pubNext = next}
        go BlogPost {} = cur {bpPrev = prev, bpNext = next}
     in go cur
    | (cur, i) <- zip posts [0 :: Int ..]
  ]

-- | type-erased article
data SomeArticle = forall kind. SomeArticle (Article kind)

deriving stock instance Show SomeArticle

withSomeArticle :: forall a. SomeArticle -> (forall kind. Article kind -> a) -> a
withSomeArticle (SomeArticle article) f = f article

instance Eq SomeArticle where
  (SomeArticle a@BlogPost {}) == (SomeArticle b@BlogPost {}) = a == b
  (SomeArticle a@Publication {}) == (SomeArticle b@Publication {}) = a == b
  _ == _ = False

instance Ord SomeArticle where
  compare (SomeArticle a@BlogPost {}) (SomeArticle b@BlogPost {}) = compare a b
  compare (SomeArticle BlogPost {}) (SomeArticle Publication {}) = LT
  compare (SomeArticle Publication {}) (SomeArticle BlogPost {}) = GT
  compare (SomeArticle a@Publication {}) (SomeArticle b@Publication {}) = compare a b

instance A.FromJSON SomeArticle where
  parseJSON =
    A.withObject "some article" $ \o -> do
      kind :: String <- o A..: "kind"
      case kind of
        "blog post" -> SomeArticle <$> (A..:) @(Article 'BlogPostKind) o "article"
        "publication" -> SomeArticle <$> (A..:) @(Article 'PublicationKind) o "article"
        _ -> fail "Expected blog post or publication"

instance A.ToJSON SomeArticle where
  toJSON (SomeArticle article) = case article of
    BlogPost {} -> A.object ["kind" A..= ("blog post" :: String), "article" A..= article]
    Publication {} -> A.object ["kind" A..= ("publication" :: String), "article" A..= article]

-- | helper data type for lists of articles of a given kind
newtype ArticlesInfo kind = ArticlesInfo
  { articles :: [Article kind]
  }
  deriving stock (Generic, Eq, Ord, Show)

instance A.ToJSON (Article kind) => A.ToJSON (ArticlesInfo kind) where
  toJSON ArticlesInfo {..} = A.object ["articles" A..= articles]

instance A.FromJSON (Article kind) => A.FromJSON (ArticlesInfo kind) where
  parseJSON =
    A.withObject "ArticlesInfo" $ \o ->
      ArticlesInfo <$> o A..: "articles"

-- | data type for tags
data Tag = Tag
  { name :: TagName,
    articles :: [SomeArticle],
    url :: String
  }
  deriving stock (Generic, Eq, Ord, Show)

instance A.ToJSON Tag where
  toJSON Tag {..} =
    A.object
      [ "tag" A..= name,
        "articles" A..= articles,
        "url" A..= url
      ]

instance A.FromJSON Tag where
  parseJSON =
    A.withObject "Tag" $ \o ->
      Tag
        <$> o A..: "tag"
        <*> o A..: "articles"
        <*> o A..: "url"

newtype TagsInfo = TagsInfo
  { tags :: [Tag]
  }
  deriving stock (Generic, Eq, Ord, Show)
  deriving anyclass (A.ToJSON)

data FeedData = FeedData
  { title :: String,
    domain :: String,
    author :: String,
    articles :: [SomeArticle],
    currentTime :: String,
    atomUrl :: String
  }
  deriving stock (Generic, Eq, Ord, Show)
  deriving anyclass (A.ToJSON)

-- | build landing page
buildIndex :: Config Identity -> Action ()
buildIndex config = cacheAction ("build" :: T.Text, indexSrcPath) $ do
  indexContent <- readFile' indexSrcPath
  indexData <- markdownToHTML . T.pack $ indexContent
  indexTemplate <- compileTemplate' "site/templates/index.html"
  gitHash <- getGitHash indexSrcPath >>= prettyGitHash config
  let withGitHash = A._Object . at "gitHash" ?~ A.String (T.pack gitHash)
      fullIndexData = withSiteMeta config . withGitHash $ indexData
      indexHTML = T.unpack $ substitute indexTemplate fullIndexData
  writeFile' (runIdentity (outputFolder config) </> "index.html") indexHTML
  where
    indexSrcPath :: FilePath
    indexSrcPath = "site/home.md"

-- | find and build all blog posts
buildBlogPostList :: Config Identity -> Action [Article 'BlogPostKind]
buildBlogPostList config = do
  blogPostPaths <- getDirectoryFiles "." ["site/posts//*.md", "site/posts//*.lhs"]
  blogPosts <- forP blogPostPaths (buildBlogPost config)
  let blogPosts' = assignAdjacentArticles . sortOn (Down . parseDate . bpDate) $ blogPosts
  _ <- forP blogPosts' (writeBlogPost config)
  return blogPosts'

-- | build blog posts page
buildBlogPosts :: Config Identity -> [Article 'BlogPostKind] -> Action ()
buildBlogPosts config articles = do
  blogPostsTemplate <- compileTemplate' "site/templates/posts.html"
  let blogPostsInfo = ArticlesInfo {articles}
      blogPostsHTML = T.unpack $ substitute blogPostsTemplate (withSiteMeta config $ A.toJSON blogPostsInfo)
  writeFile' (runIdentity (outputFolder config) </> "posts.html") blogPostsHTML

-- | build a single blog post
buildBlogPost :: Config Identity -> FilePath -> Action (Article 'BlogPostKind)
buildBlogPost config postSrcPath = cacheAction ("build" :: T.Text, postSrcPath) $ do
  postContent <- readFile' postSrcPath
  postData <- case takeExtension postSrcPath of
    ".md" -> markdownToHTML . T.pack $ postContent
    ".lhs" -> codeToHTML . T.pack $ postContent
    _ -> fail "Expected .md or .lhs"
  gitHash <- getGitHash postSrcPath >>= prettyGitHash config
  let postUrl = T.pack . dropDirectory1 $ postSrcPath -<.> "html"
      withPostUrl = A._Object . at "url" ?~ A.String postUrl
      content = T.unpack $ fromMaybe mempty $ postData ^? A.key "content" . A._String
      withReadTime = A._Object . at "readTime" ?~ A.Integer (calcReadTime content)
      withGitHash = A._Object . at "gitHash" ?~ A.String (T.pack gitHash)
      fullPostData = withSiteMeta config . withReadTime . withGitHash . withPostUrl $ postData
  convert fullPostData

-- | write blog post to file
writeBlogPost :: Config Identity -> Article 'BlogPostKind -> Action ()
writeBlogPost config post@BlogPost {..} = do
  postTemplate <- compileTemplate' "site/templates/post.html"
  writeFile' (runIdentity (outputFolder config) </> bpUrl) . T.unpack . substitute postTemplate $ A.toJSON post

-- | find and build all publications
buildPublicationList :: Config Identity -> Action [Article 'PublicationKind]
buildPublicationList config = do
  publicationPaths <- getDirectoryFiles "." ["site/publications//*.md"]
  publications <- forP publicationPaths (buildPublication config)
  let publications' = assignAdjacentArticles . sortOn (Down . parseDate . pubDate) $ publications
  _ <- forP publications' (writePublication config)
  return publications'

-- | build publications page
buildPublications :: Config Identity -> [Article 'PublicationKind] -> Action ()
buildPublications config articles = do
  publicationsTemplate <- compileTemplate' "site/templates/publications.html"
  let publicationsInfo = ArticlesInfo {articles}
      publicationsHTML = T.unpack $ substitute publicationsTemplate (withSiteMeta config $ A.toJSON publicationsInfo)
  writeFile' (runIdentity (outputFolder config) </> "publications.html") publicationsHTML

-- | build a single publication
buildPublication :: Config Identity -> FilePath -> Action (Article 'PublicationKind)
buildPublication config publicationSrcPath = cacheAction ("build" :: T.Text, publicationSrcPath) $ do
  publicationContent <- readFile' publicationSrcPath
  publicationData <- markdownToHTML . T.pack $ publicationContent
  gitHash <- getGitHash publicationSrcPath >>= prettyGitHash config
  let publicationUrl = T.pack . dropDirectory1 $ publicationSrcPath -<.> "html"
      withPublicationUrl = A._Object . at "url" ?~ A.String publicationUrl
      withGitHash = A._Object . at "gitHash" ?~ A.String (T.pack gitHash)
      fullPublicationData = withSiteMeta config . withPublicationUrl . withGitHash $ publicationData
  convert fullPublicationData

-- | write publication to file
writePublication :: Config Identity -> Article 'PublicationKind -> Action ()
writePublication config publication@Publication {..} = do
  publicationTemplate <- compileTemplate' "site/templates/publication.html"
  writeFile' (runIdentity (outputFolder config) </> pubUrl) . T.unpack . substitute publicationTemplate $ A.toJSON publication

-- | find all tags and build tag pages
buildTagList :: Config Identity -> [SomeArticle] -> Action [Tag]
buildTagList config articles =
  forP (Map.toList tags) (buildTag config . mkTag)
  where
    tags = Map.unionsWith (<>) ((`withSomeArticle` collectTags) <$> articles)
    collectTags :: forall kind. Article kind -> Map TagName [SomeArticle]
    collectTags post@BlogPost {bpTagNames} = Map.fromList $ (,pure $ SomeArticle post) <$> itemsToList bpTagNames
    collectTags publication@Publication {pubTagNames} = Map.fromList $ (,pure $ SomeArticle publication) <$> itemsToList pubTagNames
    mkTag (tagName@(TagName name), articles') =
      Tag
        { name = tagName,
          articles = sortOn (`withSomeArticle` comp) articles',
          url = "tags/" <> name <> ".html"
        }
    comp :: forall kind. Article kind -> Down UTCTime
    comp Publication {..} = Down . parseDate $ pubDate
    comp BlogPost {..} = Down . parseDate $ bpDate

-- | build tags page
buildTags :: Config Identity -> [Tag] -> Action ()
buildTags config tags = do
  tagsTemplate <- compileTemplate' "site/templates/tags.html"
  let tagsInfo = TagsInfo {tags}
      tagsHTML = T.unpack $ substitute tagsTemplate (withSiteMeta config $ A.toJSON tagsInfo)
  writeFile' (runIdentity (outputFolder config) </> "tags.html") tagsHTML

-- | build a single tag page
buildTag :: Config Identity -> Tag -> Action Tag
buildTag config tag@Tag {..} =
  do
    tagTemplate <- compileTemplate' "site/templates/tag.html"
    let tagData = withSiteMeta config $ A.toJSON tag
        tagHTML = T.unpack $ substitute tagTemplate tagData
    writeFile' (runIdentity (outputFolder config) </> url) tagHTML
    convert tagData

-- | calculate read time of a post based on the number of words
-- and the average reading speed of around 200 words per minute
calcReadTime :: String -> Integer
calcReadTime = fromIntegral . uncurry roundUp . flip divMod 200 . length . words
  where
    roundUp mins secs = mins + if secs == 0 then 0 else 1

-- | build contact page
buildContact :: Config Identity -> Action ()
buildContact config = cacheAction ("build" :: T.Text, contactSrcPath) $ do
  contactContent <- readFile' contactSrcPath
  contactData <- codeToHTML . T.pack $ contactContent
  contactTemplate <- compileTemplate' "site/templates/contact.html"
  gitHash <- getGitHash contactSrcPath >>= prettyGitHash config
  let withGitHash = A._Object . at "gitHash" ?~ A.String (T.pack gitHash)
      fullContactData = withSiteMeta config . withGitHash $ contactData
      contactHTML = T.unpack $ substitute contactTemplate fullContactData
  writeFile' (runIdentity (outputFolder config) </> "contact.html") contactHTML
  where
    contactSrcPath :: FilePath
    contactSrcPath = "site/contact.lhs"

-- | build resume page
buildResume :: Config Identity -> Action ()
buildResume config = cacheAction ("build" :: T.Text, resumeSrcPath) $ do
  resumeJson <- readFile' resumeSrcPath
  resumeTemplate <- compileTemplate' "site/templates/resume.html"
  gitHash <- getGitHash resumeSrcPath >>= prettyGitHash config
  let resumeData =
        fromMaybe "{}" $
          do
            json <- A.decode' . TL.encodeUtf8 $ TL.pack resumeJson
            go json
        where
          go :: A.Value -> Maybe A.Value
          go (A.Object a) = Just . A.Object $ HML.mapMaybe go a
          go (A.Array a) = case V.mapMaybe go a of
            a'
              | null a -> Nothing
              | otherwise -> Just . A.Object . HML.singleton "items" . A.Array $ a'
          go x = Just x
  let withGitHash = A._Object . at "gitHash" ?~ A.String (T.pack gitHash)
      fullResumeData = withSiteMeta config . withGitHash $ resumeData
      resumeHTML = T.unpack $ substitute resumeTemplate fullResumeData
  writeFile' (runIdentity (outputFolder config) </> "resume.html") resumeHTML
  where
    resumeSrcPath :: FilePath
    resumeSrcPath = "site/resume.json"

-- | copy all static files
copyStaticFiles :: Config Identity -> Action ()
copyStaticFiles config = do
  staticFilePaths <- getDirectoryFiles "." ["site/images//*", "site/css//*", "site/js//*", "site/fonts//*"]
  void $
    forP staticFilePaths $ \src -> do
      let dest = runIdentity (outputFolder config) </> dropDirectory1 src
      copyFileChanged src dest

-- | parse human-readable date from an article
parseDate :: String -> UTCTime
parseDate = parseTimeOrError True defaultTimeLocale "%b %e, %Y"

-- | format a date in ISO 8601 format
formatDate :: String -> String
formatDate = toIsoDate . parseDate

-- | RFC 3339 date format
rfc3339 :: Maybe String
rfc3339 = Just "%H:%M:SZ"

-- | convert UTC time to RFC 3339 format
toIsoDate :: UTCTime -> String
toIsoDate = formatTime defaultTimeLocale (iso8601DateFormat rfc3339)

-- | build feed
buildFeed :: Config Identity -> [SomeArticle] -> Action ()
buildFeed config articles = do
  now <- liftIO getCurrentTime
  let feedData =
        FeedData
          { title = runIdentity . siteTitle . siteMeta $ config,
            domain = show . runIdentity . siteBaseUrl . siteMeta $ config,
            author = runIdentity . siteAuthor . siteMeta $ config,
            articles = (`withSomeArticle` toFeedPost) <$> articles,
            currentTime = toIsoDate now,
            atomUrl = "/feed.xml"
          }
  feedTemplate <- compileTemplate' "site/templates/feed.xml"
  writeFile' (runIdentity (outputFolder config) </> "feed.xml") . T.unpack $ substitute feedTemplate (A.toJSON feedData)
  where
    toFeedPost :: forall kind. Article kind -> SomeArticle
    toFeedPost p@BlogPost {..} = SomeArticle $ p {bpDate = formatDate bpDate}
    toFeedPost p@Publication {..} = SomeArticle $ p {pubDate = formatDate pubDate}

-- | build site using all actions
buildRules :: Config Identity -> Action ()
buildRules config = do
  buildIndex config
  posts <- buildBlogPostList config
  buildBlogPosts config posts
  publications <- buildPublicationList config
  buildPublications config publications
  let articles = (SomeArticle <$> posts) <> (SomeArticle <$> publications)
  tags <- buildTagList config articles
  buildTags config tags
  buildContact config
  buildResume config
  copyStaticFiles config
  buildFeed config articles

-- | data type for git hash
data GitHash = GitHash {gitHash :: String, gitDate :: String, gitAuthor :: String, gitMessage :: String}
  deriving (Eq, Show)

-- | get git hash of last commit of a file
getGitHash :: FilePath -> Action GitHash
getGitHash path =
  let cmd format = readProcess "git" ["log", "--pretty=format:" <> format, "-n", "1", "--", path] ""
   in liftIO $ GitHash <$> cmd "%h" <*> cmd "%ci" <*> cmd "%an" <*> cmd "%s"

-- | pretty print git hash with link to GitHub commit
prettyGitHash :: Config Identity -> GitHash -> Action String
prettyGitHash config GitHash {..} = do
  Just uri <- pure $ do
    repo <- runIdentity . siteGithubRepository . siteMeta $ config
    return $ repo {uriPath = uriPath repo <> "/commit/" <> gitHash}
  let link = "<a href=\"" <> show uri <> "\">" <> gitHash <> "</a>"
  return $ "commit " <> link <> " (" <> gitDate <> ") " <> gitAuthor <> ": " <> gitMessage

-- | parser info for command line arguments
parserInfo ::
  forall b f.
  Barbie.TraversableB b =>
  b (Options.Parser `Compose` f) ->
  Options.ParserInfo (b f, Maybe FilePath)
parserInfo b =
  let parser =
        (,) <$> Barbie.bsequence b
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
