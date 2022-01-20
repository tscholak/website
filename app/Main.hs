{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE UndecidableInstances #-}

module Main where

import Control.Lens (at, (?~), (^?))
import Control.Monad (void)
import Control.Monad.IO.Class (MonadIO (liftIO))
import Data.Aeson as A (FromJSON (parseJSON), KeyValue ((.=)), ToJSON (toJSON), Value (Object, String), object, withObject, (.:), (.:?))
import Data.Aeson.Lens (AsPrimitive (_String), AsValue (_Object), key, pattern Integer)
import qualified Data.HashMap.Lazy as HML
import Data.List (sortOn)
import Data.Map.Lazy (Map)
import qualified Data.Map.Lazy as Map
import Data.Maybe (fromMaybe)
import Data.Ord (Down (Down))
import Data.Text (Text)
import qualified Data.Text as Text
import Data.Time (UTCTime, defaultTimeLocale, formatTime, getCurrentTime, iso8601DateFormat, parseTimeOrError)
import Development.Shake (Action, ShakeOptions (..), copyFileChanged, forP, getDirectoryFiles, readFile', shakeOptions, writeFile', pattern Chatty)
import Development.Shake.Classes (Binary (..))
import Development.Shake.FilePath (dropDirectory1, (-<.>), (</>))
import Development.Shake.Forward (cacheAction, shakeArgsForward)
import GHC.Generics (Generic)
import Slick (compileTemplate', convert, substitute)
import Slick.Pandoc (defaultHtml5Options, markdownToHTMLWithOpts)
import qualified Text.Pandoc as P

-- | site meta data
-- TODO: move to config file
siteMeta :: SiteMeta
siteMeta =
  SiteMeta
    { baseUrl = "https://tscholak.github.io",
      siteTitle = "Torsten Scholak",
      siteAuthor = "Torsten Scholak",
      siteDescription = "Torsten Scholak's personal website",
      siteKeywords = "Haskell, functional programming",
      twitterHandle = Just "tscholak",
      twitchHandle = Just "tscholak",
      youtubeHandle = Just "tscholak",
      githubUser = Just "tscholak"
    }

-- | the output directory
outputFolder :: FilePath
outputFolder = "build/"

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
markdownToHTML :: Text -> Action Value
markdownToHTML = markdownToHTMLWithOpts markdownOptions defaultHtml5Options

-- | convert literal Haskell code to html
codeToHTML :: Text -> Action Value
codeToHTML = markdownToHTMLWithOpts opts defaultHtml5Options
  where
    opts = P.def {P.readerExtensions = pandocExtensions}
    pandocExtensions = P.extensionsFromList [P.Ext_literate_haskell]

-- | add site meta to a JSON object
withSiteMeta :: Value -> Value
withSiteMeta (Object obj) = Object $ HML.union obj siteMetaObj
  where
    Object siteMetaObj = toJSON siteMeta
withSiteMeta v = error $ "only add site meta to objects, not " ++ show v

data SiteMeta = SiteMeta
  { baseUrl :: String, -- e.g. https://example.ca
    siteTitle :: String,
    siteAuthor :: String,
    siteDescription :: String,
    siteKeywords :: String,
    twitterHandle :: Maybe String, -- Without @
    twitchHandle :: Maybe String,
    youtubeHandle :: Maybe String,
    githubUser :: Maybe String
  }
  deriving stock (Generic, Eq, Ord, Show)
  deriving anyclass (ToJSON)

newtype TagName = TagName String
  deriving stock (Generic, Eq, Ord, Show)

instance FromJSON TagName where
  parseJSON v = TagName <$> parseJSON v

instance ToJSON TagName where
  toJSON (TagName tagName) = toJSON tagName

instance Binary TagName where
  put (TagName tagName) = put tagName
  get = TagName <$> get

data ArticleKind = BlogPostKind | PublicationKind
  deriving stock (Eq, Ord, Show, Generic)

articleKind :: forall kind. Article kind -> ArticleKind
articleKind BlogPost {} = BlogPostKind
articleKind Publication {} = PublicationKind

data Article kind where
  BlogPost ::
    { bpTitle :: String,
      bpContent :: String,
      bpURL :: String,
      bpDate :: String,
      bpTagNames :: [TagName],
      bpTeaser :: String,
      bpReadTime :: Int,
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
      pubURL :: String,
      pubDate :: String,
      pubTagNames :: [TagName],
      pubTldr :: String,
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
  put BlogPost {..} = put bpTitle >> put bpContent >> put bpURL >> put bpDate >> put bpTagNames >> put bpTeaser >> put bpReadTime >> put bpImage >> put bpPrev >> put bpNext
  get = BlogPost <$> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get

instance Binary (Article 'PublicationKind) where
  put Publication {..} = put pubTitle >> put pubAuthor >> put pubJournal >> put pubContent >> put pubURL >> put pubDate >> put pubTagNames >> put pubTldr >> put pubImage >> put pubLink >> put pubPDF >> put pubCode >> put pubTalk >> put pubSlides >> put pubPoster >> put pubPrev >> put pubNext
  get = Publication <$> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get

instance ToJSON (Article 'BlogPostKind) where
  toJSON BlogPost {..} =
    object
      [ "title" A..= bpTitle,
        "content" A..= bpContent,
        "url" A..= bpURL,
        "date" A..= bpDate,
        "tags" A..= bpTagNames,
        "teaser" A..= bpTeaser,
        "readTime" A..= bpReadTime,
        "image" A..= bpImage,
        "prev" A..= bpPrev,
        "next" A..= bpNext
      ]

instance ToJSON (Article 'PublicationKind) where
  toJSON Publication {..} =
    object
      [ "title" A..= pubTitle,
        "author" A..= pubAuthor,
        "journal" A..= pubJournal,
        "content" A..= pubContent,
        "url" A..= pubURL,
        "date" A..= pubDate,
        "tags" A..= pubTagNames,
        "tldr" A..= pubTldr,
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

instance FromJSON (Article 'BlogPostKind) where
  parseJSON =
    withObject "Blog post" $ \o ->
      BlogPost
        <$> o A..: "title"
        <*> o A..: "content"
        <*> o A..: "url"
        <*> o A..: "date"
        <*> o A..: "tags"
        <*> o A..: "teaser"
        <*> o A..: "readTime"
        <*> o A..:? "image"
        <*> o A..:? "prev"
        <*> o A..:? "next"

instance FromJSON (Article 'PublicationKind) where
  parseJSON =
    withObject "Publication" $ \o ->
      Publication <$> o
        A..: "title" <*> o
        A..: "author" <*> o
        A..: "journal" <*> o
        A..: "content" <*> o
        A..: "url" <*> o
        A..: "date" <*> o
        A..: "tags" <*> o
        A..: "tldr" <*> o
        A..:? "image" <*> o
        A..: "link" <*> o
        A..:? "pdf" <*> o
        A..:? "code" <*> o
        A..:? "talk" <*> o
        A..:? "slides" <*> o
        A..:? "poster" <*> o
        A..:? "prev" <*> o
        A..:? "next"

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

instance FromJSON SomeArticle where
  parseJSON =
    withObject "some article" $ \o -> do
      kind :: String <- o A..: "kind"
      case kind of
        "blog post" -> SomeArticle <$> (A..:) @(Article 'BlogPostKind) o "article"
        "publication" -> SomeArticle <$> (A..:) @(Article 'PublicationKind) o "article"
        _ -> fail "Expected blog post or publication"

instance ToJSON SomeArticle where
  toJSON (SomeArticle article) = case article of
    BlogPost {} -> object ["kind" A..= ("blog post" :: String), "article" A..= article]
    Publication {} -> object ["kind" A..= ("publication" :: String), "article" A..= article]

newtype ArticlesInfo kind = ArticlesInfo
  { articles :: [Article kind]
  }
  deriving stock (Generic, Eq, Ord, Show)

instance ToJSON (Article kind) => ToJSON (ArticlesInfo kind) where
  toJSON ArticlesInfo {..} = object ["articles" A..= articles]

instance FromJSON (Article kind) => FromJSON (ArticlesInfo kind) where
  parseJSON =
    withObject "ArticlesInfo" $ \o ->
      ArticlesInfo <$> o A..: "articles"

data Tag = Tag
  { name :: TagName,
    articles :: [SomeArticle],
    url :: String
  }
  deriving stock (Generic, Eq, Ord, Show)

instance ToJSON Tag where
  toJSON Tag {..} =
    object
      [ "tag" A..= name,
        "articles" A..= articles,
        "url" A..= url
      ]

instance FromJSON Tag where
  parseJSON =
    withObject "Tag" $ \o ->
      Tag <$> o
        A..: "tag" <*> o
        A..: "articles" <*> o
        A..: "url"

newtype TagsInfo = TagsInfo
  { tags :: [Tag]
  }
  deriving stock (Generic, Eq, Ord, Show)
  deriving anyclass (ToJSON)

data FeedData = FeedData
  { title :: String,
    domain :: String,
    author :: String,
    articles :: [SomeArticle],
    currentTime :: String,
    atomUrl :: String
  }
  deriving stock (Generic, Eq, Ord, Show)
  deriving anyclass (ToJSON)

-- | build landing page
buildIndex :: Action ()
buildIndex = cacheAction ("build" :: Text, indexSrcPath) $ do
  indexContent <- readFile' indexSrcPath
  indexData <- markdownToHTML . Text.pack $ indexContent
  indexTemplate <- compileTemplate' "site/templates/index.html"
  let fullIndexData = withSiteMeta indexData
      indexHTML = Text.unpack $ substitute indexTemplate fullIndexData
  writeFile' (outputFolder </> "index.html") indexHTML
  where
    indexSrcPath :: FilePath
    indexSrcPath = "site/home.md"

-- | find and build all blog posts
buildBlogPostList :: Action [Article 'BlogPostKind]
buildBlogPostList = do
  blogPostPaths <- getDirectoryFiles "." ["site/posts//*.md"]
  forP blogPostPaths buildBlogPost

-- | build blog posts page
buildBlogPosts :: [Article 'BlogPostKind] -> Action ()
buildBlogPosts articles = do
  blogPostsTemplate <- compileTemplate' "site/templates/posts.html"
  let sortedArticles = sortOn (Down . parseDate . bpDate) articles
      blogPostsInfo = ArticlesInfo {articles = sortedArticles}
      blogPostsHTML = Text.unpack $ substitute blogPostsTemplate (withSiteMeta $ toJSON blogPostsInfo)
  writeFile' (outputFolder </> "posts.html") blogPostsHTML

-- | build a single blog post
buildBlogPost :: FilePath -> Action (Article 'BlogPostKind)
buildBlogPost postSrcPath = cacheAction ("build" :: Text, postSrcPath) $ do
  postContent <- readFile' postSrcPath
  postData <- markdownToHTML . Text.pack $ postContent
  let postUrl = Text.pack . dropDirectory1 $ postSrcPath -<.> "html"
      withPostUrl = _Object . at "url" ?~ String postUrl
      content = Text.unpack $ fromMaybe "" $ postData ^? key "content" . _String
      withReadTime = _Object . at "readTime" ?~ Integer (calcReadTime content)
      fullPostData = withSiteMeta . withReadTime . withPostUrl $ postData
  postTemplate <- compileTemplate' "site/templates/post.html"
  writeFile' (outputFolder </> Text.unpack postUrl) . Text.unpack $ substitute postTemplate fullPostData
  convert fullPostData

-- | find and build all publications
buildPublicationList :: Action [Article 'PublicationKind]
buildPublicationList = do
  publicationPaths <- getDirectoryFiles "." ["site/publications//*.md"]
  forP publicationPaths buildPublication

-- | build publications page
buildPublications :: [Article 'PublicationKind] -> Action ()
buildPublications articles = do
  publicationsTemplate <- compileTemplate' "site/templates/publications.html"
  let sortedArticles = sortOn (Down . parseDate . pubDate) articles
      publicationsInfo = ArticlesInfo {articles = sortedArticles}
      publicationsHTML = Text.unpack $ substitute publicationsTemplate (withSiteMeta $ toJSON publicationsInfo)
  writeFile' (outputFolder </> "publications.html") publicationsHTML

-- | build a single publication
buildPublication :: FilePath -> Action (Article 'PublicationKind)
buildPublication publicationSrcPath = cacheAction ("build" :: Text, publicationSrcPath) $ do
  publicationContent <- readFile' publicationSrcPath
  publicationData <- markdownToHTML . Text.pack $ publicationContent
  let publicationUrl = Text.pack . dropDirectory1 $ publicationSrcPath -<.> "html"
      withPublicationUrl = _Object . at "url" ?~ String publicationUrl
      fullPublicationData = withSiteMeta . withPublicationUrl $ publicationData
  publicationTemplate <- compileTemplate' "site/templates/publication.html"
  writeFile' (outputFolder </> Text.unpack publicationUrl) . Text.unpack $ substitute publicationTemplate fullPublicationData
  convert fullPublicationData

-- | find all tags and build tag pages
buildTagList :: [SomeArticle] -> Action [Tag]
buildTagList articles =
  forP (Map.toList tags) (buildTag . mkTag)
  where
    tags = Map.unionsWith (<>) ((`withSomeArticle` collectTags) <$> articles)
    collectTags :: forall kind. Article kind -> Map TagName [SomeArticle]
    collectTags post@BlogPost {bpTagNames} = Map.fromList $ (,pure $ SomeArticle post) <$> bpTagNames
    collectTags publication@Publication {pubTagNames} = Map.fromList $ (,pure $ SomeArticle publication) <$> pubTagNames
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
buildTags :: [Tag] -> Action ()
buildTags tags = do
  tagsTemplate <- compileTemplate' "site/templates/tags.html"
  let tagsInfo = TagsInfo {tags}
      tagsHTML = Text.unpack $ substitute tagsTemplate (withSiteMeta $ toJSON tagsInfo)
  writeFile' (outputFolder </> "tags.html") tagsHTML

-- | build a single tag page
buildTag :: Tag -> Action Tag
buildTag tag@Tag {..} =
  do
    tagTemplate <- compileTemplate' "site/templates/tag.html"
    let tagData = withSiteMeta $ toJSON tag
        tagHTML = Text.unpack $ substitute tagTemplate tagData
    writeFile' (outputFolder </> url) tagHTML
    convert tagData

-- | calculate read time of a post based on the number of words
-- and the average reading speed of around 200 words per minute
calcReadTime :: String -> Integer
calcReadTime = fromIntegral . uncurry roundUp . flip divMod 200 . length . words
  where
    roundUp mins secs = mins + if secs == 0 then 0 else 1

-- | build about page
buildAbout :: Action ()
buildAbout = cacheAction ("build" :: Text, aboutSrcPath) $ do
  aboutContent <- readFile' aboutSrcPath
  aboutData <- codeToHTML . Text.pack $ aboutContent
  aboutTemplate <- compileTemplate' "site/templates/about.html"
  let fullAboutData = withSiteMeta aboutData
      aboutHTML = Text.unpack $ substitute aboutTemplate fullAboutData
  writeFile' (outputFolder </> "about.html") aboutHTML
  where
    aboutSrcPath :: FilePath
    aboutSrcPath = "site/about.lhs"

-- | copy all static files
copyStaticFiles :: Action ()
copyStaticFiles = do
  staticFilePaths <- getDirectoryFiles "." ["site/images//*", "site/css//*", "site/js//*", "site/fonts//*"]
  void $
    forP staticFilePaths $ \src -> do
      let dest = outputFolder </> dropDirectory1 src
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
buildFeed :: [SomeArticle] -> Action ()
buildFeed articles = do
  now <- liftIO getCurrentTime
  let feedData =
        FeedData
          { title = siteTitle siteMeta,
            domain = baseUrl siteMeta,
            author = siteAuthor siteMeta,
            articles = (`withSomeArticle` toFeedPost) <$> articles,
            currentTime = toIsoDate now,
            atomUrl = "/feed.xml"
          }
  feedTemplate <- compileTemplate' "site/templates/feed.xml"
  writeFile' (outputFolder </> "feed.xml") . Text.unpack $ substitute feedTemplate (toJSON feedData)
  where
    toFeedPost :: forall kind. Article kind -> SomeArticle
    toFeedPost p@BlogPost {..} = SomeArticle $ p {bpDate = formatDate bpDate}
    toFeedPost p@Publication {..} = SomeArticle $ p {pubDate = formatDate pubDate}

-- | build site using all actions
buildRules :: Action ()
buildRules = do
  buildIndex
  posts <- buildBlogPostList
  buildBlogPosts posts
  publications <- buildPublicationList
  buildPublications publications
  let articles = (SomeArticle <$> posts) <> (SomeArticle <$> publications)
  tags <- buildTagList articles
  buildTags tags
  buildAbout
  copyStaticFiles
  buildFeed articles

-- | main program
main :: IO ()
main = do
  let options = shakeOptions {shakeVerbosity = Chatty, shakeLintInside = ["\\"]}
  shakeArgsForward options buildRules
