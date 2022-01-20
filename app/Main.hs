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
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE UndecidableInstances #-}

module Main where

import Control.Lens (at, makeLenses, (?~), (^?))
import Control.Monad (void)
import Control.Monad.IO.Class (MonadIO (liftIO))
import Data.Aeson as A (FromJSON (parseJSON), KeyValue ((.=)), ToJSON (toJSON), Value (Object, String), object, withObject, (.:), (.:?))
import Data.Aeson.Lens (AsPrimitive (_String), AsValue (_Object), key, pattern Integer)
import qualified Data.HashMap.Lazy as HML
import Data.Map.Lazy (Map)
import qualified Data.Map.Lazy as Map
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import qualified Data.Text as Text
import Data.Time (UTCTime, defaultTimeLocale, formatTime, getCurrentTime, iso8601DateFormat, parseTimeOrError)
import Debug.Trace (traceShowId)
import Development.Shake (Action, ShakeOptions (..), copyFileChanged, forP, getDirectoryFiles, readFile', shakeOptions, writeFile', pattern Chatty)
import Development.Shake.Classes (Binary (..))
import Development.Shake.FilePath (dropDirectory1, (-<.>), (</>))
import Development.Shake.Forward (cacheAction, shakeArgsForward)
import GHC.Generics (Generic)
import Slick (compileTemplate', convert, markdownToHTML, substitute)

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
    { _bpTitle :: String,
      _bpContent :: String,
      _bpURL :: String,
      _bpDate :: String,
      _bpTagNames :: [TagName],
      _bpTeaser :: String,
      _bpReadTime :: Int,
      _bpImage :: Maybe String,
      _bpPrev :: Maybe (Article 'BlogPostKind),
      _bpNext :: Maybe (Article 'BlogPostKind)
    } ->
    Article 'BlogPostKind
  Publication ::
    { _pubTitle :: String,
      _pubAuthor :: String,
      _pubJournal :: String,
      _pubContent :: String,
      _pubURL :: String,
      _pubDate :: String,
      _pubTagNames :: [TagName],
      _pubTldr :: String,
      _pubImage :: Maybe String,
      _pubLink :: Maybe String,
      _pubPDF :: Maybe String,
      _pubCode :: Maybe String,
      _pubTalk :: Maybe String,
      _pubPoster :: Maybe String,
      _pubPrev :: Maybe (Article 'PublicationKind),
      _pubNext :: Maybe (Article 'PublicationKind)
    } ->
    Article 'PublicationKind

$(makeLenses ''Article)

deriving stock instance Eq (Article kind)

deriving stock instance Ord (Article kind)

deriving stock instance Show (Article kind)

instance Binary (Article 'BlogPostKind) where
  put BlogPost {..} = put _bpTitle >> put _bpContent >> put _bpURL >> put _bpDate >> put _bpTagNames >> put _bpTeaser >> put _bpReadTime >> put _bpImage >> put _bpPrev >> put _bpNext
  get = BlogPost <$> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get

instance Binary (Article 'PublicationKind) where
  put Publication {..} = put _pubTitle >> put _pubAuthor >> put _pubJournal >> put _pubContent >> put _pubURL >> put _pubDate >> put _pubTagNames >> put _pubTldr >> put _pubImage >> put _pubLink >> put _pubPDF >> put _pubCode >> put _pubTalk >> put _pubPoster >> put _pubPrev >> put _pubNext
  get = Publication <$> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get <*> get

instance ToJSON (Article 'BlogPostKind) where
  toJSON BlogPost {..} =
    object
      [ "title" A..= _bpTitle,
        "content" A..= _bpContent,
        "url" A..= _bpURL,
        "date" A..= _bpDate,
        "tags" A..= _bpTagNames,
        "teaser" A..= _bpTeaser,
        "readTime" A..= _bpReadTime,
        "image" A..= _bpImage,
        "prev" A..= _bpPrev,
        "next" A..= _bpNext
      ]

instance ToJSON (Article 'PublicationKind) where
  toJSON Publication {..} =
    object
      [ "title" A..= _pubTitle,
        "author" A..= _pubAuthor,
        "journal" A..= _pubJournal,
        "content" A..= _pubContent,
        "url" A..= _pubURL,
        "date" A..= _pubDate,
        "tags" A..= _pubTagNames,
        "tldr" A..= _pubTldr,
        "image" A..= _pubImage,
        "link" A..= _pubLink,
        "pdf" A..= _pubPDF,
        "code" A..= _pubCode,
        "talk" A..= _pubTalk,
        "poster" A..= _pubPoster,
        "prev" A..= _pubPrev,
        "next" A..= _pubNext
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
        A..:? "link" <*> o
        A..:? "pdf" <*> o
        A..:? "code" <*> o
        A..:? "talk" <*> o
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
  let blogPostsInfo = ArticlesInfo {articles}
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
  let publicationsInfo = ArticlesInfo {articles}
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
buildTagList articles = do
  let tags :: Map TagName [SomeArticle] = Map.unionsWith (<>) ((`withSomeArticle` collectTags) <$> articles)
  forP (Map.toList tags) (buildTag . mkTag)
  where
    collectTags :: forall kind. Article kind -> Map TagName [SomeArticle]
    collectTags post@BlogPost {_bpTagNames} = Map.fromList $ (,pure $ SomeArticle post) <$> _bpTagNames
    collectTags publication@Publication {_pubTagNames} = Map.fromList $ (,pure $ SomeArticle publication) <$> _pubTagNames
    mkTag (tagName@(TagName name), articles') =
      Tag
        { name = tagName,
          articles = articles',
          url = "tags/" <> name <> ".html"
        }

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
buildAbout = do
  aboutTemplate <- compileTemplate' "site/templates/about.html"
  let aboutData = withSiteMeta $ toJSON siteMeta
      aboutHTML = Text.unpack $ substitute aboutTemplate aboutData
  writeFile' (outputFolder </> "about.html") aboutHTML

-- | copy all static files
copyStaticFiles :: Action ()
copyStaticFiles = do
  staticFilePaths <- getDirectoryFiles "." ["site/images//*", "site/css//*", "site/js//*", "site/fonts//*"]
  void $
    forP staticFilePaths $ \src -> do
      let dest = outputFolder </> dropDirectory1 src
      copyFileChanged src dest

-- | format a date in ISO 8601 format
formatDate :: String -> String
formatDate humanDate = toIsoDate parsedTime
  where
    parsedTime = parseTimeOrError True defaultTimeLocale "%b %e, %Y" humanDate

rfc3339 :: Maybe String
rfc3339 = Just "%H:%M:SZ"

toIsoDate :: UTCTime -> String
toIsoDate = formatTime defaultTimeLocale (iso8601DateFormat rfc3339)

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
    toFeedPost p@BlogPost {..} = SomeArticle $ p {_bpDate = formatDate _bpDate}
    toFeedPost p@Publication {..} = SomeArticle $ p {_pubDate = formatDate _pubDate}

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

main :: IO ()
main = do
  let options = shakeOptions {shakeVerbosity = Chatty, shakeLintInside = ["\\"]}
  shakeArgsForward options buildRules
