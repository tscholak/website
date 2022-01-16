{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}

module Main where

import Control.Lens (at, (?~), (^.), (^?))
import Control.Monad (void)
import Control.Monad.IO.Class (MonadIO (liftIO))
import Data.Aeson as A (FromJSON (parseJSON), KeyValue ((.=)), ToJSON (toJSON), Value (Object, String), object, withObject, (.:), (.:?))
import Data.Aeson.Lens (AsPrimitive (_String), AsValue (_Object), key, pattern Integer)
import qualified Data.HashMap.Lazy as HML
import Data.Map.Lazy (Map)
import qualified Data.Map.Lazy as Map
import Data.Maybe (fromMaybe)
import Data.Set (Set)
import Data.Text (Text)
import qualified Data.Text as Text
import Data.Time (UTCTime, defaultTimeLocale, formatTime, getCurrentTime, iso8601DateFormat, parseTimeOrError)
import Development.Shake (Action, ShakeOptions (..), copyFileChanged, forP, getDirectoryFiles, readFile', shakeOptions, writeFile', pattern Chatty)
import Development.Shake.Classes (Binary (..))
import Development.Shake.FilePath (dropDirectory1, (-<.>), (</>))
import Development.Shake.Forward (cacheAction, shakeArgsForward)
import GHC.Generics (Generic)
import Slick (compileTemplate', convert, markdownToHTML, substitute)

-- | site meta data
siteMeta :: SiteMeta
siteMeta =
  SiteMeta
    { siteAuthor = "Torsten Scholak",
      baseUrl = "https://tscholak.github.io",
      siteTitle = "Torsten Scholak",
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
  { siteAuthor :: String,
    baseUrl :: String, -- e.g. https://example.ca
    siteTitle :: String,
    twitterHandle :: Maybe String, -- Without @
    twitchHandle :: Maybe String,
    youtubeHandle :: Maybe String,
    githubUser :: Maybe String
  }
  deriving stock (Generic, Eq, Ord, Show)
  deriving anyclass (ToJSON)

newtype PostsInfo = PostsInfo
  { posts :: [Post]
  }
  deriving stock (Generic, Eq, Ord, Show)
  deriving anyclass (FromJSON, ToJSON)

newtype TagName = TagName String
  deriving stock (Generic, Eq, Ord, Show)

instance FromJSON TagName where
  parseJSON v = TagName <$> parseJSON v

instance ToJSON TagName where
  toJSON (TagName tagName) = toJSON tagName

instance Binary TagName where
  put (TagName tagName) = put tagName
  get = TagName <$> get

data Tag = Tag
  { name :: TagName,
    posts :: [Post],
    url :: String
  }
  deriving stock (Generic, Eq, Ord, Show)

instance ToJSON Tag where
  toJSON Tag {..} =
    object
      [ "tag" A..= name,
        "posts" A..= posts,
        "url" A..= url
      ]

instance FromJSON Tag where
  parseJSON =
    withObject "Tag" $ \o ->
      Tag <$> o
        A..: "tag" <*> o
        A..: "posts" <*> o
        A..: "url"

newtype TagsInfo = TagsInfo
  { tags :: [Tag]
  }
  deriving stock (Generic, Eq, Ord, Show)
  deriving anyclass (FromJSON, ToJSON)

data Post = Post
  { title :: String,
    content :: String,
    url :: String,
    date :: String,
    tagNames :: [TagName],
    teaser :: String,
    readTime :: Int,
    image :: Maybe String,
    prev :: Maybe Post,
    next :: Maybe Post
  }
  deriving stock (Generic, Eq, Ord, Show)
  deriving anyclass (Binary)

instance ToJSON Post where
  toJSON Post {..} =
    object
      [ "title" A..= title,
        "content" A..= content,
        "url" A..= url,
        "date" A..= date,
        "tags" A..= tagNames,
        "teaser" A..= teaser,
        "readTime" A..= readTime,
        "image" A..= image,
        "prev" A..= prev,
        "next" A..= next
      ]

instance FromJSON Post where
  parseJSON =
    withObject "Post" $ \o ->
      Post <$> o
        A..: "title" <*> o
        A..: "content" <*> o
        A..: "url" <*> o
        A..: "date" <*> o
        A..: "tags" <*> o
        A..: "teaser" <*> o
        A..: "readTime" <*> o
        A..:? "image" <*> o
        A..:? "prev" <*> o
        A..:? "next"

data FeedData = FeedData
  { title :: String,
    domain :: String,
    author :: String,
    posts :: [Post],
    currentTime :: String,
    atomUrl :: String
  }
  deriving stock (Generic, Eq, Ord, Show)
  deriving anyclass (ToJSON)

-- | build landing page
buildIndex :: Action ()
buildIndex = do
  indexTemplate <- compileTemplate' "site/templates/index.html"
  let indexData = withSiteMeta $ toJSON siteMeta
      indexHTML = Text.unpack $ substitute indexTemplate indexData
  writeFile' (outputFolder </> "index.html") indexHTML

-- | find and build all posts
buildPostList :: Action [Post]
buildPostList = do
  postPaths <- getDirectoryFiles "." ["site/posts//*.md"]
  forP postPaths buildPost

-- | build posts page
buildPosts :: [Post] -> Action ()
buildPosts posts = do
  postsTemplate <- compileTemplate' "site/templates/posts.html"
  let postsInfo = PostsInfo {posts}
      postsHTML = Text.unpack $ substitute postsTemplate (withSiteMeta $ toJSON postsInfo)
  writeFile' (outputFolder </> "posts.html") postsHTML

-- | build a single post
buildPost :: FilePath -> Action Post
buildPost postSrcPath = cacheAction ("build" :: Text, postSrcPath) $ do
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

-- | find all tags and build tag pages
buildTagList :: [Post] -> Action [Tag]
buildTagList posts = do
  let tags :: Map TagName [Post] = Map.unionsWith (<>) (collectTags <$> posts)
  forP (Map.toList tags) (buildTag . mkTag)
  where
    collectTags post@Post {tagNames} = Map.fromList $ (,pure post) <$> tagNames
    mkTag (tagName@(TagName name), posts') =
      Tag
        { name = tagName,
          posts = posts',
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
  staticFilePaths <- getDirectoryFiles "." ["site/images//*", "site/css//*", "site/js//*"]
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

buildFeed :: [Post] -> Action ()
buildFeed posts = do
  now <- liftIO getCurrentTime
  let feedData =
        FeedData
          { title = siteTitle siteMeta,
            domain = baseUrl siteMeta,
            author = siteAuthor siteMeta,
            posts = toFeedPost <$> posts,
            currentTime = toIsoDate now,
            atomUrl = "/feed.xml"
          }
  feedTemplate <- compileTemplate' "site/templates/feed.xml"
  writeFile' (outputFolder </> "feed.xml") . Text.unpack $ substitute feedTemplate (toJSON feedData)
  where
    toFeedPost :: Post -> Post
    toFeedPost p = p {date = formatDate $ date p}

buildRules :: Action ()
buildRules = do
  buildIndex
  posts <- buildPostList
  buildPosts posts
  tags <- buildTagList posts
  buildTags tags
  buildAbout
  copyStaticFiles
  buildFeed posts

main :: IO ()
main = do
  let options = shakeOptions {shakeVerbosity = Chatty, shakeLintInside = ["\\"]}
  shakeArgsForward options buildRules
