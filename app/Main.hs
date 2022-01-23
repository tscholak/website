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

import Control.Lens (at, ix, (?~), (^?))
import Control.Monad (void)
import Control.Monad.IO.Class (MonadIO (liftIO))
import Data.Aeson (toJSON)
import qualified Data.Aeson as A (FromJSON (parseJSON), KeyValue ((.=)), ToJSON (toJSON), Value (..), decode', object, withObject, (.:), (.:?))
import qualified Data.Aeson.Lens as A (AsPrimitive (_String), AsValue (_Object), key, pattern Integer)
import qualified Data.HashMap.Lazy as HML
import Data.List (sortOn)
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
import Development.Shake (Action, ShakeOptions (..), copyFileChanged, forP, getDirectoryFiles, readFile', shakeOptions, writeFile', pattern Chatty)
import Development.Shake.Classes (Binary (..))
import Development.Shake.FilePath (dropDirectory1, takeExtension, (-<.>), (</>))
import Development.Shake.Forward (cacheAction, shakeArgsForward)
import GHC.Generics (Generic)
import Network.URI (URI (uriPath), parseURI)
import Slick (compileTemplate', convert, substitute)
import Slick.Pandoc (defaultHtml5Options, markdownToHTMLWithOpts)
import System.Process (readProcess)
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
      githubUser = Just "tscholak",
      githubRepository = Just "https://github.com/tscholak/website"
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

-- | add site meta to a JSON object
withSiteMeta :: A.Value -> A.Value
withSiteMeta (A.Object obj) = A.Object $ HML.union obj siteMetaObj
  where
    A.Object siteMetaObj = A.toJSON siteMeta
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
    githubUser :: Maybe String,
    githubRepository :: Maybe String
  }
  deriving stock (Generic, Eq, Ord, Show)
  deriving anyclass (A.ToJSON)

newtype TagName = TagName String
  deriving stock (Generic, Eq, Ord, Show)

instance A.FromJSON TagName where
  parseJSON v = TagName <$> A.parseJSON v

instance A.ToJSON TagName where
  toJSON (TagName tagName) = A.toJSON tagName

instance Binary TagName where
  put (TagName tagName) = put tagName
  get = TagName <$> get

data ArticleKind = BlogPostKind | PublicationKind
  deriving stock (Eq, Ord, Show, Generic)

articleKind :: forall kind. Article kind -> ArticleKind
articleKind BlogPost {} = BlogPostKind
articleKind Publication {} = PublicationKind

newtype Items a = Items {items :: NonEmpty a}
  deriving stock (Eq, Ord, Show, Generic)
  deriving anyclass (A.ToJSON, A.FromJSON, Binary)

itemsToList :: Maybe (Items a) -> [a]
itemsToList = maybe [] (NonEmpty.toList . items)

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

assignAdjacentArticles :: forall kind. [Article kind] -> [Article kind]
assignAdjacentArticles posts =
  [ let prev = posts ^? ix (i + 1)
        next = posts ^? ix (i - 1)
        go Publication {} = cur {pubPrev = prev, pubNext = next}
        go BlogPost {} = cur {bpPrev = prev, bpNext = next}
     in go cur
    | (cur, i) <- zip posts [0 :: Int ..]
  ]

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
buildIndex :: Action ()
buildIndex = cacheAction ("build" :: T.Text, indexSrcPath) $ do
  indexContent <- readFile' indexSrcPath
  indexData <- markdownToHTML . T.pack $ indexContent
  indexTemplate <- compileTemplate' "site/templates/index.html"
  gitHash <- getGitHash indexSrcPath >>= prettyGitHash
  let withGitHash = A._Object . at "gitHash" ?~ A.String (T.pack gitHash)
      fullIndexData = withSiteMeta . withGitHash $ indexData
      indexHTML = T.unpack $ substitute indexTemplate fullIndexData
  writeFile' (outputFolder </> "index.html") indexHTML
  where
    indexSrcPath :: FilePath
    indexSrcPath = "site/home.md"

-- | find and build all blog posts
buildBlogPostList :: Action [Article 'BlogPostKind]
buildBlogPostList = do
  blogPostPaths <- getDirectoryFiles "." ["site/posts//*.md", "site/posts//*.lhs"]
  blogPosts <- forP blogPostPaths buildBlogPost
  let blogPosts' = assignAdjacentArticles . sortOn (Down . parseDate . bpDate) $ blogPosts
  _ <- forP blogPosts' writeBlogPost
  return blogPosts'

-- | build blog posts page
buildBlogPosts :: [Article 'BlogPostKind] -> Action ()
buildBlogPosts articles = do
  blogPostsTemplate <- compileTemplate' "site/templates/posts.html"
  let blogPostsInfo = ArticlesInfo {articles}
      blogPostsHTML = T.unpack $ substitute blogPostsTemplate (withSiteMeta $ A.toJSON blogPostsInfo)
  writeFile' (outputFolder </> "posts.html") blogPostsHTML

-- | build a single blog post
buildBlogPost :: FilePath -> Action (Article 'BlogPostKind)
buildBlogPost postSrcPath = cacheAction ("build" :: T.Text, postSrcPath) $ do
  postContent <- readFile' postSrcPath
  postData <- case takeExtension postSrcPath of
    ".md" -> markdownToHTML . T.pack $ postContent
    ".lhs" -> codeToHTML . T.pack $ postContent
    _ -> fail "Expected .md or .lhs"
  gitHash <- getGitHash postSrcPath >>= prettyGitHash
  let postUrl = T.pack . dropDirectory1 $ postSrcPath -<.> "html"
      withPostUrl = A._Object . at "url" ?~ A.String postUrl
      content = T.unpack $ fromMaybe "" $ postData ^? A.key "content" . A._String
      withReadTime = A._Object . at "readTime" ?~ A.Integer (calcReadTime content)
      withGitHash = A._Object . at "gitHash" ?~ A.String (T.pack gitHash)
      fullPostData = withSiteMeta . withReadTime . withGitHash . withPostUrl $ postData
  convert fullPostData

-- | write blog post to file
writeBlogPost :: Article 'BlogPostKind -> Action ()
writeBlogPost post@BlogPost {..} = do
  postTemplate <- compileTemplate' "site/templates/post.html"
  writeFile' (outputFolder </> bpUrl) . T.unpack . substitute postTemplate $ A.toJSON post

-- | find and build all publications
buildPublicationList :: Action [Article 'PublicationKind]
buildPublicationList = do
  publicationPaths <- getDirectoryFiles "." ["site/publications//*.md"]
  publications <- forP publicationPaths buildPublication
  let publications' = assignAdjacentArticles . sortOn (Down . parseDate . pubDate) $ publications
  _ <- forP publications' writePublication
  return publications'

-- | build publications page
buildPublications :: [Article 'PublicationKind] -> Action ()
buildPublications articles = do
  publicationsTemplate <- compileTemplate' "site/templates/publications.html"
  let publicationsInfo = ArticlesInfo {articles}
      publicationsHTML = T.unpack $ substitute publicationsTemplate (withSiteMeta $ A.toJSON publicationsInfo)
  writeFile' (outputFolder </> "publications.html") publicationsHTML

-- | build a single publication
buildPublication :: FilePath -> Action (Article 'PublicationKind)
buildPublication publicationSrcPath = cacheAction ("build" :: T.Text, publicationSrcPath) $ do
  publicationContent <- readFile' publicationSrcPath
  publicationData <- markdownToHTML . T.pack $ publicationContent
  gitHash <- getGitHash publicationSrcPath >>= prettyGitHash
  let publicationUrl = T.pack . dropDirectory1 $ publicationSrcPath -<.> "html"
      withPublicationUrl = A._Object . at "url" ?~ A.String publicationUrl
      withGitHash = A._Object . at "gitHash" ?~ A.String (T.pack gitHash)
      fullPublicationData = withSiteMeta . withPublicationUrl . withGitHash $ publicationData
  convert fullPublicationData

-- | write publication to file
writePublication :: Article 'PublicationKind -> Action ()
writePublication publication@Publication {..} = do
  publicationTemplate <- compileTemplate' "site/templates/publication.html"
  writeFile' (outputFolder </> pubUrl) . T.unpack . substitute publicationTemplate $ A.toJSON publication

-- | find all tags and build tag pages
buildTagList :: [SomeArticle] -> Action [Tag]
buildTagList articles =
  forP (Map.toList tags) (buildTag . mkTag)
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
buildTags :: [Tag] -> Action ()
buildTags tags = do
  tagsTemplate <- compileTemplate' "site/templates/tags.html"
  let tagsInfo = TagsInfo {tags}
      tagsHTML = T.unpack $ substitute tagsTemplate (withSiteMeta $ A.toJSON tagsInfo)
  writeFile' (outputFolder </> "tags.html") tagsHTML

-- | build a single tag page
buildTag :: Tag -> Action Tag
buildTag tag@Tag {..} =
  do
    tagTemplate <- compileTemplate' "site/templates/tag.html"
    let tagData = withSiteMeta $ A.toJSON tag
        tagHTML = T.unpack $ substitute tagTemplate tagData
    writeFile' (outputFolder </> url) tagHTML
    convert tagData

-- | calculate read time of a post based on the number of words
-- and the average reading speed of around 200 words per minute
calcReadTime :: String -> Integer
calcReadTime = fromIntegral . uncurry roundUp . flip divMod 200 . length . words
  where
    roundUp mins secs = mins + if secs == 0 then 0 else 1

-- | build contact page
buildContact :: Action ()
buildContact = cacheAction ("build" :: T.Text, contactSrcPath) $ do
  contactContent <- readFile' contactSrcPath
  contactData <- codeToHTML . T.pack $ contactContent
  contactTemplate <- compileTemplate' "site/templates/contact.html"
  gitHash <- getGitHash contactSrcPath >>= prettyGitHash
  let withGitHash = A._Object . at "gitHash" ?~ A.String (T.pack gitHash)
      fullContactData = withSiteMeta . withGitHash $ contactData
      contactHTML = T.unpack $ substitute contactTemplate fullContactData
  writeFile' (outputFolder </> "contact.html") contactHTML
  where
    contactSrcPath :: FilePath
    contactSrcPath = "site/contact.lhs"

-- | build resume page
buildResume :: Action ()
buildResume = cacheAction ("build" :: T.Text, resumeSrcPath) $ do
  resumeJson <- readFile' resumeSrcPath
  resumeTemplate <- compileTemplate' "site/templates/resume.html"
  gitHash <- getGitHash resumeSrcPath >>= prettyGitHash
  let resumeData = go $ fromMaybe "{}" . A.decode' . TL.encodeUtf8 $ TL.pack resumeJson
        where
          go :: A.Value -> A.Value
          go (A.Object a) = A.Object $ HML.map go a
          go (A.Array a)
            | null a = A.Null
            | otherwise = A.Object . HML.singleton "items" . A.Array $ go <$> a
          go x = x
  let withGitHash = A._Object . at "gitHash" ?~ A.String (T.pack gitHash)
      fullResumeData = withSiteMeta . withGitHash $ resumeData
      resumeHTML = T.unpack $ substitute resumeTemplate fullResumeData
  writeFile' (outputFolder </> "resume.html") resumeHTML
  where
    resumeSrcPath :: FilePath
    resumeSrcPath = "site/resume.json"

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
  writeFile' (outputFolder </> "feed.xml") . T.unpack $ substitute feedTemplate (A.toJSON feedData)
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
  buildContact
  buildResume
  copyStaticFiles
  buildFeed articles

data GitHash = GitHash {gitHash :: String, gitDate :: String, gitAuthor :: String, gitMessage :: String}
  deriving (Eq, Show)

-- | get git hash of last commit of a file
getGitHash :: FilePath -> Action GitHash
getGitHash path = do
  let cmd format = liftIO $ readProcess "git" ["log", "--pretty=format:" <> format, "-n", "1", "--", path] ""
  gitHash <- cmd "%h"
  gitDate <- cmd "%ad"
  gitAuthor <- cmd "%an"
  gitMessage <- cmd "%s"
  return GitHash {..}

-- | pretty print git hash with link to GitHub commit
prettyGitHash :: GitHash -> Action String
prettyGitHash GitHash {..} = do
  Just uri <- pure $ do
    repo <- githubRepository siteMeta
    repoUri <- parseURI repo
    return $ repoUri {uriPath = uriPath repoUri <> "/commit/" <> gitHash}
  let link = "<a href=\"" <> show uri <> "\">" <> gitHash <> "</a>"
  return $ "commit " <> link <> " (" <> gitDate <> ") " <> gitAuthor <> ": " <> gitMessage

-- | main program
main :: IO ()
main = do
  let options = shakeOptions {shakeVerbosity = Chatty, shakeLintInside = ["\\"]}
  shakeArgsForward options buildRules
