---
title: "Speed Creates Capability: Why Efficient Attention Matters for AI Strategy"
publication:
  status: published
  date: Nov 18, 2025
teaser: >
  If your models think slowly, your roadmap does too. This essay argues that
  efficient attention is the hidden control knob for margins, RL capacity, and
  agent reliability, and shows how Apriel-H1 makes chain-of-thought, test-time
  compute, and large-scale RL practical at production scale.
tags:
  items: [ai, strategy, inference-efficiency, hybrid-architectures]
image: speed-creates-capability.svg
---

In my previous post, [Stop Renting Your Moat](/posts/stop-renting-moat.html), I argued that relying on foundation model APIs creates a strategic dependency that limits your ability to build differentiated products. Owning your learning loop (i.e., your data, your training infrastructure, and your model development) is the only thing that secures you control over capabilities that matter to your business.

This is great in theory, but what does it mean in practice? Today I want to dig into one specific aspect: inference speed. Inference speed is often treated as an operational concern: reduce costs, improve latency, optimize user experience. But it should instead be a core part of your AI strategy, because it determines which capabilities you can actually build and deploy.

The bottleneck has shifted from a year ago. Back then, the primary constraint for building reliable AI systems was model quality and how to make them reason well. Think of [DeepSeek-R1](https://arxiv.org/abs/2501.12948). Nowadays we have strong reasoning, including [open-weight models](https://moonshotai.github.io/Kimi-K2/thinking.html), and the new constraint has become making them also fast enough to be useful. Reasoning models are getting slower as they get better, because better reasoning requires longer chains of thought. Every additional reasoning step adds latency, which is why the models achieving the best results on hard problems are also the least practical to deploy.

It turns out that this is an architectural concern and ultimately comes down to attention mechanisms. If your reasoning model uses full attention, then every token attends to every other token in the context. This gives great fidelity but scales quadratically with context length. Doubling context length quadruples compute cost. For reasoning models generating 64k-token internal deliberations, such quadratic scaling makes long contexts prohibitively expensive to run in practice. That quadratic cost becomes the hard ceiling on what you can build: You can't run longer chains because latency becomes unacceptable, you can't deploy agents that keep full context because memory consumption explodes, and you can't scale test-time compute because generating multiple candidates takes too long.

Breaking that quadratic scaling therefore doesn't just reduce costs but creates capabilities that are qualitatively different from what full attention can deliver at acceptable latency. A model that runs at twice the throughput can explore twice as many solution paths in the same time budget, maintain twice as much context without compression, or serve twice as many agents in parallel. An agent that keeps full conversation history across 50 tool calls behaves fundamentally differently from one that must compress and forget after every 10 interactions.

At the beginning of the year, hybrid architectures mixing efficient and full attention were research curiosities. Today they're shipping in production at scale. In the last six months alone: NVIDIA's [Nemotron-H-47B](https://arxiv.org/abs/2504.03624) (9:1 Mamba:attention hybrid, 3x faster), [Falcon-H1-34B](https://arxiv.org/abs/2507.22448) (1:1 intra-layer hybrid, 4x faster prefill, 8x faster decode), [MiniMax's M1](https://arxiv.org/abs/2506.13585) (7:1 Lightning:attention hybrid, 3-4x faster at 100K tokens), NVIDIA's [Nemotron-Nano-9B-v2](https://arxiv.org/abs/2508.14444) (7:1 hybrid, up to 6x faster throughput), [Qwen3-Next-80B-A3B](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) (3:1 hybrid, >10x faster throughput at >32K context), [DeepSeek V3.2-Exp](https://api-docs.deepseek.com/news/news250929) (sparse attention, 3x faster), and [Kimi-Linear-48B](https://arxiv.org/abs/2510.26692) (3:1 hybrid, up to 6x faster at 1M tokens).

What's striking is how consistent the pattern is across different organizations, architectures, and design choices. It always comes down to mixing efficient attention mechanisms (sparse or linear) with some full attention layers to preserve global context. The exact ratio varies (3:1, 7:1, 9:1 efficient-to-full) but the principle is the same. In the end we see similar throughput gains from 2x to 10x depending on context length.

This means that the industry has identified quadratic attention cost as the bottleneck preventing deployment of capabilities teams want to ship, and efficient hybrids are the response. What makes this strategic rather than just an efficiency win is that these architectures break the fundamental tradeoffs that have constrained what you can build.

Breaking the Fast-Cheap-Good Triangle
-------------------------------------

The conventional wisdom about inference optimization follows the classic engineering tradeoff: you can have fast, cheap, or good. Pick two. Want high-quality reasoning with long chains of thought? That's good but slow and expensive. Want fast responses? Run shorter chains, sacrifice quality. Want cheap serving costs? Accept higher latency. This tradeoff has governed production deployment decisions for the past two years.

Efficient architectures break this triangle by making inference fundamentally cheaper in compute terms, which cascades through every dimension simultaneously.

Start with the first-order effects: doubling throughput means you can serve the same workload at half the cost, or twice the workload on the same hardware. For API providers, this difference determines gross margin, because it changes which use cases are economically viable to support. Enterprises running millions of daily requests see that cost difference compound into seven-figure annual savings.

But the cheaper serving cost enables something more valuable: you can now afford to run chain-of-thought reasoning by default, not as an expensive special mode you activate selectively. When inference is 3-5x cheaper, you make test-time compute the default serving path rather than the exception. Every request gets multi-step reasoning, verification, reflection. The quality improvement from running CoT inference universally can be larger than most post-training gains.

The UX implications are direct and measurable. Coding agents that execute tests, query documentation, and validate solutions need to complete complex workflows in seconds, not tens of seconds. A 20-second agent breaks user flow because they context-switch to other work and lose the thread. A 5-second agent stays within the bounds of acceptable wait time. That 15-second difference determines whether users actually integrate the agent into their workflow or find reasons to avoid it. For production agents handling complex multi-step tasks (like database queries, API calls, file operations, or error recovery) being able to maintain full context across 50 tool calls instead of compressing aggressively after 10 changes reliability from "mostly works" to "actually dependable." These are differences that determine product success.

The second-order effect is where efficient architectures create lasting competitive advantage. Cheaper inference means you can afford substantially more reinforcement learning during post-training. RL requires generating millions of reasoning traces, scoring them, updating the model, repeating. With 3-5x faster sampling, you run 3-5 times as many rollouts on the same budget. That means dramatically denser feedback signals, far more thorough exploration of solution space, better model behaviors discovered during training. This compounds. Better exploration finds better policies, which improve model quality, which generates better training data for the next RL iteration.

I think this is the real hundred-million-dollar lever. Companies like Qwen, DeepSeek, and Kimi are investing heavily in efficient architectures alongside RL-based post-training because the architecture choice determines how much RL you can afford to run. More RL means better models. Better models mean better user experience, higher win rates against competitors, and ultimately stickier products. Efficient model architectures multiply your post-training capacity on the same budget, which directly improves model quality.

There is also a third-order effect. The above advantages compound over time through a reinforcing cycle. Efficient architecture enables more RL training, which produces better models. Better models drive higher user adoption and usage. Higher usage would normally explode serving costs, but efficient architecture keeps costs manageable even as scale increases. That cost headroom funds another round of intensive RL training for the next model iteration. Competitors running less efficient architectures can't match this cycle, because they either can't afford the RL at scale, or their serving costs grow unsustainably as usage increases, forcing them to throttle features or raise prices. In this way, the architecture choice determines whether you can sustain the compounding advantage or whether each success creates new constraints.

Proving It Works for Reasoning
------------------------------

My team at ServiceNow built [Apriel-H1](https://arxiv.org/abs/2511.02651) to settle whether you can retrofit efficiency into an existing reasoning model without breaking it. I wrote about the journey [here](https://huggingface.co/blog/ServiceNow-AI/apriel-h1). We started with Apriel-15B, our full-attention reasoning model, and replaced 30 of 50 attention layers with Mamba blocks. We hit 2.1x throughput with quality holding—math and coding benchmarks stayed flat, conversational reasoning improved. We did this with 76.8 billion tokens of distillation and SFT. Retrofitting works.

Distillation de-risks when you have a strong model. From-scratch training gets you the ceiling (4x to >10x throughput gains) but costs two orders of magnitude more compute. Companies like Qwen, DeepSeek, and NVIDIA take this path because they get to co-design architecture and data from day one.

There's a middle path: slot the architecture change mid-training. Pretrain with full attention to build a strong foundation, then switch to a hybrid mid-training to capture efficiency gains. If it doesn't work, you stay with full attention and haven't lost the pretrain investment.

The industry is moving. Qwen, DeepSeek, and NVIDIA are shipping efficient hybrids because they figured out that throughput determines RL capacity, and RL capacity determines who wins on model quality. They're treating architecture as strategic, not just operational.

If you're renting inference through APIs, your roadmap depends on someone else's architecture priorities. When you need longer context or agent workflows that keep full state, you're constrained by their optimization choices and economics.

Own your training loop, you control which capabilities exist. Own your architecture, you control which capabilities deploy. The paths are proven. The decision is which one matches your constraints.

---

Appendix: Generating the Simplex Diagram
----------------------------------------

This [Literate Haskell](https://wiki.haskell.org/Literate_programming) appendix
contains the code that generates the quality-speed-cost simplex diagram shown
at the top of this essay.

We use a few language extensions and imports:

\begin{code}
  {-# LANGUAGE OverloadedStrings #-}
  {-# LANGUAGE RecordWildCards   #-}

  module SpeedCreatesCapability where

  import qualified Data.Text              as T
  import qualified Data.Text.Lazy.IO      as TL
  import qualified Data.Text.Lazy.Builder as B
  import           Data.Text.Lazy.Builder (Builder)
  import           Numeric                (showFFloat)
  import           System.Directory       (createDirectoryIfMissing)
  import           System.FilePath        (takeDirectory)
\end{code}

Configuration and data types:

\begin{code}
  -- | Canvas dimensions
  width, height :: Double
  width  = 900
  height = 600

  marginX, marginY :: Double
  marginX = 60
  marginY = 60

  -- | Barycentric coordinates: (quality, speed, cost)
  data BaryCoord = Bary Double Double Double
    deriving (Show, Eq)

  -- | Normalized coordinates in [0,1]^2
  data NormCoord = Norm Double Double
    deriving (Show, Eq)

  -- | Pixel coordinates
  data PixelCoord = Pixel Double Double
    deriving (Show, Eq)

  -- | A region in the simplex
  data Region = Region
    { regionName     :: T.Text
    , regionColor    :: T.Text
    , regionOpacity  :: Double
    , regionVertices :: [BaryCoord]
    } deriving (Show, Eq)
\end{code}

Region positioning philosophy:

These barycentric coordinates are tuned to show regions that reflect
real-world architectural tradeoffs, not extreme theoretical corners.

Key design principles:

1. Each region is a large triangle to show the viable design space
2. Regions overlap because production models can sit in multiple zones
3. Quality values reflect second/third-order effects:
   - Linear attention can reach higher quality by spending speed on RL + CoT
   - Hybrids bridge the gap between full attention quality and efficient speed/cost

Quality ranges:

- Full attention: q ∈ [0.60, 0.90] - top tier, but extends down to overlap hybrids
- Hybrids:        q ∈ [0.35, 0.85] - bridge region, overlaps both extremes
- Linear:         q ∈ [0.20, 0.52] - bottom edge is fast/cheap, top reflects RL gains

The apex of linear/hybrid regions represents spending efficiency on *capability*,
not just on lower cloud bills. You burn surplus speed on RL + test-time compute
to climb the quality ladder at roughly fixed cost. If we wanted "slash costs at
any latency", we'd make it right-leaning (c > s), but that contradicts the essay:
we use speed primarily to unlock deeper reasoning at production latency. Therefore,
the linear and hybrid regions are slightly left-leaning (s > c).

\begin{code}
  regions :: [Region]
  regions =
    [ Region
        { regionName    = "Full\nattention\nmodels"
        , regionColor   = "#9ca3af"  -- muted gray
        , regionOpacity = 0.28
        , regionVertices =
            -- q ∈ [0.60, 0.90]: High quality, modest speed/cost
            [ Bary 0.90 0.05 0.05  -- FA1: near pure quality
            , Bary 0.60 0.30 0.10  -- FA2: trade some quality for speed
            , Bary 0.60 0.10 0.30  -- FA3: trade some quality for cost
            ]
        }
    , Region
        { regionName    = "Linear\nattention\nmodels"
        , regionColor   = "#f59e0b"  -- pale amber
        , regionOpacity = 0.18
        , regionVertices =
            -- q ∈ [0.20, 0.52]: Bottom edge hugs speed/cost extremes
            [ Bary 0.20 0.70 0.10  -- L1: speed-focused extreme
            , Bary 0.20 0.10 0.70  -- L2: cost-focused extreme
            , Bary 0.52 0.35 0.13  -- L3: quality via RL + CoT
            ]
        }
    , Region
        { regionName    = "Hybrid\nmodels"
        , regionColor   = "#2563eb"  -- rich blue
        , regionOpacity = 0.30
        , regionVertices =
            -- q ∈ [0.35, 0.85]: Bridge between full and linear
            [ Bary 0.85 0.09 0.06  -- H1: high quality, closer to full
            , Bary 0.35 0.55 0.10  -- H2: speed-leaning, toward linear
            , Bary 0.35 0.10 0.55  -- H3: cost-leaning, symmetric
            ]
        }
    ]
\end{code}

Barycentric coordinate system:

The outer triangle vertices in normalized space (x, y) ∈ [0,1]²:

\begin{code}
  -- | Quality vertex (top)
  qVertex :: NormCoord
  qVertex = Norm 0.5 1.0

  -- | Speed vertex (bottom-left)
  sVertex :: NormCoord
  sVertex = Norm 0.0 0.0

  -- | Cost vertex (bottom-right)
  cVertex :: NormCoord
  cVertex = Norm 1.0 0.0
\end{code}

Coordinate transformations:

\begin{code}
  -- | Map barycentric (q, s, c) to normalized (x, y)
  baryToNorm :: BaryCoord -> NormCoord
  baryToNorm (Bary qCoord sCoord cCoord) =
    let Norm qx qy = qVertex
        Norm sx sy = sVertex
        Norm cx cy = cVertex
        x = qCoord * qx + sCoord * sx + cCoord * cx
        y = qCoord * qy + sCoord * sy + cCoord * cy
    in Norm x y

  -- | Map normalized (x, y) to pixel coordinates
  normToPixels :: NormCoord -> PixelCoord
  normToPixels (Norm x y) =
    let scaleX = width  - 2 * marginX
        scaleY = height - 2 * marginY
        pxX    = marginX + x * scaleX
        pxY    = height - marginY - y * scaleY  -- Flip y so quality is at the top
    in Pixel pxX pxY

  -- | Map barycentric to pixels (composition)
  baryToPixels :: BaryCoord -> PixelCoord
  baryToPixels = normToPixels . baryToNorm
\end{code}

SVG generation helpers:

We use a tiny SVG AST and builder to avoid ad-hoc string concatenation.

\begin{code}
  -- | Minimal SVG AST
  data Svg
    = Elem T.Text [(T.Text, T.Text)] [Svg]
    | Txt  T.Text

  -- | Format a Double with one decimal place
  format1 :: Double -> T.Text
  format1 x = T.pack (showFFloat (Just 1) x "")

  -- | Convert Text to a Builder
  tB :: T.Text -> Builder
  tB = B.fromText

  -- | Render attributes: key="value"
  renderAttrs :: [(T.Text, T.Text)] -> Builder
  renderAttrs =
    foldMap (\(k, v) -> " " <> tB k <> "=\"" <> tB v <> "\"")

  -- | Render the SVG AST to a Builder
  renderSvg :: Svg -> Builder
  renderSvg (Txt t) = tB t
  renderSvg (Elem name attrs children) =
    "<" <> tB name <> renderAttrs attrs <>
      if null children
        then "/>"
        else ">" <> foldMap renderSvg children <> "</" <> tB name <> ">"

  -- | Smart constructors
  elem_ :: T.Text -> [(T.Text, T.Text)] -> [Svg] -> Svg
  elem_ = Elem

  txt_ :: T.Text -> Svg
  txt_ = Txt

  -- | Polygon helper
  polygon :: [PixelCoord] -> T.Text -> Double -> Svg
  polygon pts color opacity =
    let pointText =
          T.intercalate " "
            [ format1 x <> "," <> format1 y
            | Pixel x y <- pts
            ]
    in elem_ "polygon"
         [ ("points",       pointText)
         , ("fill",         color)
         , ("fill-opacity", format1 opacity)
         , ("stroke",       color)
         , ("stroke-width", "2")
         ]
         []

  -- | Multi-line text helper
  textBlock :: PixelCoord -> T.Text -> Int -> Int -> Svg
  textBlock (Pixel x y) txt size weight =
    case T.splitOn "\n" txt of
      [single] ->
        elem_ "text" (baseAttrs y) [txt_ single]
      ls ->
        let lineSpacing = fromIntegral size * 1.2
            n           = length ls
            startY      = y - fromIntegral (n - 1) * lineSpacing / 2
            mkLine i lineTxt =
              let lineY = startY + fromIntegral i * lineSpacing
              in elem_ "text" (baseAttrs lineY) [txt_ lineTxt]
        in elem_ "g" [] (zipWith mkLine [0..] ls)
    where
      baseAttrs yPos =
        [ ("x",           format1 x)
        , ("y",           format1 yPos)
        , ("text-anchor", "middle")
        , ("font-size",   T.pack (show size))
        , ("font-weight", T.pack (show weight))
        , ("fill",        "#1f2937")
        ]
\end{code}

Main SVG building function:

\begin{code}
  -- | The full SVG node (without XML prolog)
  svgRoot :: Svg
  svgRoot =
    let -- Background
        background =
          elem_ "rect"
            [ ("width",  format1 width)
            , ("height", format1 height)
            , ("fill",   "#ffffff")
            ] []

        -- Outer simplex
        outerVertices = map normToPixels [qVertex, sVertex, cVertex]
        outerTriangle = polygon outerVertices "#d1d5db" 0.0

        -- Region triangles
        regionPolys =
          [ polygon (map baryToPixels (regionVertices r))
                    (regionColor r)
                    (regionOpacity r)
          | r <- regions
          ]

        -- Axis labels
        Pixel qx qy = normToPixels qVertex
        Pixel sx sy = normToPixels sVertex
        Pixel cx cy = normToPixels cVertex

        axisLabels =
          [ textBlock (Pixel qx (qy - 15)) "Quality" 20 600
          , textBlock (Pixel sx (sy + 25)) "Speed"   20 600
          , textBlock (Pixel cx (cy + 25)) "Low cost" 20 600
          ]

        -- Region labels at centroids
        regionLabels =
          [ let coords = map baryToPixels (regionVertices r)
                xs     = [x | Pixel x _ <- coords]
                ys     = [y | Pixel _ y <- coords]
                n      = fromIntegral (length coords)
                mx     = sum xs / n
                my     = sum ys / n
                w      = if "Hybrid" `T.isInfixOf` regionName r
                            then 600
                            else 500
            in textBlock (Pixel mx my) (regionName r) 20 w
          | r <- regions
          ]

        styleNode =
          elem_ "style" []
            [ txt_
                "text {\n      font-family: 'PragmataPro', monospace;\n    }\n"
            ]

        intText n = T.pack (show (round n :: Int))

    in elem_ "svg"
         [ ("width",  intText width)
         , ("height", intText height)
         , ("viewBox"
           , T.concat
               [ "0 0 "
               , intText width
               , " "
               , intText height
               ])
         , ("xmlns", "http://www.w3.org/2000/svg")
         ]
         (  [styleNode, background, outerTriangle]
         ++ regionPolys
         ++ axisLabels
         ++ regionLabels
         )
\end{code}

Writing the SVG to a file:

\begin{code}
  -- | Full document builder (XML prolog + SVG)
  documentBuilder :: Builder
  documentBuilder =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    <> "<?xml-stylesheet type=\"text/css\" href=\"/css/fonts.css\" ?>\n"
    <> renderSvg svgRoot
    <> "\n"

  writeSVG :: FilePath -> IO ()
  writeSVG path = do
    createDirectoryIfMissing True (takeDirectory path)
    TL.writeFile path (B.toLazyText documentBuilder)

  main :: IO ()
  main = writeSVG "images/speed-creates-capability.svg"
\end{code}
