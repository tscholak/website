---
title: "Stop Renting Moat: The Computational Proof"
date: Aug 21, 2025
teaser: WIP
tags:
  items: [ai, strategy, economics, cost-analysis]
---

Preliminaries
-------------

This is a computational essay about enterprise AI strategy and the commoditization trap. It uses code to model the economics of AI outsourcing and internal capability development. Code snippets are executable examples that demonstrate the concepts discussed.

Below some boilerplate code for this module that you can skip over if you just want to read the essay:

\begin{code}
  {-# LANGUAGE DataKinds #-}
  {-# LANGUAGE DeriveGeneric #-}
  {-# LANGUAGE DerivingStrategies #-}
  {-# LANGUAGE GeneralizedNewtypeDeriving #-}
  {-# LANGUAGE KindSignatures #-}
  {-# LANGUAGE NumericUnderscores #-}
  {-# LANGUAGE OverloadedStrings #-}
  {-# LANGUAGE RecordWildCards #-}
  {-# LANGUAGE TypeSynonymInstances #-}
  {-# LANGUAGE FlexibleInstances #-}
  {-# LANGUAGE MultiParamTypeClasses #-}
  {-# LANGUAGE ScopedTypeVariables #-}

  module OutsourcedAI where

  import GHC.Generics (Generic)
  import Text.Printf (printf)
  import Data.Ratio (Ratio, (%))
  import Quantity
    ( Quantity (Q, magnitude, units),
      unQ,
      Unit,
      convert,
      one,
      qCeiling,
      qFromIntegral,
      second,
      unit,
      (*@),
      (./),
    )
  import Data.Ord (clamp)
  
  -- import System.Directory   (getTemporaryDirectory)
  -- import System.FilePath    ((</>))
  -- import qualified Data.Text.IO as T
\end{code}

The Commoditization Trap: When Everyone Uses the Same APIs
----------------------------------------------------------

The most dangerous assumption in enterprise AI strategy is that you must choose between competing on cost or on differentiation. This false dichotomy misses how competitive advantage actually works in the AI era. In reality, **predictable unit cost and pricing power are themselves differentiators** once you control your AI stack.

Economic Rationale And Goals
----------------------------

We are comparing two deployment strategies for enterprise use:

\begin{code}
  data Deployment =
      External -- Use third-party APIs (OpenAI, Anthropic, etc.)
    | Internal -- Build and serve models in-house
\end{code}

Our primary goal is a simple, clear economic comparison:

  > Which deployment strategy delivers greater long-term profitability?

To answer this, we use a simple macroeconomic model that balances two key quantities:

\begin{code}
  -- | Expected revenue (from successfully completed AI tasks)
  type Revenue = Quantity Rational

  -- | Operational cost (to perform AI tasks)
  type Cost = Quantity Rational
\end{code}

A task is an abstract unit of AI work (e.g., a completed chat interaction, a document processed, typical of your business use case):

\begin{code}
  task = unit "task"
\end{code}

For our macroeconomic treatment of platform AI economics, we make simplifying assumptions:

1. **Abstracted task semantics**: All model completions are represented as a generic unit of work, `TaskUnit`. Complexity, structure, and purpose are absorbed into this abstract type.
2. **Implicit customer dynamics**: Instead of explicitly modeling customer behavior, we assume higher-quality tasks raise expected revenue per task, reflecting better conversion and retention, thus indirectly capturing improved Customer Lifetime Value (CLV). Therefore, expected revenue per task implicitly encodes platform economics and all customer-quality interactions.
3. **Success rate as strategic level**: The critical difference between external APIs and in-house models is how success rate (the probability of successful task completion) evolves:
  * API models start stronger but plateau sooner.
  * Internal models start weaker, but (due to data flywheel effects) improve more rapidly and plateau at a higher level.
4. **Cost structure simplification**: Operational cost per task reflects the marginal cost of AI work:
  * Token-based billing for APIs.
  * Infrastructure and R&D costs for internal models.

The key economic differentiator between strategies is how the success rate and the cost structure evolve over time. We model this evolution explicitly as a time series:

\begin{code}
  type SuccessRate = Rational -- between 0 and 1 (0% to 100%)
  type DataAdvantage = Rational -- between 0 and 1 (0% to 100%)
  type Throughput = Quantity Rational -- tasks per second
\end{code}

\begin{code}
  -- | Usage is the number of tasks per time unit
  type Usage = Quantity Rational

  data State = State
    { deployment :: Deployment -- ^ current deployment strategy
    , usage :: Usage -- ^ current usage in tasks per second
    , successRate :: SuccessRate  -- ^ current success rate
    , dataAdvantage :: DataAdvantage -- ^ current data advantage
    }

\end{code}

A deployment strategy is a function that cleanly decouples decision-making from the simulation dynamics. This allows us to write strategies without touching the core engine.

\begin{code}
  type DeploymentStrategy = Exogenous -> State -> Deployment
\end{code}

\begin{code}
  data Exogenous = Exogenous
    { externalCost :: Cost  -- ^ $ per task for external API
    , externalSuccessRate :: SuccessRate -- ^ success rate for external API
    , internalFixedCost :: Cost -- ^ fixed cost for internal deployment
    , internalVariableCost :: Cost -- ^ cost per time unit per node for internal deployment
    , internalThroughput :: Throughput -- ^ tasks per time unit per node for internal deployment
    , internalBaseNodes :: Quantity Rational -- ^ always-on nodes for internal deployment
    , marketCapacity :: Usage -- ^ maximum market size in tasks per time unit
    }
\end{code}

\begin{code}
  -- expected revenue per task based on success rate
  expectedRevenue :: SuccessRate -> Revenue
  expectedRevenue successRate = (retention successRate) *@ dollar ./ task
    where
      retention sr | sr < 0.6   = 0
                   | sr < 0.9   = 5 * (sr - 0.6)
                   | otherwise = 1.5 + 2 * (sr - 0.9)
\end{code}


Demand increase due to quality improvements models the idea that better outcomes directly attract more usage. When the success rate rises above a baseline threshold, customers are more likely to adopt the service, stick with it, and recommend it, which drives future demand. Below the threshold, improvements may have little effect on adoption, as the product is still perceived as unreliable.

\begin{code}
  -- demand increase due to quality improvements
  qualityDrivenGrowth :: SuccessRate -> Rational
  qualityDrivenGrowth successRate =
    qualitySensitivity * (successRate - successRateThreshold)
    where
      qualitySensitivity = 0.2 -- how much demand increases with success rate
      successRateThreshold = 0.6 -- success rate threshold for demand increase
\end{code}

The `qualitySensitivity` constant determines how strongly usage grows as success rate improves beyond the `successRateThreshold`. Above this threshold (set near the point where the product delivers consistently acceptable results, i.e. 0.6-0.7 for probabilistic AI systems) each 0.1 improvement in success rate typically yields an additional 1-3\% annual usage growth if `qualitySensitivity` is in the 0.1-0.3 range.

Usage change due to profitability captures the idea that healthy unit economics create both the means and the incentive to scale. When each unit of work generates a positive margin, a business can reinvest in marketing, infrastructure, and customer acquisition, which increases future usage. Conversely, negative margins force contraction—either by actively limiting work to high-value cases or through customer churn as prices rise or quality drops. In this way, profitability directly influences the rate at which demand grows or shrinks over time.

\begin{code}
  -- usage change due to profitability
  profitabilityDrivenGrowth :: Exogenous -> Usage -> Deployment -> SuccessRate -> Rational
  profitabilityDrivenGrowth exogenous@Exogenous{..} usage deployment successRate =
    clamp (- marginCap, marginCap) . unQ $ profitabilitySensitivity * margin
    where
      -- cost per task
      cost = case deployment of
        External -> externalCost
        Internal -> internalCost exogenous usage / processingCapacity exogenous usage

      -- margin per task
      margin = expectedRevenue successRate - cost

      profitabilitySensitivity = 5 *@ task ./ dollar -- how much demand increases with profitability
      marginCap = 0.1 -- cap on margin effect to prevent runaway growth
\end{code}

The `profitabilitySensitivity`` constant controls how strongly margins translate into usage growth: if one step represents a year, values in the 0.05-0.2 range for \$0.01/task margin are typical. The `marginCap` should be low enough (e.g., 0.1 for ±10\%/year) to avoid implausibly large swings from a single year's profitability spike. With 0.1, extreme margins (e.g., ±\$0.05/task) still only change usage ±10\%/year, which is reasonable for a mature enterprise platform.

Demand reduction due to market saturation reflects the slowdown that occurs as usage approaches the total addressable capacity of the market. Early growth is easy when there are many untapped customers, but as adoption nears the market's limit, each additional unit of demand is harder to capture. This "crowding out" effect models the natural tapering of growth under saturation, where further expansion requires disproportionate effort and yields diminishing returns.

\begin{code}
  -- demand reduction due to market saturation
  crowdingOut :: Exogenous -> Usage -> Rational
  crowdingOut Exogenous{..} usage =
    capacityPressure * unQ (usage / marketCapacity)
    where
      capacityPressure = 0.5 -- how much demand decreases with market saturation
\end{code}

The `capacityPressure` constant sets the strength of this drag: a value near 1.0 ensures that growth falls to zero as we hit `marketCapacity`, producing a realistic plateau; smaller values let growth continue even past nominal capacity, and the plateau will be softer.
Choosing `marketCapacity` so that the initial usage is only a few percent of capacity ensures saturation effects appear later in the simulation, not immediately.

Putting this all together, we can define the demand update rule as follows:

\begin{code}
  -- demand update rule
  updateUsage
    :: Exogenous
    -> State
    -> Usage
  updateUsage exogenous@Exogenous{..} State{..} =
    clamp (0 *@ task ./ second, marketCapacity) $
      usage * fromRational (1 + qualityDrivenGrowth successRate + profitabilityDrivenGrowth exogenous usage deployment successRate - crowdingOut exogenous usage)
\end{code}

\begin{code}
  updateRevenue
    :: Exogenous
    -> State
    -> Revenue
  updateRevenue exogenous State{..} = undefined

  updateProfit
    :: Exogenous
    -> State
    -> Profit
  updateProfit exogenous State{..} = undefined

  updateState
    :: DeploymentStrategy
    -> Exogenous
    -> State
    -> State
  updateState deploymentStrategy exogenous State{..} = 
    undefined

  simulate
    :: DeploymentStrategy
    -> [Exogenous]
    -> State
    -> [State]
  simulate _ [] s = [s]
  simulate strategy (e : es) s0 =
    let s1 = updateState strategy e s0
    in s1 : simulate strategy es s1
  
  alwaysExternal, alwaysInternal :: DeploymentStrategy
  alwaysExternal _ = undefined
  alwaysInternal _ = undefined

  waitAndSee :: DeploymentStrategy
  waitAndSee = undefined

  -- Break-even strategy: switch to internal when profitable
  -- and external when not profitable
  -- Problem: you need to know the future to do this optimally
  -- Problem: you may never break even if no investment in internal capabilities
  breakEven :: DeploymentStrategy
  breakEven = undefined

  -- total number of nodes needed to handle the given usage
  totalNodes :: Exogenous -> Usage -> Quantity Rational
  totalNodes Exogenous {..} usage =
    let neededNodes = qFromIntegral (qCeiling (usage / internalThroughput))
      in max internalBaseNodes neededNodes

  -- total processing capacity in tasks per time unit
  processingCapacity :: Exogenous -> Usage -> Quantity Rational
  processingCapacity exogenous@Exogenous{..} usage = internalThroughput * totalNodes exogenous usage

  -- internal cost per time unit
  internalCost :: Exogenous -> Usage -> Cost
  internalCost exogenous@Exogenous {..} usage = internalFixedCost + totalNodes exogenous usage * internalVariableCost

  type Profit = Quantity Rational

  dollar :: Unit
  dollar = unit "$"

  node :: Unit
  node = unit "node"
\end{code}
