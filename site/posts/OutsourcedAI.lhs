---
title: "Stop Renting Moat: The Computational Proof"
publication:
  status: draft
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
      inUnit,
      scalar,
      qCeiling,
      qFromIntegral,
      second,
      hour,
      month,
      year,
      dollar,
      unit,
      (*@),
      (./),
      qExp,
      qScale,
      qClamp,
    )
  import Data.Ord (clamp)
  import Graphics.Rendering.Chart.Easy
  import Graphics.Rendering.Chart.Backend.Diagrams
  import Control.Monad (forM_)

  type Scalar = Double

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
    deriving (Eq, Show)
\end{code}

Our primary goal is a simple, clear economic comparison:

  > Which deployment strategy delivers greater long-term profitability?

To answer this, we use a simple macroeconomic model that balances two key quantities:

\begin{code}
  -- | Expected revenue (from successfully completed AI tasks)
  type Revenue = Quantity Scalar

  -- | Operational cost (to perform AI tasks)
  type Cost = Quantity Scalar
\end{code}

A task is an abstract unit of AI work (e.g., a completed chat interaction, a document processed, typical of your business use case):

\begin{code}
  task :: Unit
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
  type SuccessRate = Scalar -- dimensionless, between 0 and 1 (0% to 100%)
  type DataAdvantage = Scalar -- dimensionless, between 0 and 1 (0% to 100%)
  type Throughput = Quantity Scalar -- tasks per time unit
\end{code}

\begin{code}
  -- | Usage is the number of tasks per time unit
  type Usage = Quantity Scalar

  data State = State
    { deployment :: !Deployment -- ^ current deployment strategy
    , usage :: !Usage -- ^ current usage in tasks per time unit
    , successRate :: !SuccessRate  -- ^ current success rate (0 to 1)
    , dataAdvantage :: !DataAdvantage -- ^ current data advantage (0 to 1)
    }

\end{code}

A deployment strategy is a function that cleanly decouples decision-making from the simulation dynamics. This allows us to write strategies without touching the core engine.

\begin{code}
  type DeploymentStrategy = Exogenous -> State -> Deployment
\end{code}

\begin{code}
  data Exogenous = Exogenous
    { -- external
      externalCost :: !Cost  -- ^ $ / task for external API
    , externalSuccessRateCeiling :: !SuccessRate -- ^ success rate for external API (at asymptote)
    , externalImprovementRate :: !(Quantity Scalar) -- ^ improvement rate for external API (1 / time unit)

      -- internal cost structure
    , internalFixedCost :: !Cost -- ^ internal fixed cost / time unit (R&D, baseline infra)
    , internalVariableCost :: !Cost -- ^ cost / time unit / active node
    , internalThroughput :: !Throughput -- ^ tasks / time unit / node
    , internalBaseNodes :: !(Quantity Scalar) -- ^ minimum nodes kept warm
    , internalSuccessRateCeiling :: !SuccessRate -- ^ success rate for internal model (at asymptote)
    , internalLearningRate :: !(Quantity Scalar) -- ^ demand increase due to learning effects (1 / time unit)
    , internalDataAccumulationRate :: !(Quantity Scalar) -- ^ data accumulation rate (1 / task)

      -- demand side
    , marketCapacity :: !Usage -- ^ maximum market size in tasks / time unit
    , qualitySensitivity :: !(Quantity Scalar) -- ^ demand increase due to quality improvements (1 / time unit)
    , qualitySuccessRateThreshold :: !SuccessRate -- ^ success rate threshold for demand increase
    , profitabilitySensitivity :: !(Quantity Scalar) -- ^ demand increase due to profitability improvements (task / $ / time unit)
    , profitabilityMarginCap :: !(Quantity Scalar) -- ^ cap on margin effect to prevent runaway growth (1 / time unit)
    , crowdingOutPressure :: !(Quantity Scalar) -- ^ demand reduction due to market saturation (1 / time unit)

      -- cost drift
    , driftExternalCost :: !(Quantity Scalar) -- ^ external cost drift (1 / time unit)
    , driftInternalFixedCost :: !(Quantity Scalar) -- ^ internal fixed cost drift (1 / time unit)
    , driftInternalVariableCost :: !(Quantity Scalar) -- ^ internal variable cost drift (1 / time unit)

      -- financial
    , discountPerStep :: !Scalar -- ^ discount rate for NPV calculations 
    }
\end{code}

\begin{code}
  -- | Expected revenue per successful task.
  -- Calibrate to a concrete use case.
  -- Here: piecewise toy curve: 0 until 0.6, then linear; bonus after 0.9.
  expectedRevenue :: SuccessRate -> Revenue
  expectedRevenue successRate = (retention successRate) *@ dollar ./ task
    where
      retention sr | sr < 0.6   = 0
                   | sr < 0.9   = 5 * (sr - 0.6)
                   | otherwise = 1.5 + 2 * (sr - 0.9)
\end{code}

Quality-Driven Demand
---------------------

Demand increase due to quality improvements models the idea that better outcomes directly attract more usage. When the success rate rises above a baseline threshold, customers are more likely to adopt the service, stick with it, and recommend it, which drives future demand. Below the threshold, improvements may have little effect on adoption, as the product is still perceived as unreliable.

\begin{code}
  -- | Demand increase due to quality improvements
  qualityDrivenDemand :: Exogenous -> SuccessRate -> Quantity Scalar
  qualityDrivenDemand Exogenous{..} successRate =
    (successRate - qualitySuccessRateThreshold) `qScale` qualitySensitivity
\end{code}

The exogenous `qualitySensitivity` constant determines how strongly usage grows as success rate improves beyond the `qualitySuccessRateThreshold`. Above this threshold (set near the point where the product delivers consistently acceptable results, i.e. `0.6` to `0.7` for probabilistic AI systems) each `0.1` improvement in success rate typically yields an additional 1 to 3 percent annual usage growth if `qualitySensitivity` is in the `0.1` to `0.3` range.

Profitability-Driven Demand
---------------------------

Usage change due to profitability captures the idea that healthy unit economics create both the means and the incentive to scale. When each unit of work generates a positive margin, a business can reinvest in marketing, infrastructure, and customer acquisition, which increases future usage. Conversely, negative margins force contraction, either by actively limiting work to high-value cases or through customer churn as prices rise or quality drops. In this way, profitability directly influences the rate at which demand grows or shrinks over time.

\begin{code}
  -- | Demand change due to profitability
  profitabilityDrivenDemand :: Exogenous -> Usage -> Deployment -> SuccessRate -> Quantity Scalar
  profitabilityDrivenDemand exogenous@Exogenous{..} usage deployment successRate =
    qClamp (- profitabilityMarginCap, profitabilityMarginCap) $ profitabilitySensitivity * margin
    where
      cost = case deployment of
        External -> externalCost
        Internal -> internalCost exogenous usage / processingCapacity exogenous usage
      margin = expectedRevenue successRate - cost
\end{code}

The exogenous `profitabilitySensitivity` constant controls how strongly margins translate into usage growth: if one step represents a year, values in the `0.05` to `0.2` range for USD 0.01 per task margin are typical. The `profitabilityMarginCap` constant should be low enough (e.g., `0.1` for plus/minus 10 percent per year) to avoid implausibly large swings from a single year's profitability spike. With `0.1`, extreme margins (e.g., USD 0.05 per task) still only change usage 10 percent per year, which is reasonable for a mature enterprise platform.

Market Saturation Effects
-------------------------

Demand reduction due to market saturation reflects the slowdown that occurs as usage approaches the total addressable capacity of the market. Early growth is easy when there are many untapped customers, but as adoption nears the market's limit, each additional unit of demand is harder to capture. This "crowding out" effect models the natural tapering of growth under saturation, where further expansion requires disproportionate effort and yields diminishing returns.

\begin{code}
  -- | Demand reduction due to market saturation
  crowdingOut :: Exogenous -> Usage -> Quantity Scalar
  crowdingOut Exogenous{..} usage =
    crowdingOutPressure * usage / marketCapacity
\end{code}

The exogenous `crowdingOutPressure` constant sets the strength of this drag: a value near `1.0` ensures that growth falls to zero as we hit `marketCapacity`, producing a realistic plateau. Smaller values let growth continue even past nominal capacity, and the plateau will be softer. Choosing `marketCapacity` so that the initial usage is only a few percent of capacity ensures saturation effects appear later in the simulation, not immediately.

Usage Update Rule
-----------------

Putting this all together, we can define the usage update rule as follows:

\begin{code}
  -- | Usage update
  updateUsage ::
    Duration ->
    Exogenous ->
    State ->
    Usage
  updateUsage dt exogenous@Exogenous {..} State {..} =
    let
      q = qualityDrivenDemand exogenous successRate
      p = profitabilityDrivenDemand exogenous usage deployment successRate
      c = crowdingOut exogenous usage
      factor = qExp $ (q + p - c) * dt
    in qClamp (0 *@ task ./ second, marketCapacity) $ factor `qScale` usage
\end{code}

Learning Dynamics
-----------------

We distinguish two deployment modes:

1. **External (exogenous improvement).**: Vendors do not learn from *our* traffic. Their models improve through global R&D, independent of our usage. We model this as a drift toward an asymptote `externalSuccessRateCeiling`, closing a fixed fraction of the remaining gap each time step to model diminishing returns. The `externalImprovementRate` controls how quickly this happens. This leads to a rapid initial improvement that slows as we approach the vendor's success rate. There is no `dataAdvantage` term here since external vendors do not learn from our data.

2. **Internal (data flywheel).**: In-house deployment compounds proprietary signal. Each step, completed tasks contribute to a `dataAdvantage`. The more successful tasks are served, the more proprietary data is accumulated. This data advantage then accelerates improvements in success rate, with diminishing returns as we approach the upper bound `1`. This captures the slow start, rapid mid-phase, and natural plateau characteristic of internal learning curves.

\begin{code}
  -- | Learning update
  updateLearning ::
    Duration ->
    Exogenous ->
    State ->
    (SuccessRate, DataAdvantage)
  updateLearning dt Exogenous{..} State{..} =
    case deployment of
      External ->
        -- asymptotic drift toward vendor's success rate
        let improve = 1 - qExp (- externalImprovementRate * dt)
            successRate' = clamp (0,1) $ successRate + improve * (externalSuccessRateCeiling - successRate)
        in (successRate', dataAdvantage) -- no data advantage growth

      Internal ->
        -- saturating growth driven by accumulated advantage
        let delta = clamp (0,1) $ scalar (internalDataAccumulationRate * usage * dt) * successRate
            dataAdvantage' = clamp (0,1) $ dataAdvantage + (1 - dataAdvantage) * delta
            improve = 1 - qExp (- internalLearningRate * dt)
            successRate' = clamp (0,1) $ successRate + improve * dataAdvantage' * (internalSuccessRateCeiling - successRate)
        in (successRate', dataAdvantage')
\end{code}

\begin{code}
  -- | Update the exogenous parameters over time (due to cost reductions)
  updateExogenous ::
    Duration ->
    Exogenous ->
    Exogenous
  updateExogenous dt exogenous@Exogenous {..} = 
    let fec = qExp (driftExternalCost * dt)
        fifc = qExp (driftInternalFixedCost * dt)
        fivc = qExp (driftInternalVariableCost * dt)
        in exogenous
          { externalCost = fec `qScale` externalCost
          , internalFixedCost = fifc `qScale` internalFixedCost
          , internalVariableCost = fivc `qScale` internalVariableCost
          }
\end{code}

\begin{code}
  type Time = Quantity Scalar
  type Duration = Quantity Scalar

  data Clock = Clock { now :: !Time, dt :: !Duration }
\end{code}

\begin{code}
  -- | Update the state by one time step
  step :: Clock -> Exogenous -> State -> (Exogenous, State)
  step Clock{..} exogenous state@State{..} = 
    let
      exogenous' = updateExogenous dt exogenous
      usage' = updateUsage dt exogenous' state
      (successRate', dataAdvantage') = updateLearning dt exogenous' state
      state' = state { usage = usage', successRate = successRate', dataAdvantage = dataAdvantage' }
    in (exogenous', state')
\end{code}

\begin{code}
  alwaysExternal, alwaysInternal :: DeploymentStrategy
  alwaysExternal _ _ = External
  alwaysInternal _ _ = Internal

  -- | Break-even strategy: start external; switch to internal when (rev - c_int) > (rev - c_ext) + eps AND usage above threshold
  -- Problem: you need to know the future to do this optimally
  -- Problem: you may never break even if no investment in internal capabilities
  breakEven :: DeploymentStrategy
  breakEven exogenous@Exogenous{..} State{..}
    | usage < thresholdUsage = External
    | marginInternal > marginExternal + epsilonPerTime = Internal
    | otherwise = External
    where
      thresholdUsage = 1.0e7 *@ task ./ second -- minimum usage to consider
      epsilonPerTask = 0.01 *@ dollar ./ task -- minimum margin improvement to switch
      epsilonPerTime = epsilonPerTask * usage
      revenue = expectedRevenue successRate * usage
      marginInternal = revenue - internalCost exogenous usage
      marginExternal = revenue - externalCost * usage

  -- total number of nodes needed to handle the given usage
  totalNodes :: Exogenous -> Usage -> Quantity Scalar
  totalNodes Exogenous {..} usage =
    let neededNodes = qFromIntegral . qCeiling $ usage / internalThroughput
      in max internalBaseNodes neededNodes

  -- total processing capacity in tasks per time unit
  processingCapacity :: Exogenous -> Usage -> Quantity Scalar
  processingCapacity exogenous@Exogenous{..} usage =
    internalThroughput * totalNodes exogenous usage

  -- internal cost per time unit
  internalCost :: Exogenous -> Usage -> Cost
  internalCost exogenous@Exogenous {..} usage =
    internalFixedCost + totalNodes exogenous usage * internalVariableCost

  baseState :: State
  baseState = State {
    deployment = External,
    usage = 5.0e8 *@ task ./ second,
    successRate = 0.65,
    dataAdvantage = 0
  }

  baseExogenousFor :: Duration -> Exogenous
  baseExogenousFor dt =
    Exogenous
      { externalCost = 0.02 *@ dollar ./ task,
        externalSuccessRateCeiling = 0.75,
        externalImprovementRate = 0.01 *@ one ./ month,
        internalFixedCost = 120_000 *@ dollar ./ month,
        internalVariableCost = 1.0e2 *@ dollar ./ hour ./ node,
        internalThroughput = 1.0e6 *@ task ./ second ./ node,
        internalBaseNodes = 1 *@ node,
        internalSuccessRateCeiling = 0.90,
        internalLearningRate = 0.05 *@ one ./ month,
        internalDataAccumulationRate = 0.01 *@ one ./ task,
        marketCapacity = 1.0e10 *@ task ./ second,
        qualitySensitivity = 0.2 *@ one ./ month,
        qualitySuccessRateThreshold = 0.6,
        profitabilitySensitivity = 5 *@ task ./ dollar ./ month,
        profitabilityMarginCap = 0.1 *@ one ./ month,
        crowdingOutPressure = 0.5 *@ one ./ month,
        driftExternalCost = log 0.99 *@ one ./ month,
        driftInternalFixedCost = log 1.005 *@ one ./ month,
        driftInternalVariableCost = log 0.995 *@ one ./ month,
        discountPerStep = let annualDiscountRate = 0.08 in exp (-annualDiscountRate * (inUnit dt year))
      }

  node :: Unit
  node = unit "node"
\end{code}

\begin{code}
  data Row = Row
    { time :: !Time,
      deploymentBeforeStep :: !Deployment,
      -- | tasks / time
      usageBeforeStep :: !Usage,
      -- | 0..1
      successRateBeforeStep :: !SuccessRate,
      -- | 0..1
      dataAdvantageBeforeStep :: !DataAdvantage,
      -- | \$ / task
      unitCostPerTask :: !(Quantity Scalar),
      -- | \$ / time
      revenueRate :: !(Quantity Scalar),
      -- | \$ / time
      costRate :: !(Quantity Scalar),
      -- | \$ / time
      profitRate :: !(Quantity Scalar),
      -- | \$ over this dt
      cashFlowForPeriod :: !(Quantity Scalar),
      -- | discounted $
      presentValueOfCashFlowForPeriod :: !(Quantity Scalar),
      -- | discounted $, cumulative
      cumulativePresentValueOfCashFlow :: !(Quantity Scalar)
    }
    deriving (Show)
\end{code}

\begin{code}
  simulate ::
    DeploymentStrategy ->
    Clock ->
    Exogenous ->
    State ->
    [Row]
  simulate strategy clock0 exogenous0 state0 =
    go (0 *@ dollar) clock0 1 exogenous0 state0
    where
      go ::
        Quantity Scalar ->
        Clock ->
        Scalar ->
        Exogenous ->
        State ->
        [Row]
      go cumulativePresentValueBeforeStep clockBeforeStep@Clock {..} discountMultiplierBeforeStep exogenousBeforeStep stateBeforeStep =
        let -- decide deployment for this step. state is still "before step"
            deploymentBeforeStep = strategy exogenousBeforeStep stateBeforeStep
            stateBeforeStep' = stateBeforeStep {deployment = deploymentBeforeStep}

            -- snapshot some "before step" observables we want to emit in the Row
            time = now
            usageBeforeStep = usage stateBeforeStep'
            successRateBeforeStep = successRate stateBeforeStep'
            dataAdvantageBeforeStep = dataAdvantage stateBeforeStep'

            -- per-time economics for this step
            revenueRate = expectedRevenue successRateBeforeStep * usageBeforeStep
            costRate = case deploymentBeforeStep of
              External -> externalCost exogenousBeforeStep * usageBeforeStep
              Internal -> internalCost exogenousBeforeStep usageBeforeStep
            profitRate = revenueRate - costRate

            -- cash flow and NPV for this step
            cashFlowForPeriod = profitRate * dt
            averageDiscountWithinStep = (1 - discountPerStep exogenousBeforeStep) / (- log (discountPerStep exogenousBeforeStep))
            presentValueOfCashFlowForPeriod = (discountMultiplierBeforeStep * averageDiscountWithinStep) `qScale` cashFlowForPeriod
            cumulativePresentValueOfCashFlow = cumulativePresentValueBeforeStep + presentValueOfCashFlowForPeriod

            -- unit cost snapshot for this step
            unitCostPerTask = case deploymentBeforeStep of
              External -> externalCost exogenousBeforeStep
              Internal ->
                internalCost exogenousBeforeStep usageBeforeStep
                  / processingCapacity exogenousBeforeStep usageBeforeStep

            row = Row {..}

            (exogenousNextStep, stateNextStep) = step clockBeforeStep exogenousBeforeStep stateBeforeStep'
            clockNextStep = clockBeforeStep {now = now + dt}
            discountMultiplierNextStep = discountMultiplierBeforeStep * discountPerStep exogenousBeforeStep
        in row : go cumulativePresentValueOfCashFlow clockNextStep discountMultiplierNextStep exogenousNextStep stateNextStep
\end{code}

\begin{code}
  pointsSuccessRateBeforeStep :: [Row] -> [(Scalar, Scalar)]
  pointsSuccessRateBeforeStep =
    map (\Row{..} -> (inUnit time month, successRateBeforeStep))

  pointsUnitCostPerTask :: [Row] -> [(Scalar, Scalar)]
  pointsUnitCostPerTask =
    map (\Row{..} -> (inUnit time month, inUnit unitCostPerTask (dollar ./ task)))

  pointsCumulativePresentValueOfCashFlow :: [Row] -> [(Scalar, Scalar)]
  pointsCumulativePresentValueOfCashFlow =
    map (\Row{..} -> (inUnit time month, inUnit cumulativePresentValueOfCashFlow dollar))

  main :: IO ()
  main = do
    let strategies =
          [ ("alwaysExternal", alwaysExternal)
          , ("alwaysInternal", alwaysInternal)
          , ("breakEven",      breakEven)
          ]

        monthlyClock = Clock { now = 0 *@ month, dt = 1 *@ month }

        horizonMonths = 60

        baseExogenous = baseExogenousFor (dt monthlyClock)

        takeH (name, strat) =
          (name, take horizonMonths (simulate strat monthlyClock baseExogenous baseState))

        runs = map takeH strategies

        fileOptions = FileOptions { _fo_size = (800, 600), _fo_format = SVG, _fo_fonts = loadSansSerifFonts }

    -- Success rate plot
    toFile fileOptions "success_rate.svg" $ do
      layout_title .= "Success Rate over Time"
      layout_x_axis . laxis_title .= "Month"
      layout_y_axis . laxis_title .= "Success rate"
      forM_ runs $ \(name, rows) ->
        plot (line name [pointsSuccessRateBeforeStep rows])

    -- Unit cost plot
    toFile fileOptions "unit_cost_per_task.svg" $ do
      layout_title .= "Unit Cost per Task ($)"
      layout_x_axis . laxis_title .= "Month"
      layout_y_axis . laxis_title .= "USD per task"
      forM_ runs $ \(name, rows) ->
        plot (line name [pointsUnitCostPerTask rows])

    -- Cumulative NPV plot
    toFile fileOptions "cumulative_present_value_of_profit.svg" $ do
      layout_title .= "Cumulative Present Value of Profit ($)"
      layout_x_axis . laxis_title .= "Month"
      layout_y_axis . laxis_title .= "USD"
      forM_ runs $ \(name, rows) ->
        plot (line name [pointsCumulativePresentValueOfCashFlow rows])
\end{code}