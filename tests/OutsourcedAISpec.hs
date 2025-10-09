{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module OutsourcedAISpec where

import Test.Hspec
import Test.Hspec.QuickCheck (prop, modifyMaxSuccess)
import Test.QuickCheck hiding ((.&.))
import OutsourcedAI
import Quantity (Quantity(..), (*@), (./), dollar, second, month, hour, one, inUnit)
import Data.Ord (clamp)

-- Define baseExogenous for testing purposes
baseExogenous :: Exogenous
baseExogenous = baseExogenousFor (1 *@ month)

spec :: Spec
spec = do
  describe "Revenue calculation" $ do
    it "returns 0 for success rate below 0.6" $ do
      (expectedRevenue 0.5) `inUnit` (dollar ./ task) `shouldBe` 0
      (expectedRevenue 0.59) `inUnit` (dollar ./ task) `shouldBe` 0
    
    it "scales linearly between 0.6 and 0.9" $ do
      let rev = (expectedRevenue 0.7) `inUnit` (dollar ./ task)
      rev `shouldSatisfy` (\r -> abs (r - 0.5) < 0.01)
      
    it "provides bonus above 0.9" $ do
      let rev = (expectedRevenue 0.95) `inUnit` (dollar ./ task)
      rev `shouldSatisfy` (\r -> abs (r - 1.6) < 0.01)

  describe "Quality-driven demand" $ do
    it "increases demand when success rate exceeds threshold" $ do
      let demand = qualityDrivenDemand baseExogenous 0.8
      demand `shouldSatisfy` (> (0 *@ one ./ month))
      
    it "decreases demand when success rate is below threshold" $ do
      let demand = qualityDrivenDemand baseExogenous 0.4
      demand `shouldSatisfy` (< (0 *@ one ./ month))

  describe "Profitability-driven demand" $ do
    it "increases demand with positive margins" $ do
      let state = baseState { successRate = 0.8, deployment = External }
      let demand = profitabilityDrivenDemand baseExogenous (usage state) (deployment state) (successRate state)
      demand `shouldSatisfy` (> (0 *@ one ./ month))

  describe "Market saturation" $ do
    it "creates no crowding effect at low usage" $ do
      let lowUsage = 1.0e6 *@ task ./ second
      let crowd = crowdingOut baseExogenous lowUsage
      crowd `shouldSatisfy` (\c -> abs c < (0.01 *@ one ./ month))
      
    it "creates significant crowding near capacity" $ do
      let highUsage = 9.0e9 *@ task ./ second
      let crowd = crowdingOut baseExogenous highUsage
      crowd `shouldSatisfy` (> (0.4 *@ one ./ month))

  describe "Learning dynamics" $ do
    it "improves external success rate over time" $ do
      let state = baseState { deployment = External, successRate = 0.65 }
      let dt = 1 *@ month
      let (successRate', _) = updateLearning dt baseExogenous state
      successRate' `shouldSatisfy` (> successRate state)
      
    it "accumulates data advantage for internal deployment" $ do
      let state = baseState { deployment = Internal, usage = 1.0e3 *@ task ./ second, dataAdvantage = 0.1 }
      let dt = 1 *@ month
      let (_, dataAdvantage') = updateLearning dt baseExogenous state
      dataAdvantage' `shouldSatisfy` (> dataAdvantage state)
      
    it "caps success rate at ceiling" $ do
      let state = baseState { deployment = External, successRate = 0.74 }
      let dt = 10 *@ month
      let (successRate', _) = updateLearning dt baseExogenous state
      successRate' `shouldSatisfy` (<= externalSuccessRateCeiling baseExogenous)

  describe "Cost calculations" $ do
    it "calculates external cost correctly" $ do
      let usage = 1000 *@ task ./ second
      let cost = externalCost baseExogenous * usage
      cost `inUnit` (dollar ./ second) `shouldSatisfy` (\c -> abs (c - 20) < 0.1)
      
    it "includes fixed and variable costs for internal" $ do
      let usage = 1.0e6 *@ task ./ second
      let cost = internalCost baseExogenous usage
      cost `shouldSatisfy` (\c -> magnitude c > 0)
      
    it "scales internal nodes with usage" $ do
      let lowUsage = 1.0e5 *@ task ./ second
      let highUsage = 1.0e7 *@ task ./ second
      let lowNodes = totalNodes baseExogenous lowUsage
      let highNodes = totalNodes baseExogenous highUsage
      highNodes `shouldSatisfy` (> lowNodes)

  describe "Deployment strategies" $ do
    it "alwaysExternal returns External" $ do
      alwaysExternal baseExogenous baseState `shouldBe` External
      
    it "alwaysInternal returns Internal" $ do
      alwaysInternal baseExogenous baseState `shouldBe` Internal
      
    it "breakEven starts with External at low usage" $ do
      let state = baseState { usage = 1.0e5 *@ task ./ second }
      breakEven baseExogenous state `shouldBe` External

  describe "Simulation" $ do
    it "advances time correctly" $ do
      let clock = Clock { now = 0 *@ month, dt = 1 *@ month }
      let rows = take 3 $ simulate alwaysExternal clock baseExogenous baseState
      let times = map time rows
      times `shouldBe` [0 *@ month, 1 *@ month, 2 *@ month]

    it "updates state over time" $ do
      let clock = Clock { now = 0 *@ month, dt = 1 *@ month }
      let rows = take 5 $ simulate alwaysExternal clock baseExogenous baseState
      let successRates = map successRateBeforeStep rows
      -- Success rate should generally increase over time for external
      last successRates `shouldSatisfy` (> head successRates)

    it "applies deployment strategy at each step" $ do
      let clock = Clock { now = 0 *@ month, dt = 1 *@ month }
      let rows = take 3 $ simulate alwaysInternal clock baseExogenous baseState
      let deployments = map deploymentBeforeStep rows
      deployments `shouldBe` [Internal, Internal, Internal]

  describe "Update functions" $ do
    it "updateUsage respects market capacity" $ do
      let state = baseState { usage = 9.9e9 *@ task ./ second }
      let dt = 1 *@ month
      let usage' = updateUsage dt baseExogenous state
      usage' `shouldSatisfy` (<= marketCapacity baseExogenous)
      
    it "updateUsage prevents negative usage" $ do
      let state = baseState { usage = 1 *@ task ./ second, successRate = 0.3 }
      let dt = 1 *@ month
      let usage' = updateUsage dt baseExogenous state
      usage' `shouldSatisfy` (>= (0 *@ task ./ second))
      
    it "updateExogenous applies cost drift" $ do
      let dt = 1 *@ month
      let exogenous' = updateExogenous dt baseExogenous
      magnitude (externalCost exogenous') `shouldSatisfy` (< magnitude (externalCost baseExogenous))

  -- Property-based tests
  describe "Revenue properties" $ do
    prop "revenue is non-negative" $ \sr ->
      let sr' = clamp (0, 1) sr
          rev = expectedRevenue sr'
      in magnitude rev >= 0
    
    prop "revenue is monotonic in success rate" $ \sr1 sr2 ->
      let sr1' = clamp (0, 1) sr1
          sr2' = clamp (0, 1) sr2
          rev1 = expectedRevenue sr1'
          rev2 = expectedRevenue sr2'
      in sr1' <= sr2' ==> magnitude rev1 <= magnitude rev2
    
    prop "revenue is 0 below threshold (0.6)" $ \sr ->
      let sr' = clamp (0, 0.59) sr
          rev = expectedRevenue sr'
      in abs (magnitude rev) < 0.001

  describe "Learning dynamics properties" $ do
    prop "success rate stays bounded [0,1]" $ \sr deploy ->
      let sr' = clamp (0, 1) sr
          state = baseState { successRate = sr', deployment = if deploy then External else Internal }
          dt = 1 *@ month
          (newSr, _) = updateLearning dt baseExogenous state
      in newSr >= 0 && newSr <= 1
    
    prop "data advantage never decreases for Internal" $ \da ->
      let da' = clamp (0, 1) da
          state = baseState { deployment = Internal, dataAdvantage = da' }
          dt = 1 *@ month
          (_, newDa) = updateLearning dt baseExogenous state
      in newDa >= da'

  describe "Usage update properties" $ do
    prop "usage stays non-negative" $ \sr ->
      let sr' = clamp (0, 1) sr
          state = baseState { successRate = sr' }
          dt = 1 *@ month
          usage' = updateUsage dt baseExogenous state
      in magnitude usage' >= 0
    
    prop "usage respects market capacity" $ \sr ->
      let sr' = clamp (0, 1) sr
          state = baseState { successRate = sr', usage = 9.9e9 *@ task ./ second }
          dt = 1 *@ month
          usage' = updateUsage dt baseExogenous state
      in usage' <= marketCapacity baseExogenous

  describe "Cost calculation properties" $ do
    prop "costs are positive for positive usage" $ \u ->
      u > 0 ==>
        let usage = u *@ task ./ second
            extCost = externalCost baseExogenous * usage
            intCost = internalCost baseExogenous usage
        in magnitude extCost > 0 && magnitude intCost > 0
    
    prop "totalNodes respects minimum" $ \u ->
      u > 0 ==>
        let usage = u *@ task ./ second
            nodes = totalNodes baseExogenous usage
        in nodes >= internalBaseNodes baseExogenous

  describe "State invariants" $ do
    prop "state remains valid after step" $ \sr da deploy ->
      let sr' = clamp (0, 1) sr
          da' = clamp (0, 1) da
          state = baseState { successRate = sr', dataAdvantage = da', 
                            deployment = if deploy then External else Internal }
          dt = 1 *@ month
          clock = Clock { now = 0 *@ month, dt = dt }
          (_, newState) = step clock baseExogenous state
      in successRate newState >= 0 && successRate newState <= 1 && 
         dataAdvantage newState >= 0 && dataAdvantage newState <= 1 && 
         magnitude (usage newState) >= 0


