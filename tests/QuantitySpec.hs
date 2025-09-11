module QuantitySpec (spec) where

import Control.Exception (evaluate)
import Quantity
  ( Quantity,
    Unit,
    convert,
    convertWith,
    canonical,
    canonicalWith,
    one,
    unQ,
    scalar,
    toUnit,
    inUnit,
    qCeiling,
    qFromIntegral,
    qScale,
    qExp,
    qLog,
    qPow,
    qClamp,
    second,
    minute,
    hour,
    day,
    year,
    dollar,
    meter,
    kg,
    volt,
    ampere,
    joule,
    newton,
    watt,
    unit,
    (*@),
    (.*),
    (./),
    (.^),
  )
import Test.Hspec
  ( Spec,
    describe,
    errorCall,
    it,
    shouldBe,
    shouldThrow,
    shouldSatisfy,
  )

task, node :: Unit
task = unit "task"
node = unit "node"

spec :: Spec
spec = do
  describe "Basic unit operations" $ do
    it "creates simple quantities" $ do
      let q = 5 *@ meter
      show q `shouldBe` "5 m"

    it "multiplies quantities correctly" $ do
      let distance = 5 *@ meter
          time = 2 *@ second
          velocity = distance / time
      show velocity `shouldBe` "2.5 m·s^-1.0"

    it "adds quantities with same units" $ do
      let q1 = 3 *@ meter
          q2 = 7 *@ meter
          result = q1 + q2
      show result `shouldBe` "10.0 m"

  describe "Throughput calculation example" $ do
    it "calculates total cost correctly" $ do
      let taskPerNodeHour = task ./ node ./ hour
          dollarPerNodeHour = dollar ./ node ./ hour
          taskPerHour = task ./ hour

          throughput = 100 *@ taskPerNodeHour
          cost = 5 *@ dollarPerNodeHour
          usage = 50000 *@ taskPerHour

          totalCost = cost * usage / throughput

      show totalCost `shouldBe` "2500.0 $·h^-1.0"

  describe "Unit conversions" $ do
    it "converts time units" $ do
      let oneHour = 1 *@ hour
          inSeconds = convert oneHour (1 *@ second)
      show inSeconds `shouldBe` "3600.0 s"

    it "preserves dimensionless quantities" $ do
      let dimensionless = 42 *@ one
      show dimensionless `shouldBe` "42"

  describe "RealFrac operations" $ do
    it "qCeiling should preserve units" $ do
      let quantity = 5.7 *@ meter
          result = qCeiling quantity :: Quantity Integer
      show result `shouldBe` "6 m"

  describe "Unit arithmetic edge cases" $ do
    it "division by same unit gives dimensionless result" $ do
      let result = (6 *@ meter) / (2 *@ meter)
      show result `shouldBe` "3.0"

    it "handles multiple divisions correctly" $ do
      let acceleration = meter ./ second ./ second
          result = 10 *@ acceleration
      show result `shouldBe` "10 m·s^-2.0"

    it "handles fractional powers" $ do
      let sqrtMeter = meter .^ (1 / 2)
          result = 4 *@ sqrtMeter
      show result `shouldBe` "4 m^0.5"

    it "complex units that should simplify" $ do
      let forcePerMass = newton ./ kg -- N/kg = m/s²
          result = 9.8 *@ forcePerMass
      show result `shouldBe` "9.8 N·kg^-1.0"

    it "units cancel properly in complex expressions" $ do
      let energy = joule
          time = second
          power = energy ./ time -- Watts
          result = 100 *@ power
      show result `shouldBe` "100 J·s^-1.0"

  describe "Conversion edge cases" $ do
    it "converts dimensionless quantities" $ do
      let dimensionless = 42 *@ one
          converted = convert dimensionless (1 *@ one)
      show converted `shouldBe` "42.0"

    it "converts compound time units" $ do
      let frequency = one ./ second
          period = 2 *@ (second)
          freq = (1 *@ one) / period
          result = convert freq (1 *@ frequency)
      show result `shouldBe` "0.5 s^-1.0"

  describe "qCeiling edge cases" $ do
    it "handles exact integers" $ do
      let exact = 5.0 *@ meter
          result = qCeiling exact :: Quantity Integer
      show result `shouldBe` "5 m"

    it "handles negative quantities" $ do
      let negative = (-1.7) *@ meter
          result = qCeiling negative :: Quantity Integer
      show result `shouldBe` "-1 m"

    it "preserves units with different target types" $ do
      let quantity = 3.14 *@ meter
          result = qCeiling quantity :: Quantity Int
      show result `shouldBe` "4 m"

  describe "Pretty printing edge cases" $ do
    it "handles large positive exponents" $ do
      let bigPower = meter .^ 10
          result = 1 *@ bigPower
      show result `shouldBe` "1 m^10.0"

    it "handles mixed positive and negative exponents" $ do
      let mixed = kg .* (meter .^ 2) ./ (second .^ 2)
          result = 1 *@ mixed
      show result `shouldBe` "1 kg·m^2.0·s^-2.0"

    it "handles zero exponents (should be filtered out)" $ do
      let cancelled = meter .* (meter .^ (-1))
          result = 1 *@ cancelled
      show result `shouldBe` "1"

  describe "Canonical unit operations" $ do
    it "adds quantities with convertible time units" $ do
      let oneHour = 1.0 *@ hour
          oneMinute = 1.0 *@ minute
          result = oneHour + oneMinute
      show result `shouldBe` "3660.0 s"

    it "subtracts quantities with convertible time units" $ do
      let twoHours = 2.0 *@ hour
          thirtyMinutes = 30.0 *@ minute
          result = twoHours - thirtyMinutes
      show result `shouldBe` "5400.0 s"

    it "compares quantities with convertible time units" $ do
      let oneHour = 1.0 *@ hour
          sixtyMinutes = 60.0 *@ minute
      compare oneHour sixtyMinutes `shouldBe` EQ

    it "adds quantities with compound unit conversion" $ do
      let energy1 = 10.0 *@ joule
          energy2 = 5.0 *@ (newton .* meter)
          result = energy1 + energy2
      show result `shouldBe` "15.0 kg·m^2.0·s^-2.0"

    it "handles iterative canonicalization with V*A -> J/s -> N*m/s" $ do
      let power1 = 100.0 *@ (volt .* ampere)
          power2 = 50.0 *@ (joule ./ second)
          result = power1 + power2  
      show result `shouldBe` "150.0 kg·m^2.0·s^-3.0"

    it "demonstrates watt unit equivalence with V*A and J/s" $ do
      let power1 = 75.0 *@ watt
          power2 = 25.0 *@ (volt .* ampere)
          power3 = 50.0 *@ (joule ./ second)
          result = power1 + power2 + power3
      show result `shouldBe` "150.0 kg·m^2.0·s^-3.0"

    it "complex edge case: electrical power calculations with multiple conversions" $ do
      -- All are power units (kg⋅m²⋅s⁻³) requiring different conversion paths
      let electricalPower = (12.0 *@ volt) * (5.0 *@ ampere)  -- V⋅A → kg⋅m²⋅s⁻³
          mechanicalPower = (100.0 *@ newton) * (3.0 *@ (meter ./ second))  -- N⋅m/s → kg⋅m²⋅s⁻³  
          energyRate = (200.0 *@ joule) / (4.0 *@ second)  -- J/s → kg⋅m²⋅s⁻³
          totalPower = electricalPower + mechanicalPower + energyRate
      show totalPower `shouldBe` "410.0 kg·m^2.0·s^-3.0"

  describe "Error cases" $ do
    it "throws error on truly incompatible unit addition" $ do
      let distance = 5 *@ meter
          time = 2 *@ second
      evaluate (distance + time) `shouldThrow` errorCall "add: incompatible units: m vs s"

    it "throws error on truly incompatible unit subtraction" $ do
      let mass = 10 *@ kg
          length = 5 *@ meter
      evaluate (mass - length) `shouldThrow` errorCall "subtract: incompatible units: kg vs m"

    it "throws error on truly incompatible unit comparison" $ do
      let voltage = 12 *@ volt
          current = 3 *@ ampere
      evaluate (compare voltage current) `shouldThrow` errorCall "compare: incompatible units: A^-1.0·kg·m^2.0·s^-3.0 vs A"

    it "throws error on power vs energy addition after canonical normalization" $ do
      -- Edge case: both have similar base units but different time exponents
      let powerQuantity = 50.0 *@ watt  -- kg⋅m²⋅s⁻³
          energyQuantity = 100.0 *@ joule  -- kg⋅m²⋅s⁻²
      evaluate (powerQuantity + energyQuantity) `shouldThrow` errorCall "add: incompatible units: kg·m^2.0·s^-3.0 vs kg·m^2.0·s^-2.0"

  describe "unQ function - strict dimensionless extraction" $ do
    it "extracts magnitude from dimensionless quantities" $ do
      let dimensionless = 42.5 *@ one
      unQ dimensionless `shouldBe` 42.5

    it "throws error on quantities with units" $ do
      let withUnits = 5.0 *@ meter
      evaluate (unQ withUnits) `shouldThrow` errorCall "unQ: quantity has units"

  describe "scalar function - unit-safe extraction" $ do
    it "extracts dimensionless values directly" $ do
      let dimensionless = 42.5 *@ one
      scalar dimensionless `shouldBe` 42.5

    it "extracts scalar from converted dimensionless quantity" $ do
      let duration = 2.0 *@ hour
      let converted = convert duration (1 *@ second)
      let dimensionless = converted / (1 *@ second)  -- make dimensionless
      scalar dimensionless `shouldBe` 7200.0

  describe "Mathematical functions on dimensionless quantities" $ do
    it "qExp works on dimensionless quantities" $ do
      let q = 1.0 *@ one
      qExp q `shouldSatisfy` (\x -> abs (x - exp 1.0) < 0.01)

    it "qExp throws error on quantities with units" $ do
      let q = 1.0 *@ meter
      evaluate (qExp q) `shouldThrow` errorCall "unQ: quantity has units"

    it "qLog works on positive dimensionless quantities" $ do
      let q = (exp 2.0) *@ one
      qLog q `shouldSatisfy` (\x -> abs (x - 2.0) < 0.01)

    it "qPow raises dimensionless quantity to power" $ do
      let q = 2.0 *@ one
      qPow q 3.0 `shouldSatisfy` (\x -> abs (x - 8.0) < 0.01)

  describe "qScale - scalar multiplication" $ do
    it "scales quantity magnitude while preserving units" $ do
      let q = 5.0 *@ meter
      let scaled = qScale 3.0 q
      show scaled `shouldBe` "15.0 m"

    it "works with dimensionless quantities" $ do
      let q = 2.0 *@ one
      let scaled = qScale 0.5 q
      show scaled `shouldBe` "1.0"

  describe "qClamp - quantity clamping" $ do
    it "clamps quantity between bounds with same units" $ do
      let q = 15.0 *@ meter
      let bounds = (5.0 *@ meter, 10.0 *@ meter)
      let clamped = qClamp bounds q
      show clamped `shouldBe` "10.0 m"

    it "handles quantity within bounds" $ do
      let q = 7.0 *@ meter
      let bounds = (5.0 *@ meter, 10.0 *@ meter)
      let clamped = qClamp bounds q
      show clamped `shouldBe` "7.0 m"

    it "clamps to lower bound" $ do
      let q = 2.0 *@ meter
      let bounds = (5.0 *@ meter, 10.0 *@ meter)
      let clamped = qClamp bounds q
      show clamped `shouldBe` "5.0 m"

    it "works with convertible units" $ do
      let q = 2.0 *@ hour  -- 2 hours = 7200 seconds
      let bounds = (1.0 *@ hour, 1.5 *@ hour)  -- 3600-5400 seconds
      let clamped = qClamp bounds q
      show clamped `shouldBe` "1.5 h"

  describe "qFromIntegral - type conversion" $ do
    it "converts integer quantities to numeric quantities" $ do
      let intQ = 5 *@ meter :: Quantity Integer
      let doubleQ = qFromIntegral intQ :: Quantity Double
      show doubleQ `shouldBe` "5.0 m"

    it "preserves units during conversion" $ do
      let intQ = 100 *@ (meter ./ second) :: Quantity Integer
      let doubleQ = qFromIntegral intQ :: Quantity Double
      show doubleQ `shouldBe` "100.0 m·s^-1.0"

  describe "toUnit - direct unit conversion" $ do
    it "converts to compatible units" $ do
      let q = 1.0 *@ hour
      let inSeconds = toUnit q second
      show inSeconds `shouldBe` "3600.0 s"

    it "throws error on incompatible units" $ do
      let q = 5.0 *@ meter
      evaluate (toUnit q second) `shouldThrow` errorCall "convert: incompatible units: m vs s"

  describe "inUnit - extract value in specific unit" $ do
    it "extracts value in specified unit" $ do
      let q = 2.0 *@ hour
      inUnit q second `shouldBe` 7200.0

    it "works with dimensionless quantities" $ do
      let q = 42.0 *@ one
      inUnit q one `shouldBe` 42.0

    it "throws error on incompatible units" $ do
      let q = 5.0 *@ meter
      evaluate (inUnit q second) `shouldThrow` errorCall "convert: incompatible units: m vs s"
