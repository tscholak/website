{-# LANGUAGE GeneralisedNewtypeDeriving #-}
{-# LANGUAGE InstanceSigs #-}

module Quantity
  ( Unit,
    unit,
    Quantity (..),
    unQ,
    (*@),
    (.*),
    (./),
    (.^),
    canonical,
    canonicalWith,
    convert,
    convertWith,
    one,
    -- Helpers
    toUnit,
    inUnit,
    scalar,
    qScale,
    qExp,
    qLog,
    qPow,
    qClamp,
    qCeiling,
    qFromIntegral,
    -- Common units
    second,
    minute,
    hour,
    day,
    month,
    year,
    dollar,
    meter,
    kg,
    volt,
    ampere,
    joule,
    newton,
    watt,
  )
where

import Data.List (find, intercalate, sort)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as M
import Data.Ratio (numerator)

-------------------------------------------------------------------
-- Unit representation
-------------------------------------------------------------------

newtype Unit = U (Map String Rational)
  deriving (Eq, Ord)

one :: Unit
one = U M.empty

unit :: String -> Unit
unit s = U (M.singleton s 1)

combine :: (Rational -> Rational -> Rational) -> Unit -> Unit -> Unit
combine f (U a) (U b) = U . M.filter (/= 0) $ M.unionWith f a b

-- Infix operators for Unit composition
infixl 7 .*, ./

infixr 8 .^

(.*) :: Unit -> Unit -> Unit
(.*) = combine (+)

(./) :: Unit -> Unit -> Unit
(./) (U a) (U b) = U . M.filter (/= 0) $ M.unionWith (+) a (M.map negate b)

(.^) :: Unit -> Rational -> Unit
(.^) (U m) k = U (M.map (* k) m)

-------------------------------------------------------------------
-- Quantity
-------------------------------------------------------------------

data Quantity a = Q {magnitude :: a, units :: Unit}
  deriving (Eq)

infixr 0 *@

(*@) :: a -> Unit -> Quantity a
x *@ u = Q x u

-- extract the magnitude if the unit is dimensionless
unQ :: Quantity a -> a
unQ (Q x (U m))
  | M.null m = x
  | otherwise = hasUnitsError "unQ"

-------------------------------------------------------------------
-- Num / Fractional instances
-------------------------------------------------------------------

hasUnitsError :: String -> a
hasUnitsError op = error $ op ++ ": quantity has units"

incompatibleUnitsError :: String -> Unit -> Unit -> a
incompatibleUnitsError op u v = error $ op ++ ": incompatible units: " ++ prettyUnit u ++ " vs " ++ prettyUnit v

-- Helper function that tries canonical normalization before operations
withCompatibleUnits :: (Fractional a) => String -> (a -> a -> b) -> Quantity a -> Quantity a -> Quantity b
withCompatibleUnits opName op q1@(Q x u) q2@(Q y v)
  | u == v = Q (op x y) u
  | u' == v' = Q (op x' y') u'
  | otherwise = incompatibleUnitsError opName u' v'
  where
    Q x' u' = canonical q1
    Q y' v' = canonical q2

instance (Fractional a, Ord a) => Ord (Quantity a) where
  compare q1 q2 = magnitude $ withCompatibleUnits "compare" compare q1 q2

instance (Fractional a, Num a) => Num (Quantity a) where
  q1 + q2 = withCompatibleUnits "add" (+) q1 q2
  q1 - q2 = withCompatibleUnits "subtract" (-) q1 q2

  (Q x u) * (Q y v) = Q (x * y) (u .* v)
  negate (Q x u) = Q (negate x) u
  abs (Q x u) = Q (abs x) u
  signum (Q x _) = Q (signum x) one
  fromInteger n = Q (fromInteger n) one

instance (Fractional a) => Fractional (Quantity a) where
  (Q x u) / (Q y v) = Q (x / y) (u ./ v)
  recip (Q x u) = Q (recip x) (one ./ u)
  fromRational r = Q (fromRational r) one

instance (Show a) => Show (Quantity a) where
  show (Q x (U m))
    | M.null m = show x
    | otherwise = show x <> " " <> prettyUnit (U m)

-------------------------------------------------------------------
-- Pretty-printing
-------------------------------------------------------------------

prettyUnit :: Unit -> String
prettyUnit (U m)
  | M.null m = "1"
  | otherwise =
      let showPow (s, 1) = s
          showPow (s, e) = s <> "^" <> show (fromRational e :: Double)
          units = map showPow (M.toList m)
       in case units of
            [] -> "1"
            [u] -> u
            _ -> intercalate "·" units

-------------------------------------------------------------------
-- Helpers
-------------------------------------------------------------------

-- | Convert to the given unit.
-- Errors if units are incompatible.
toUnit :: (Fractional a) => Quantity a -> Unit -> Quantity a
toUnit q u = convert q (1 *@ u)

-- | Get the dimensionless magnitude in the given unit.
-- Errors if units are incompatible.
inUnit :: (Fractional a) => Quantity a -> Unit -> a
inUnit q u = magnitude (toUnit q u)

-- | Extract dimensionless scalar value, with automatic unit conversion.
-- Errors if the quantity cannot be made dimensionless.
scalar :: (Fractional a) => Quantity a -> a
scalar q = q `inUnit` one

-- | Multiply a quantity by a dimensionless scalar.
qScale :: Num a => a -> Quantity a -> Quantity a
qScale s (Q m u) = Q (s*m) u

-- | Exponential on a dimensionless quantity.
-- Errors if the quantity has units.
qExp :: (Floating a) => Quantity a -> a
qExp q = exp (unQ q)

-- | Natural log of a dimensionless quantity.
-- Errors if the quantity has units or is non-positive.
qLog :: (Floating a, Ord a) => Quantity a -> a
qLog q = log (unQ q)

-- | Raise a dimensionless quantity to a real power.
-- Errors if the quantity has units.
qPow :: (Floating a) => Quantity a -> a -> a
qPow q p = (unQ q) ** p

-- | Clamp a quantity between two bounds, converting bounds to the quantity's unit.
-- Errors if bounds are incompatible with q.
qClamp :: (Fractional a, Ord a) => (Quantity a, Quantity a) -> Quantity a -> Quantity a
qClamp (lo, hi) q =
  let Q lo' uq = convert lo q
      Q hi' _  = convert hi q
      x        = max lo' (min (magnitude q) hi')
  in Q x uq

-- | Ceiling of a real-valued quantity, returning an integral quantity with the same unit.
qCeiling :: (RealFrac a, Integral b) => Quantity a -> Quantity b
qCeiling (Q x u) = Q (ceiling x) u

-- | Convert the magnitude of a quantity from an integral type to a numeric type, preserving units.
qFromIntegral :: (Integral a, Num b) => Quantity a -> Quantity b
qFromIntegral (Q x u) = Q (fromIntegral x) u

-------------------------------------------------------------------
-- Common units
-------------------------------------------------------------------

second, minute, hour, day, month, year :: Unit
second = unit "s"
minute = unit "min"
hour = unit "h"
day = unit "day"
month = unit "month"
year = unit "year"

dollar :: Unit
dollar = unit "$"

meter, kg, volt, ampere, joule, newton, watt :: Unit
meter = unit "m"
kg = unit "kg"
volt = unit "V"
ampere = unit "A"
joule = unit "J"
newton = unit "N"
watt = unit "W"

-------------------------------------------------------------------
-- Built-in conversion system
-------------------------------------------------------------------

-- Atomic unit substitution rules
data SubstitutionRule = SubstitutionRule
  { fromUnit :: String,
    toScale :: Rational,
    toUnits :: Unit
  }

type RewriteRules = [SubstitutionRule]

-- Define atomic conversions to base units
baseUnitRules :: RewriteRules
baseUnitRules =
  [ -- Time units -> seconds
    SubstitutionRule {fromUnit = "min", toScale = 60, toUnits = second},
    SubstitutionRule {fromUnit = "h", toScale = 3600, toUnits = second},
    SubstitutionRule {fromUnit = "day", toScale = (24 * 60 * 60), toUnits = second},
    SubstitutionRule {fromUnit = "month", toScale = (30 * 24 * 60 * 60), toUnits = second},
    SubstitutionRule {fromUnit = "year", toScale = (365 * 24 * 60 * 60), toUnits = second},
    -- Energy/Force -> kg⋅m⋅s⁻²
    SubstitutionRule {fromUnit = "N", toScale = 1, toUnits = (kg .* meter ./ (second .^ 2))},
    SubstitutionRule {fromUnit = "J", toScale = 1, toUnits = (kg .* (meter .^ 2) ./ (second .^ 2))},
    -- Electrical -> base units
    SubstitutionRule {fromUnit = "V", toScale = 1, toUnits = (kg .* (meter .^ 2) ./ (ampere .* (second .^ 3)))},
    SubstitutionRule {fromUnit = "W", toScale = 1, toUnits = (kg .* (meter .^ 2) ./ (second .^ 3))}
    -- Note: V⋅A will automatically become watts through V expansion
  ]

canonical :: (Fractional a) => Quantity a -> Quantity a
canonical = canonicalWith baseUnitRules

canonicalWith :: (Fractional a) => RewriteRules -> Quantity a -> Quantity a
canonicalWith rules q = rewriteToFixpoint rules [] q
  where
    rewriteToFixpoint :: (Fractional a) => RewriteRules -> [Unit] -> Quantity a -> Quantity a
    rewriteToFixpoint rules seen q@(Q x u)
      | u `elem` seen = q -- Cycle detected, stop here
      | otherwise =
          let q' = rewriteOnce rules q
           in if units q' == u
                then simplifyUnit q' -- Fixpoint reached, now simplify
                else rewriteToFixpoint rules (u : seen) q'

    rewriteOnce :: (Fractional a) => RewriteRules -> Quantity a -> Quantity a
    rewriteOnce rules (Q x (U unitMap)) =
      M.foldlWithKey' applyRule (Q x one) unitMap
      where
        applyRule (Q acc uAcc) unitName exponent =
          case findRule unitName rules of
            Nothing -> Q acc (uAcc .* (unit unitName .^ exponent))
            Just (SubstitutionRule _ scale targetUnit) ->
              let scaleFactor = fromRational (scale ^^ numerator exponent)
                  expandedUnit = targetUnit .^ exponent
               in Q (acc * scaleFactor) (uAcc .* expandedUnit)

    findRule :: String -> RewriteRules -> Maybe SubstitutionRule
    findRule unitName = find (\(SubstitutionRule u _ _) -> u == unitName)

    -- Simplify by canceling units and normalizing order
    simplifyUnit :: (Fractional a) => Quantity a -> Quantity a
    simplifyUnit (Q x (U unitMap)) =
      let simplified = M.filter (/= 0) unitMap -- Remove zero exponents
          sortedUnits = M.fromList $ sort $ M.toList simplified
       in Q x (U sortedUnits)

convert :: (Fractional a) => Quantity a -> Quantity a -> Quantity a
convert = convertWith baseUnitRules

convertWith :: (Fractional a) => RewriteRules -> Quantity a -> Quantity a -> Quantity a
convertWith rules q (Q _ targetU) =
  let Q x uSrc = canonicalWith rules q
      Q y uTgt = canonicalWith rules (1 *@ targetU)
   in if uSrc == uTgt
        then Q (x / y) targetU
        else incompatibleUnitsError "convert" uSrc uTgt
