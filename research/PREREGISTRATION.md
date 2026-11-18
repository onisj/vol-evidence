# Pre-registration: stress gauge backtest

Written and committed before the backtest was run. The gauge design, weights, hypotheses and decision rules below are fixed; results are reported against them whichever way they fall.

## Gauge design

Four inputs, each read on day *t* using only data up to and including *t*, each converted to a 0 to 100 "calm" score by its percentile rank within the trailing 252 trading days (the share of the prior 251 values strictly below today's value):

| Input | Series | Calm score |
|---|---|---|
| Implied volatility level | ^VIX | 100 − percentile of VIX |
| Volatility term structure | ^VIX / ^VIX3M | 100 − percentile of the ratio |
| Credit appetite | HYG / IEF price ratio | percentile of the ratio |
| Realised volatility | SPY 21-day realised volatility, annualised | 100 − percentile |

Gauge = equal-weighted mean of the four calm scores (0.25 each). Equal weights were chosen because there is no prior reason to favour any input; they will not be changed on the basis of this test.

Labels: **calm** at 60 or above, **normal** from 40 to 60, **stressed** below 40.

## Outcomes

For each day, SPY over the next 21 trading days: maximum drawdown from that day's close, realised volatility, total return, and whether a drawdown of 5% or more occurred.

## Hypotheses

- **H1 (sorting).** Across gauge quintiles, the chance of a 5%+ drawdown and next-month realised volatility both fall monotonically from the lowest quintile to the highest.
- **H2 (beats the baseline).** The gauge carries information about next-month drawdown beyond raw VIX: in a rank regression of next-month maximum drawdown on the gauge and on raw VIX, the gauge's coefficient has |t| > 2.
- **H3 (volatility).** The gauge's rank correlation with next-month realised volatility is stronger (more negative) than raw VIX's.

## Method

- Sample: every trading day where all inputs have a full 252-day history, through the last day with 21 days of forward data.
- Correlations and regressions use non-overlapping 21-day samples. 95% intervals for rank correlations come from a moving-block bootstrap (block length 6, 2,000 resamples, seed 7).
- Baselines: raw VIX (sign-flipped so higher means calmer) and the VIX-percentile input on its own.

## Decision rules

- If H2 or H3 fails, the README states that VIX alone is the better stress gauge, and the gauge stays as specified. Re-weighting on this data would be in-sample fitting.
- Any later change to the gauge requires a new pre-registration and an out-of-sample period.
