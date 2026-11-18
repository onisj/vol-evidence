# Pre-registration v3: does the v2 result replicate in other markets, and forward in time?

Written and committed before any v3 data was downloaded. Questions, models, scoring rules, hypotheses and decision rules are fixed; results will be reported against them whichever way they fall.

## Why

v2 found, on the S&P 500 from 2000 to 2026, that VIX beats the base rate for next-month risk, that adding the variance risk premium or a composite does not help, and that no model beat the base rate before sudden shocks. That is one market whose history I have studied twice. v3 tests the same claims where I have not looked:

- **Part A, other markets:** four markets with their own implied-volatility index, none of which I have examined in this project.
- **Part B, the future:** a public ledger of S&P 500 forecasts, committed before their outcomes exist.

## Part A: replication in four markets

| Market | Price series | Implied-volatility index |
|---|---|---|
| Nasdaq-100 | yfinance `^NDX` | yfinance `^VXN` |
| Russell 2000 | yfinance `^RUT` | yfinance `^RVX` |
| Emerging markets | yfinance `EEM` (ETF) | yfinance `^VXEEM` |
| Gold | yfinance `GLD` (ETF) | yfinance `^GVZ` |

A market is included only if price and index overlap for at least 15 years: a 10-year first training window plus at least 5 years out of sample. A market that falls short is reported as not run.

**Unchanged from v2:** the targets (a 5%+ drawdown within 21 trading days, and log next-month realised volatility at the 10th, 50th and 90th percentiles), the features, the walk-forward protocol (yearly expanding refit; a training row is used only if its forward window closed before the test year), and the scoring (Brier, pinball, reliability, calibration-in-the-large, 10-90 coverage, block bootstrap with 63-day blocks, 2,000 resamples, seed 7).

**Models.** The market's own implied-volatility index takes the place of VIX:

| Model | Inputs |
|---|---|
| M0 climatology | none |
| M1 implied vol | log of the market's implied-volatility index |
| M2 HAR | log trailing realised volatility over 1, 5 and 22 days |
| M3 implied vol + VRP | M1 plus the market's own variance risk premium |

No credit input: there is no free, matching credit series for these markets. The first test year for each market is the eleventh calendar year after its first common date.

**Hypotheses, tested in each market:**

- **R1 (implied vol has skill).** M1 beats M0 on both Brier score and pinball loss, with 95% intervals excluding zero.
- **R2 (VRP adds nothing).** M3 does not beat M1 on Brier score by 1% or more with an interval excluding zero. This is the v2 finding restated: R2 holds if the VRP *fails* to add.
- **R3 (implied vol beats realised vol for the volatility range).** M1 has lower pinball loss than M2, with an interval excluding zero.
- **R4 (shocks).** Descriptive only: each model's Brier skill in each market's worst calendar year for 5%+ drawdowns, with no pass/fail threshold.

**Replication rule.** The v2 headline, "implied volatility alone has skill and the variance risk premium adds nothing", counts as replicated if R1 and R2 both hold in at least three of the markets that run. Otherwise it is reported as not replicated.

## Part B: a forward ledger for the S&P 500

- `research/forecast_ledger.py` fits M0, M1, M2 and M3 for the S&P 500 on all data whose forward window has closed. It then appends, for the latest close, each model's probability of a 5%+ drawdown over the next 21 trading days and its volatility quantiles, to `research/results/forecast_ledger.csv`.
- Ledger rows contain only the date, model and forecast numbers: no raw index levels.
- Each entry is committed to the public repository on the day it is made, so the commit time proves the forecast came before the outcome.
- **Scoring:** once at least 120 forecast days have resolved, roughly six months of daily entries, the ledger is scored with the same rules, and H1 and H3 from v2 are re-tested on it.
- **No backfill:** no forecast is ever added for a day earlier than the day it is committed.

## Decision rules

- Results are reported per market and pooled as above. A market that contradicts v2 is reported as such.
- No model, input or threshold is changed on the basis of Part A results. Any change needs a v4 pre-registration.
