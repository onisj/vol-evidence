# Pre-registration v2: forecasting next-month risk as calibrated probabilities

Written and committed before any v2 data was downloaded. The questions, models, scoring rules, hypotheses and decision rules below are fixed; results will be reported against them whichever way they fall.

## Why a v2

Study 1 found that a four-input stress gauge sorts next-month risk but adds nothing beyond raw VIX. Three limits of that study motivate this one:

1. **Few crises.** Its sample started in 2008, so its most stressed days rest mostly on two episodes (2008 and 2020).
2. **Ranks instead of forecasts.** A 0 to 100 score cannot be checked for calibration: nobody can verify what a 37 means.
3. **No honest refitting.** Inputs were combined with fixed equal weights. Fitted combinations were never tested out of sample.

v2 asks a sharper question: **do forecasts built from several market inputs give better-calibrated probabilities of next-month stress than VIX alone, out of sample, over three decades?**

## Disclosure

I have already examined S&P 500 (via SPY) and VIX data from May 2008 to October 2026 in Study 1, which used a different method and different targets. To guard against that knowledge leaking into this design, every verdict is reported on two out-of-sample windows:

- **Full:** 2000 to the last day with a complete forward month.
- **Unseen:** 2000 to 2007, a period I have not looked at in either study. This is the cleanest test.

## Data

All daily, free, fetched by the script; raw series are not redistributed.

| Series | Source | Start |
|---|---|---|
| S&P 500 index level | yfinance `^GSPC` | 1990 |
| VIX | yfinance `^VIX` | 1990 |
| Moody's Baa corporate yield minus 10-year Treasury | FRED `BAA10Y` | 1986 |
| Euro Stoxx 50 and VSTOXX (second market) | yfinance `^STOXX50E` and `^V2TX` | as available |

VIX3M is not used because it starts in late 2007, too late for this sample.

## Targets

On each trading day *t*, measured on the index over the next 21 trading days:

- **Drawdown event:** 1 if the index falls 5% or more below its close on day *t* at any point in the window, else 0.
- **Next-month realised volatility:** annualised standard deviation of daily log returns over the window.

## Inputs (all known at the close of day *t*)

- `log_vix`: log of VIX.
- `log_rv_d`, `log_rv_w`, `log_rv_m`: log of trailing realised volatility over 1, 5 and 22 days (the HAR set; the 1-day value uses the absolute return).
- `vrp`: variance risk premium, (VIX/100)² minus trailing 22-day realised variance, both annualised.
- `credit`: BAA10Y level, and `credit_chg`: its change over the past 22 trading days.

## Models

Each model has a probability forecast for the drawdown event (logistic regression) and quantile forecasts of log next-month realised volatility at the 10th, 50th and 90th percentiles (linear quantile regression).

| Model | Inputs |
|---|---|
| M0 climatology | none (training-period frequency and quantiles) |
| M1 VIX | `log_vix` |
| M2 HAR | `log_rv_d`, `log_rv_w`, `log_rv_m` |
| M3 VIX + VRP | `log_vix`, `vrp` |
| M4 composite | `log_vix`, `log_rv_w`, `log_rv_m`, `vrp`, `credit`, `credit_chg` |

## Walk-forward protocol

- Expanding window. For each calendar year *Y* from 2000 onward, fit every model on all days before 1 January of *Y* whose forward window ended before that date. Then forecast every day of *Y*.
- No parameter, threshold or input choice is tuned on out-of-sample data. The first training window is 1990 to 1999.

## Scoring

- **Probabilities:**
  - Brier score.
  - Brier skill score against M0, computed as 1 − Brier(model) / Brier(M0).
  - A reliability table in ten forecast bins.
  - Calibration-in-the-large: mean forecast minus observed frequency.
- **Volatility quantiles:** pinball (quantile) loss averaged over the three quantiles, plus empirical coverage of the 10th to 90th percentile interval (target 80%).
- **Comparisons:** a paired difference in daily loss, with a 95% interval from a moving-block bootstrap over days (block length 63 trading days, 2,000 resamples, seed 7). Overlapping forward windows make daily losses autocorrelated, so a block bootstrap is required.

## Hypotheses

- **H1 (VIX has skill).** M1 beats M0: Brier skill score > 0 and lower pinball loss, both with 95% intervals excluding zero.
- **H2 (the composite beats VIX).** M4 beats M1 on both Brier score and pinball loss, each by at least 2% relative, with 95% intervals excluding zero.
- **H3 (the variance risk premium adds to VIX).** M3 beats M1 on Brier score by at least 1% relative, with a 95% interval excluding zero.
- **H4 (calibration).** M1's out-of-sample calibration-in-the-large is within ±2 percentage points, and its 10th to 90th percentile volatility interval covers between 75% and 85% of outcomes.
- **H5 (second market, secondary).** If at least ten years of Euro Stoxx 50 and VSTOXX data are available, repeat M0, M1, M2 and M3 there with the same protocol, starting the first training window at the first available date and forecasting from its eleventh year. H1 and H3 are re-tested there. Credit is not used, because no free euro-area credit series covers the period. If too little data is available, H5 is reported as not run.

## Decision rules

- **If H2 fails,** the conclusion is that VIX alone is as good as the composite for this task, and that is what the write-up says. No input is added, removed or re-weighted on the basis of these results.
- **If H1 fails,** the published stress gauge's sorting result should be read as in-sample only.
- **Any v3** requires a new pre-registration and data not used here: for example, observations after the last day of this study, or another market.
