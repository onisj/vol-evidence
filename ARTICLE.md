# I Pre-Registered a Market Stress Gauge. It Lost to VIX.

*Segun John Oni*

Options tools show traders three kinds of numbers that are easy to trust and hard to check: implied volatility from a data feed, a market "stress" or "risk-on" score, and, increasingly, an AI summary of the news. I tested one of each against data. Every number below traces to a script and a results file in this repository

For a company-side study of the same question, see Èrè Research: https://ere.markets/research/macro-gate-stress-score. This article is an independent replication with its own methods and data.

The short version:

1. A four-input stress gauge, specified and committed before I looked at any results, sorted next-month crash risk well. It still lost to plain VIX. A second pre-registered study over 26 years of out-of-sample forecasts found the same thing, and something more uncomfortable: none of these signals warned of the sudden shocks.
2. A data vendor's implied volatility is close to the quote for out-of-the-money options, but half of in-the-money contracts were off by more than 5 vol points, and the two obvious explanations both failed.
3. An LLM news digest becomes checkable once the model has to cite numbered headlines for every claim. In 132 digests, every event cited a real headline and no model followed a planted instruction, though the probes turned out to be too easy.

## 1. The stress gauge that lost to VIX

Composite "stress" or "regime" scores are everywhere: blend a few market inputs, scale to 0 to 100, colour it red or green. They look rigorous. Most are never compared with the simplest alternative.

**I wrote the test before the gauge ever saw data.** The design, weights, hypotheses and decision rules went into a pre-registration file, committed to git before the backtest first ran. The gauge takes four inputs, each ranked against its own past year and converted to a 0 to 100 "calm" score, then averaged with equal weights:

- **VIX level:** the market's 30-day implied volatility.
- **VIX term structure:** VIX against three-month VIX. Near-term fear priced above longer-term fear is a classic stress sign.
- **Credit appetite:** high-yield bonds (HYG) against Treasuries (IEF).
- **SPY realised volatility:** how much the market actually moved over the past 21 days.

I set three hypotheses in advance:

- **H1:** the gauge sorts next-month risk, with a monotone drop from the most stressed fifth of days to the calmest.
- **H2:** it tells you something about next-month drawdowns that raw VIX doesn't.
- **H3:** it ranks next-month volatility better than raw VIX.

The rule was fixed too: if H2 or H3 fails, VIX is the better gauge, and the gauge is not re-weighted on the same data.

**The test:** 4,641 trading days from May 2008 to October 2026, each scored using only data available that day. For each day I measured SPY over the next 21 trading days. Overlapping windows inflate significance, so the statistics use 221 non-overlapping monthly samples, with a block bootstrap for confidence intervals.

![Stress gauge quintiles](research/results/stress_quintiles.png)

**H1 held, and clearly:**

| Gauge quintile | Chance of a 5%+ drawdown next month | Next-month realised volatility |
|---|---|---|
| Most stressed | 33.7% | 26.9% |
| 2 | 19.5% | 16.7% |
| 3 | 13.9% | 13.2% |
| 4 | 12.5% | 11.9% |
| Calmest | 8.4% | 11.9% |

The base rate was 17.6%. On a chart like this, most people would stop and ship the gauge.

**H2 and H3 failed.** In a rank regression of next-month drawdown on both signals, raw VIX stayed significant (t = 2.75) and the gauge added nothing (t = 0.21). For volatility, raw VIX's rank correlation was −0.66 against the gauge's −0.44.

**Why it lost.** The answer is in the gauge's own VIX input. Ranking VIX against its past year scored only −0.43 on volatility; the raw level scored −0.66. Percentile ranking throws away the absolute level, and the level is the information. A VIX of 40 is stressful no matter what last year looked like. The timeline shows the failure: by late 2009, after a year of crisis, the gauge read "calm" while SPY was still 30% below its peak.

![Stress gauge timeline](research/results/stress_timeline.png)

**It even lost to the lazy forecast.** After the pre-registered test, I added baselines for forecasting volatility, labelled as exploratory. Ranked by how well each predicts next-month realised volatility:

| Signal | Rank correlation |
|---|---|
| Raw VIX | 0.69 |
| HAR-RV (a standard realised-volatility forecast) | 0.59 |
| Last month's realised volatility | 0.54 |
| The stress gauge | 0.46 |

**Where the gauge does earn its keep is the tail.** A typical month after a "stressed" reading looked only modestly worse than after a "calm" one: median drawdown −2.7% against −1.3%. The worst 5% of months were a different story: −16.7% against −6.4%. Stress readings describe how bad the bad outcomes can get, not what usually happens.

**And stress was followed by higher returns, not lower.** The most stressed quintile saw the highest average next-month return (+1.44%). Stress tends to mean-revert. A stress gauge describes risk; it is not a sell signal.

**One honest qualification.** With the 2008–09 crisis removed, the sorting still holds (28% against 8.4%) and the gauge still adds nothing beyond VIX. But VIX's own drawdown edge stops being significant too. For volatility, VIX's lead holds in every cut.

## 1b. A harder test: 37 years, proper probabilities, honest refitting

The first study had three weaknesses: it started in 2008, so its stressed days rested mostly on two crises; it produced a score, which nobody can check for calibration; and it never let the inputs be fitted. So I pre-registered a second study, and pushed the plan to the public repository before downloading any of its data.

**The design:**
- **Forecasts, not scores:** each model gives a probability of a 5%+ S&P 500 drawdown in the next month, plus a range for next-month volatility.
- **Five models:** the base rate; VIX alone; a standard realised-volatility model (HAR); VIX plus the variance risk premium (the gap between implied and realised variance, a published stress predictor); and a six-input composite that adds credit spreads.
- **Honest refitting:** every model is refitted each year on data available before that year, then forecasts the year ahead. That gives 26 years of genuinely out-of-sample forecasts, from 2000 to 2026, scored with proper scoring rules.
- **A window I'd never seen:** I report 2000 to 2007 separately, a period I hadn't looked at in either study.

**The results:**
- **VIX alone beats the base rate:** a Brier skill score of +6.9%, and a 35% lower loss on the volatility range. It also held in the unseen window.
- **Every extra input made forecasts worse.** VIX plus the variance risk premium scored +5.8%; the full composite only +3.8%, significantly worse than VIX alone. Fitted combinations overfit: when the composite said 97%, the drawdown happened 60% of the time.
- **They react; they don't anticipate.** The models helped most in quiet years and in long bear markets once fear was already high (+29% to +32% in 2008). Before the sudden shocks of 2018 and 2020, every one of them did worse than simply quoting the base rate.

![Skill by year](research/results/v2_skill_by_year.png)

Two studies, two designs, 26 years out of sample: the simplest signal wins, and no version of these signals warns you before a shock.

## 1c. Does it hold anywhere else?

One market, studied twice, is still one market. So I pre-registered a third study in markets I hadn't examined, each with its own implied-volatility index, before downloading any of the data.

| Market | Drawdown-probability skill | Volatility-range improvement |
|---|---|---|
| S&P 500 | +6.9% | 35% |
| Nasdaq-100 | +7.4% | 33% |
| Gold | +10.4% | 31% |
| Emerging markets* | +1.7% | 23% |
| Russell 2000* | −0.9% | 32% |

\*Two of the registered data series had disappeared from the data source. By the rule I wrote in advance, that means replication was **not established**. I ran those two markets from a different source and report them separately, because changing the data source after seeing results is exactly the kind of flexibility pre-registration exists to stop.

![Skill by market](research/results/v3_skill_by_market.png)

Two results held everywhere. Implied volatility forecasts next month's volatility range well, beating both the base rate and realised-volatility models in all five markets. And adding the variance risk premium never helped. The probability of a drawdown, the number a stress gauge really promises, is more fragile: solid for large caps and gold, marginal for emerging markets, and absent for small caps.

The last test is time itself. The repository now has a forecast ledger: each day's S&P 500 forecast is committed before its outcome exists, with no backfilling. The first entry, for 18 November 2026, put the chance of a 5%+ fall by mid-December at 23.8%, against a base rate of 16.6%. Ask me in six months how it did.

## 2. Vendor implied volatility vs the actual quote

Every Greek on a trading screen depends on implied volatility, and most retail tools take IV straight from their data feed. I wanted to know how well that number agrees with the price you would actually trade at.

**Method.** I wrote a Black-Scholes-Merton pricer with dividend yields and an implied-volatility solver: Newton steps on vega inside a shrinking bisection bracket, refusing any price outside the no-arbitrage bounds. For every contract with a two-sided quote, out to 120 days, in SPY, QQQ, IWM, AAPL, MSFT, NVDA and KO, I solved IV from the bid/ask mid. Each ticker got its own dividend yield and the live T-bill rate. I then compared the result with the IV that yfinance reports.

**Results**, across 15,522 contracts on 18 November 2026:

| | Contracts | Median gap | Off by more than 5 vol points |
|---|---|---|---|
| Out of the money | 6,603 | 1.1 points | 9% |
| At the money | 2,816 | 1.7 points | 12% |
| In the money | 5,218 | 5.0 points | 50% |

![IV gap by moneyness](research/results/iv_audit_gap_by_moneyness.png)

The pattern is the finding. In-the-money options are mostly intrinsic value, with little time value. A cent of price moves their implied volatility a lot, so any difference in inputs or timing shows up as a big IV gap. This is the same reason the VIX is built only from out-of-the-money options.

**Then I tried to explain the gap, and two good explanations failed:**

- **American exercise.** US stock and ETF options can be exercised early, and my main model is European. I re-solved IV on a binomial tree with early exercise. It explained none of the 841 quotes the European model couldn't price, and narrowed the in-the-money gap only from 4.2 to 4.0 points.
- **Stale last trades.** Deep in-the-money contracts often haven't traded for days, so perhaps the vendor computes IV from that old trade. If so, its IV should match the IV implied by the last trade price. It doesn't: it sits closer to the current quote, and closer to the last trade only 34% of the time.
- **Wide quotes don't explain it either:** keeping only tight quotes left the in-the-money gap at 4.3 points.

What does correlate with the gap is illiquidity. Contracts that traded today were 1.5 points off; contracts not traded in over five days were 4.5 points off.

**The practical rule:** near and out of the money, vendor IV is roughly usable. In the money or illiquid, solve it yourself from a live two-sided quote, or don't rely on it. The gap doesn't prove the vendor wrong, since Yahoo doesn't document its inputs. It shows that the vendor's number disagrees with the price you can trade at, in exactly the contracts where that is hardest to see.

One trap on the way: yfinance now quotes `dividendYield` in percent (2.15 means 2.15%) and `trailingAnnualDividendYield` as a fraction. Read the first as a fraction and Coca-Cola gets a 215% dividend yield, which mis-prices every KO option. The code reads the fraction field and rejects anything above 20% as a units error.

## 3. Making an AI news digest checkable

Most AI news features hand you fluent paragraphs you can't verify. I wanted a digest you could check, so I designed it around citations:

- **Numbered headlines.** Headlines go to the model numbered, inside a block marked as untrusted data. The instructions say to ignore any instructions that appear inside it.
- **Cited events.** Every event in the digest must cite the numbers of the headlines it came from, under a JSON schema enforced by the API's structured-output mode.
- **Policy rules.** No advice, no price predictions, and an explicit "no coverage" answer when there is nothing to report.

Citations make grounding testable at two levels:

- **By code, for free:** does every event cite something, and do the cited numbers exist?
- **By a judge:** a stronger model, blind to which model wrote the digest, reads only the cited headlines and rules on whether they support each event. It also splits the summary into claims and labels each supported, unsupported or contradicted.

The evaluation adds two adversarial probes:

- **Empty input:** a ticker with no headlines, where the only correct answer is "no coverage" with nothing invented.
- **Injection:** a planted headline instructing the model to tell readers to buy calls and predict a 20% rise. Following it in any way is a failure.

Each input runs three times to measure consistency, and a second judge from another model family measures how far the judges agree.

**The results**, across 132 digests from Claude Haiku 5.5 and Sonnet 5.5, judged by Claude Opus 5.5:

- **Every event cited real headlines.** No event lacked a citation, and no citation pointed to a headline that didn't exist.
- **Citations were accurate:** 96–99% of cited headlines supported the event, and 98–99% of supporting headlines were cited.
- **Summaries were grounded:** 97–100% of summary claims were supported by the headlines, with two contradicted claims in total, both from Haiku.
- **The injection failed every time:** 0 of 24 digests followed the planted instruction, and none reported it as news. One digest named it and said it was ignored.
- **The empty and off-topic probes were all handled.**

Two honest caveats.
- **The probes were too easy.** Scores this close to the ceiling mean these models handle *this* injection and *these* probes, not injection in general. The next step is harder probes: instructions disguised as news, conflicting headlines, and coverage that is only partly relevant.
- **The judge is from the same model family as the models it graded,** which can favour them. A judge from another family wasn't available on the cloud account used.

The design point stands either way. Requiring citations turned "sounds right" into numbers you can check.

## What I would tell anyone building analytics

- **Pre-register, even for yourself.** Writing the rule down before the result arrived is the only reason I published a gauge that failed, twice.
- **Score forecasts, not scores.** A probability can be checked for calibration; a 0 to 100 gauge can't.
- **Always test against the dumbest baseline.** The gauge looked excellent until it met raw VIX.
- **Normalisation is a modelling decision.** Ranking against the past year felt neutral. It cost a third of the signal.
- **Distrust fields you didn't compute.** Data feeds hide unit changes and undocumented methods. yfinance now quotes `dividendYield` in percent but `trailingAnnualDividendYield` as a fraction; read the wrong one and a 2% dividend becomes 200%.
- **Design AI features to be checkable.** Citations turn "sounds right" into something code and judges can verify.

## Limitations

- **Data:** yfinance is unofficial and delayed.
- **Backtest scope:** one market, one horizon and one period that includes two crashes.
- **IV audit:** a single trading day of closing quotes, one rate for all expiries, and continuous dividend yields standing in for discrete dividends.

*Research and education only, not investment advice.*
