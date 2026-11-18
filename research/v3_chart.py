"""Chart of out-of-sample skill across all markets studied (v2, v3 and the v3 deviation)."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "results"


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v2 = json.loads((OUT / "v2_summary.json").read_text())["us"]["full"]
    v3 = json.loads((OUT / "v3_summary.json").read_text())["markets"]
    dev = json.loads((OUT / "v3_deviation_summary.json").read_text())["markets"]

    rows = [("S&P 500\n(v2)", v2["brier_skill_vs_climatology"]["M1 VIX"],
             v2["M1_vs_M0"]["pinball"]["improvement_pct"], False)]
    for name, r in v3.items():
        if "models" in r:
            rows.append((name, r["brier_skill_vs_climatology"]["M1 implied vol"],
                         r["M1_vs_M0"]["pinball"]["improvement_pct"], False))
    for name, r in dev.items():
        if "models" in r:
            rows.append((f"{name}\n(deviation)", r["brier_skill_vs_climatology"]["M1 implied vol"],
                         r["M1_vs_M0"]["pinball"]["improvement_pct"], True))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = [r[0] for r in rows]
    for ax, vals, title, color in (
        (axes[0], [100 * r[1] for r in rows], "Drawdown probability:\nBrier skill vs base rate (%)", "#b5523b"),
        (axes[1], [r[2] for r in rows], "Volatility range:\nquantile loss reduction vs base rate (%)", "#3b6ea5"),
    ):
        bars = ax.bar(range(len(rows)), vals, color=color)
        for bar, row in zip(bars, rows):
            if row[3]:
                bar.set_hatch("//")
                bar.set_alpha(0.6)
        ax.axhline(0, color="#999", lw=1)
        ax.set_xticks(range(len(rows)), labels, fontsize=8)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Implied volatility alone, out of sample (hatched: data source deviates from pre-registration)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "v3_skill_by_market.png", dpi=160)


if __name__ == "__main__":
    main()
