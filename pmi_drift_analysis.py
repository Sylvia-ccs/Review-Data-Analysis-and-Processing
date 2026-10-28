#!/usr/bin/env python3
"""PMI drift analysis for target aspect term."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from trend_analysis import analyze_trends
except ImportError:
    analyze_trends = None

DEFAULT_PMI_CSV  = "outputs/trends/pmi_associations_by_time.csv"
DEFAULT_REVIEWS  = "yelp_academic_dataset_review.json"
DEFAULT_BUSINESS = "yelp_academic_dataset_business.json"
DEFAULT_TARGET   = "price"
DEFAULT_OUT      = "outputs/pmi_drift"
DEFAULT_TOP_N    = 20

def _pmi_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("time", "word", "target"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "rank" in df.columns:
        df["rank"] = df["rank"].astype(int)
    if "pmi" in df.columns:
        df["pmi"] = df["pmi"].astype(float)
    return df

def _pmi_from_raw(reviews: str, business: str, target: str, time_unit: str = "year",
                  max_reviews: int | None = None) -> pd.DataFrame:
    if analyze_trends is None:
        raise RuntimeError("trend_analysis.py must be importable.")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        analyze_trends(reviews_path=reviews, business_path=business, output_dir=tmp,
                       max_reviews=max_reviews, target=target, top_n=500,
                       min_cooc=5, time_unit=time_unit)
        return _pmi_from_csv(Path(tmp) / "pmi_associations_by_time.csv")

def _spearman_rho(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 2: return float("nan")
    a = np.asarray([p[0] for p in pairs], dtype=float)
    b = np.asarray([p[1] for p in pairs], dtype=float)
    if a.std() == 0 or b.std() == 0: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def year_overlap(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    years = sorted(df["time"].unique())
    top_sets = {y: set(df.loc[df["time"] == y].nsmallest(top_n, "rank")["word"]) for y in years}
    rank_maps = {y: dict(zip(df.loc[df["time"] == y].nsmallest(top_n, "rank")["word"],
                             df.loc[df["time"] == y].nsmallest(top_n, "rank")["rank"])) for y in years}

    rows = []
    for y1, y2 in zip(years[:-1], years[1:]):
        s1, s2 = top_sets[y1], top_sets[y2]
        inter = s1 & s2; union = s1 | s2
        jaccard = len(inter) / len(union) if union else 0.0
        pairs = [(rank_maps[y1][w], rank_maps[y2][w]) for w in inter]
        rho = _spearman_rho(pairs)
        rows.append({
            "year_from": y1, "year_to": y2, "top_n": top_n,
            "|S_y1|": len(s1), "|S_y2|": len(s2), "overlap": len(inter),
            "jaccard": round(jaccard, 4),
            "rank_corr_on_overlap": round(rho, 4) if rho == rho else None,
            "new_in_y2": sorted(s2 - s1), "dropped_from_y1": sorted(s1 - s2),
        })
    return pd.DataFrame(rows)

def word_lifecycle(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    years = sorted(df["time"].unique())
    top = df[df["rank"] <= top_n]
    rows = []
    for word, g in top.groupby("word"):
        ys = sorted(g["time"].unique())
        if not ys: continue
        span = len(ys); pmi_vals = g.set_index("time")["pmi"]
        rows.append({
            "word": word, "first_year": ys[0], "last_year": ys[-1],
            "years_present": span, "coverage_pct": round(span / len(years), 4),
            "years_list": ",".join(ys), "min_rank_ever": int(g["rank"].min()),
            "mean_rank": round(float(g["rank"].mean()), 2),
            "mean_pmi": round(float(pmi_vals.mean()), 4),
            "max_pmi": round(float(pmi_vals.max()), 4),
            "min_pmi": round(float(pmi_vals.min()), 4),
        })
    return pd.DataFrame(rows).sort_values(["years_present", "mean_rank"], ascending=[False, True]).reset_index(drop=True)

def top_movers(df: pd.DataFrame, min_years: int = 3, k: int = 15) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["time"].str.extract(r"(\d{4})").astype(int)
    rows = []
    for word, g in df.groupby("word"):
        g = g.sort_values("year")
        if len(g) < min_years: continue
        x = g["year"].to_numpy(dtype=float); y = g["pmi"].to_numpy(dtype=float)
        if np.allclose(x, x[0]): continue
        slope = float(np.polyfit(x - x.mean(), y, 1)[0])
        rows.append({
            "word": word, "n_years": len(g), "slope_pmi_per_year": round(slope, 4),
            "pmi_first": round(float(y[0]), 4), "pmi_last": round(float(y[-1]), 4),
            "delta_pmi": round(float(y[-1] - y[0]), 4),
            "first_year": int(x[0]), "last_year": int(x[-1]),
        })
    movers = pd.DataFrame(rows)
    if movers.empty: return movers
    rising = movers.nlargest(k, "slope_pmi_per_year")
    falling = movers.nsmallest(k, "slope_pmi_per_year")
    return pd.concat([rising, falling]).drop_duplicates("word").reset_index(drop=True)

def plot_heatmap(df: pd.DataFrame, out_png: Path, top_n: int) -> None:
    years = sorted(df["time"].unique())
    top_words = set()
    for y in years: top_words |= set(df.loc[df["time"] == y].nsmallest(top_n, "rank")["word"])
    if not top_words: return

    wide = df[df["word"].isin(top_words)].pivot_table(
        index="word", columns="time", values="pmi", aggfunc="mean"
    ).reindex(columns=years)
    order = wide.mean(axis=1).sort_values(ascending=False).index
    wide = wide.loc[order]

    if wide.empty: return
    fig_h = max(4.5, 0.28 * len(wide) + 1.5)
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(years) + 3), fig_h))
    im = ax.imshow(wide.values, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_facecolor("white"); ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_yticks(range(len(wide))); ax.set_yticklabels(wide.index, fontsize=8)
    ax.set_title(f"PMI(word | price) by year — top {top_n} per year (union)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="PMI (log2)")
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

def plot_overlap(overlap: pd.DataFrame, out_png: Path) -> None:
    if overlap.empty: return
    labels = [f"{a}→{b}" for a, b in zip(overlap["year_from"], overlap["year_to"])]
    rho_vals = overlap["rank_corr_on_overlap"].astype(float).fillna(0)
    
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.plot(labels, overlap["jaccard"], marker="o", color="C0", label="Jaccard")
    ax1.set_ylabel("Jaccard overlap", color="C0"); ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_ylim(0, 1.05); ax1.set_xlabel("Consecutive year pair")
    ax1.grid(True, alpha=0.3); ax1.set_title("Year-to-year stability of top-N PMI words")
    
    ax2 = ax1.twinx()
    ax2.plot(labels, rho_vals, marker="s", color="C3", label="Spearman ρ")
    ax2.set_ylabel("Spearman ρ", color="C3"); ax2.tick_params(axis="y", labelcolor="C3")
    ax2.set_ylim(-1.05, 1.05)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

def plot_movers(movers: pd.DataFrame, out_png: Path, k: int) -> None:
    if movers.empty: return
    rising = movers[movers["slope_pmi_per_year"] > 0].nlargest(k, "slope_pmi_per_year").sort_values("slope_pmi_per_year")
    falling = movers[movers["slope_pmi_per_year"] < 0].nsmallest(k, "slope_pmi_per_year").sort_values("slope_pmi_per_year")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.32 * k + 1.5)))
    if not rising.empty: axes[0].barh(rising["word"], rising["slope_pmi_per_year"], color="C2")
    else: axes[0].text(0.5, 0.5, "No rising words\n(data too sparse)", ha="center", va="center")
    axes[0].set_title("Top rising words (PMI slope)"); axes[0].axvline(0, color="k", lw=1, alpha=0.5)
    
    if not falling.empty: axes[1].barh(falling["word"], falling["slope_pmi_per_year"], color="C3")
    else: axes[1].text(0.5, 0.5, "No falling words\n(data too sparse)", ha="center", va="center")
    axes[1].set_title("Top falling words (PMI slope)"); axes[1].axvline(0, color="k", lw=1, alpha=0.5)
    
    for ax in axes: ax.grid(True, alpha=0.25, axis="x")
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

def build_summary(df: pd.DataFrame, overlap: pd.DataFrame, lifecycle: pd.DataFrame, movers: pd.DataFrame, target: str) -> dict:
    years = sorted(df["time"].unique())
    mean_jaccard = float(overlap["jaccard"].mean()) if not overlap.empty else None
    mean_rho = float(overlap["rank_corr_on_overlap"].dropna().mean()) if not overlap.empty and not overlap["rank_corr_on_overlap"].dropna().empty else None

    stable_words = lifecycle.head(10)[["word", "years_present", "coverage_pct"]].to_dict("records") if not lifecycle.empty else []
    rising = movers[movers["slope_pmi_per_year"] > 0].nlargest(10, "slope_pmi_per_year")[["word", "slope_pmi_per_year"]].to_dict("records") if not movers.empty else []
    falling = movers[movers["slope_pmi_per_year"] < 0].nsmallest(10, "slope_pmi_per_year")[["word", "slope_pmi_per_year"]].to_dict("records") if not movers.empty else []

    return {
        "target": target, "years_analysed": years, "n_years": len(years),
        "mean_jaccard": round(mean_jaccard, 4) if mean_jaccard is not None else None,
        "mean_spearman": round(mean_rho, 4) if mean_rho is not None else None,
        "most_persistent_words": stable_words,
        "top_rising_words": rising, "top_falling_words": falling,
    }

def run(pmi_csv: str | None, reviews: str | None, business: str | None, target: str,
        out_dir: str, top_n: int, max_reviews: int | None) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    if pmi_csv and Path(pmi_csv).exists():
        print(f"[load] {pmi_csv}")
        df = _pmi_from_csv(pmi_csv)
    elif reviews and business:
        print(f"[run ] recomputing PMI for target='{target}' from raw files")
        df = _pmi_from_raw(reviews, business, target, time_unit="year", max_reviews=max_reviews)
    else:
        raise SystemExit("Provide --pmi-csv OR (--reviews and --business).")

    df = df[df["target"] == target].copy() if "target" in df.columns else df
    if df.empty:
        raise SystemExit(f"No PMI rows for target='{target}'.")
    
    df = df.sort_values(["time", "rank"]).reset_index(drop=True)

    overlap = year_overlap(df, top_n=top_n)
    lifecycle = word_lifecycle(df, top_n=top_n)
    movers = top_movers(df, min_years=3, k=15)

    overlap.to_csv(out / "year_overlap.csv", index=False)
    lifecycle.to_csv(out / "word_lifecycle.csv", index=False)
    movers.to_csv(out / "top_movers.csv", index=False)

    plot_heatmap(df, out / "pmi_heatmap.png", top_n=top_n)
    plot_overlap(overlap, out / "overlap_over_time.png")
    plot_movers(movers, out / "top_movers.png", k=15)

    summary = build_summary(df, overlap, lifecycle, movers, target)
    (out / "drift_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

# =============================================================================
#     # ----- Did the words change from year to year? -----
#     print(f"\n{'='*70}")
#     print(f"Semantic Drift Analysis for '{target.upper()}'")
#     print(f"{'='*70}")
#     print(f"Mean Jaccard overlap (year-to-year stability): {summary['mean_jaccard']:.4f}")
#     print("  (1.0 = identical sets, 0.0 = completely different words each year)\n")
#     
#     print("Year-by-year changes (Top words to enter and leave):")
#     for _, row in overlap.iterrows():
#         print(f"  {row['year_from']} -> {row['year_to']}:")
#         print(f"    Jaccard: {row['jaccard']:.4f}")
#         if row['new_in_y2']:
#             print(f"    New in {row['year_to']}: {', '.join(row['new_in_y2'][:5])}{'...' if len(row['new_in_y2']) > 5 else ''}")
#         if row['dropped_from_y1']:
#             print(f"    Dropped from {row['year_from']}: {', '.join(row['dropped_from_y1'][:5])}{'...' if len(row['dropped_from_y1']) > 5 else ''}")
#         print()
# 
#     if summary["most_persistent_words"]:
#         print("Most Persistent Words (stable across years):")
#         for rec in summary["most_persistent_words"][:5]:
#             print(f"  - {rec['word']} (present in {rec['years_present']} years)")
#         print()
# 
#     if summary["top_rising_words"]:
#         print(f"Words becoming MORE associated with '{target}':")
#         for rec in summary["top_rising_words"][:5]:
#             print(f"  - {rec['word']} (PMI slope: {rec['slope_pmi_per_year']})")
#         print()
# 
#     if summary["top_falling_words"]:
#         print(f"Words becoming LESS associated with '{target}':")
#         for rec in summary["top_falling_words"][:5]:
#             print(f"  - {rec['word']} (PMI slope: {rec['slope_pmi_per_year']})")
#             
#     print(f"\nArtifacts written to {out.resolve()}")
# =============================================================================
    return summary

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmi-csv", default=None)
    ap.add_argument("--reviews", default=DEFAULT_REVIEWS)
    ap.add_argument("--business", default=DEFAULT_BUSINESS)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--max-reviews", type=int, default=None)
    args = ap.parse_args()

    pmi_csv = args.pmi_csv
    if pmi_csv is None and Path(DEFAULT_PMI_CSV).exists():
        pmi_csv = DEFAULT_PMI_CSV

    run(pmi_csv=pmi_csv, reviews=args.reviews, business=args.business,
        target=args.target, out_dir=args.out, top_n=args.top_n, max_reviews=args.max_reviews)

if __name__ == "__main__":
    main()