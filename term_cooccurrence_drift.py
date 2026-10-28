#!/usr/bin/env python3
"""
Term Co-occurrence Drift (PMI) for a target aspect term.
Reads Yelp JSON, computes PMI with rare-word filtering, and outputs all drift artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re as std_re
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import orjson
    def _json_loads(b: bytes | str) -> Any:
        return orjson.loads(b)
except ImportError:
    def _json_loads(b: bytes | str) -> Any:
        return json.loads(b)

try:
    import regex as uregex
except ImportError:
    uregex = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ==========================================================================
# ▼▼▼ DEFAULT CONFIG ▼▼▼
# ==========================================================================
DEFAULT_REVIEWS      = "yelp_academic_dataset_review.json"
DEFAULT_OUT          = "outputs/filtered_pmi_drift"
DEFAULT_TARGET       = "price"
DEFAULT_TOP_N        = 20
DEFAULT_MIN_WORD_DF  = 20      # Filter out words occurring < 50 times per year
DEFAULT_MIN_COOC     = 10      # Minimum co-occurrence count
DEFAULT_WORKERS      = None
DEFAULT_CHUNK_SIZE   = 20_000
# ==========================================================================

if uregex:
    _TOKEN_RE = uregex.compile(
        r"""(?ix)
        (?:https?://|www\.)[^\s]+
        | [#@][\p{L}\p{N}_]+
        | [\p{L}\p{M}]+(?:['’][\p{L}\p{M}]+)*
        | \d+(?:[.,]\d+)*
        | \p{Regional_Indicator}{2}
        | (?:\p{Extended_Pictographic}|\p{Emoji_Presentation})
          (?:\uFE0F|\uFE0E)?
          (?:\u200D(?:\p{Extended_Pictographic}|\p{Emoji_Presentation})(?:\uFE0F|\uFE0E)?)*
        | (?::|;|=|8|x|X)(?:-|\^)?(?:\)|\(|D|P|p|/|\\) 
        """,
        flags=uregex.VERSION1,
    )
else:
    _TOKEN_RE = std_re.compile(
        r"https?://\S+|www\.\S+|[#@][A-Za-z0-9_]+|[A-Za-z]+(?:['’][A-Za-z]+)*"
        r"|\d+(?:[.,]\d+)*|[^\w\s]",
        flags=std_re.UNICODE | std_re.IGNORECASE,
    )

_URL_RE = std_re.compile(r"^(?:https?://|www\.)", std_re.IGNORECASE)

def custom_tokenize(text: str) -> list[str]:
    if not text: return []
    out = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if _URL_RE.match(raw): out.append("<url>")
        else: out.append(raw)
    return out

def _chunked_lines(path: str | Path, limit: int | None, chunk_size: int) -> Iterator[list[bytes]]:
    buf = []
    count = 0
    with open(path, "rb") as f:
        for line in f:
            if not line.strip(): continue
            buf.append(line)
            count += 1
            if len(buf) >= chunk_size:
                yield buf
                buf = []
            if limit is not None and count >= limit:
                if buf: yield buf
                return
    if buf: yield buf

def _parse_year(d: Any) -> str | None:
    if not isinstance(d, str) or len(d) < 4: return None
    return d[:4]

STOPWORDS = set("""a an the and or but if then than to of for from in on at by with as is are was were be been being it its this that these those i me my we our you your he she they them their us not no do does did have has had am can could should would will just very really so too only also there here what when where who why how all any some many much more most less other another into out up down over under again about after before between while because during without within""".split())

_W_target = None

def _init_worker(tgt):
    global _W_target
    _W_target = tgt

def _process_chunk(lines: list[bytes]):
    docs = Counter(); word_docs = defaultdict(Counter); target_docs = Counter(); cooc = defaultdict(Counter)

    for raw in lines:
        try: row = _json_loads(raw)
        except Exception: continue
        time_key = _parse_year(row.get("date"))
        if not time_key: continue
        text = row.get("text") or ""
        tokens = custom_tokenize(text)
        
        token_set = set(tokens)
        docs[time_key] += 1
        word_docs[time_key].update(token_set)
        if _W_target in token_set:
            target_docs[time_key] += 1
            cooc[time_key].update(token_set - {_W_target})

    return docs, dict(word_docs), target_docs, dict(cooc)

def _wrap_progress(it, total, desc):
    return tqdm(it, total=total, desc=desc, unit="chunk") if tqdm else it

def _spearman_rho(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 2: return float("nan")
    a = np.asarray([p[0] for p in pairs], dtype=float)
    b = np.asarray([p[1] for p in pairs], dtype=float)
    if a.std() == 0 or b.std() == 0: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ==========================================================================
# Main Analysis Function
# ==========================================================================
def calculate_year_overlap(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
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

def calculate_word_lifecycle(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
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

def calculate_top_movers(df: pd.DataFrame, min_years: int = 3, k: int = 15) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["time"].astype(str).str.extract(r"(\d{4})").astype(int)
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


# ==========================================================================
# Plotting Function
# ==========================================================================
def plot_heatmap(df: pd.DataFrame, out_png: Path, target: str, top_n: int) -> None:
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
    ax.set_title(f"Filtered PMI(word | {target}) by year — top {top_n} per year")
    fig.colorbar(im, ax=ax, shrink=0.6, label="PMI (log2)")
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

def plot_overlap(overlap: pd.DataFrame, out_png: Path, target: str) -> None:
    if overlap.empty: return
    labels = [f"{a}→{b}" for a, b in zip(overlap["year_from"], overlap["year_to"])]
    rho_vals = overlap["rank_corr_on_overlap"].astype(float).fillna(0)
    
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.plot(labels, overlap["jaccard"], marker="o", color="C0", label="Jaccard")
    ax1.set_ylabel("Jaccard overlap", color="C0"); ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_ylim(0, 1.05); ax1.set_xlabel("Year pair")
    ax1.grid(True, alpha=0.3); ax1.set_title(f"Year-to-year stability of top words for '{target}'")
    
    ax2 = ax1.twinx()
    ax2.plot(labels, rho_vals, marker="s", color="C3", label="Spearman ρ")
    ax2.set_ylabel("Spearman ρ", color="C3"); ax2.tick_params(axis="y", labelcolor="C3")
    ax2.set_ylim(-0.05, 1.05)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

def plot_movers(movers: pd.DataFrame, out_png: Path, k: int = 15) -> None:
    if movers.empty: return
    rising = movers[movers["slope_pmi_per_year"] > 0].nlargest(k, "slope_pmi_per_year").sort_values("slope_pmi_per_year")
    falling = movers[movers["slope_pmi_per_year"] < 0].nsmallest(k, "slope_pmi_per_year").sort_values("slope_pmi_per_year")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.32 * k + 1.5)))
    if not rising.empty: axes[0].barh(rising["word"], rising["slope_pmi_per_year"], color="C2")
    else: axes[0].text(0.5, 0.5, "No rising words\n(data too sparse)", ha="center", va="center")
    axes[0].set_title("Top rising words (Filtered PMI slope)"); axes[0].axvline(0, color="k", lw=1, alpha=0.5)
    
    if not falling.empty: axes[1].barh(falling["word"], falling["slope_pmi_per_year"], color="C3")
    else: axes[1].text(0.5, 0.5, "No falling words\n(data too sparse)", ha="center", va="center")
    axes[1].set_title("Top falling words (Filtered PMI slope)"); axes[1].axvline(0, color="k", lw=1, alpha=0.5)
    
    for ax in axes: ax.grid(True, alpha=0.25, axis="x")
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)


# ==========================================================================
# Main Pipeline
# ==========================================================================
def run_drift_analysis(reviews_path, target, out_dir, top_n=20, min_word_df=50, 
                       min_cooc=10, max_reviews=None, workers=None, chunk_size=20000):
    if workers is None: workers = min(os.cpu_count() or 1, 8)
    target_token = custom_tokenize(target)[0]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # 1. Extracting raw counts using multiple processes
    print(f"Target: '{target_token}' | Min Word DF: {min_word_df} | Workers: {workers}")
    agg_docs = Counter()
    agg_word_docs = defaultdict(Counter)
    agg_target_docs = Counter()
    agg_cooc = defaultdict(Counter)

    t0 = time.time()
    total_chunks = max(1, (max_reviews + chunk_size - 1) // chunk_size) if max_reviews else None

    if workers <= 1:
        _init_worker(target_token)
        for chunk in _wrap_progress(_chunked_lines(reviews_path, max_reviews, chunk_size), total_chunks, "Processing"):
            d, wd, td, c = _process_chunk(chunk)
            agg_docs.update(d)
            for k, v in wd.items(): agg_word_docs[k].update(v)
            agg_target_docs.update(td)
            for k, v in c.items(): agg_cooc[k].update(v)
    else:
        with Pool(processes=workers, initializer=_init_worker, initargs=(target_token,)) as pool:
            res = pool.imap_unordered(_process_chunk, _chunked_lines(reviews_path, max_reviews, chunk_size), chunksize=1)
            for d, wd, td, c in _wrap_progress(res, total_chunks, f"Processing"):
                agg_docs.update(d)
                for k, v in wd.items(): agg_word_docs[k].update(v)
                agg_target_docs.update(td)
                for k, v in c.items(): agg_cooc[k].update(v)

    print(f"[done] Extraction finished in {time.time()-t0:.1f}s")

    # 2. Calculate PMI and filter out low-frequency words in a DataFrame
    rows = []
    for t_k in sorted(agg_docs):
        tot = agg_docs[t_k]
        tgt_df = agg_target_docs[t_k]
        if not tot or not tgt_df: continue
        
        for w, c in agg_cooc[t_k].items():
            if w == target_token or w in STOPWORDS or c < min_cooc: continue
            wdf = agg_word_docs[t_k][w]
            
            # 【关键】过滤低频词
            if wdf < min_word_df: continue
            
            pmi = math.log2((c * tot) / (tgt_df * wdf))
            rows.append({"time": t_k, "target": target_token, "word": w,
                         "cooccurrence_reviews": c, "target_review_frequency": tgt_df,
                         "word_review_frequency": wdf, "pmi": pmi})

    if not rows:
        print("No PMI data generated after filtering. Try lowering --min-word-df.")
        return

    df = pd.DataFrame(rows)
    
    # Re-rank based on PMI after filtering.
    df["rank"] = df.groupby("time")["pmi"].rank(ascending=False, method="first").astype(int)
    df = df.sort_values(["time", "rank"]).reset_index(drop=True)
    df.to_csv(out / "pmi_associations_by_time.csv", index=False)

    # 3. Run multiple analysis
    overlap_df = calculate_year_overlap(df, top_n)
    lifecycle_df = calculate_word_lifecycle(df, top_n)
    movers_df = calculate_top_movers(df, min_years=3, k=15)

    # 4. Export CSV
    overlap_df.to_csv(out / "year_overlap.csv", index=False)
    lifecycle_df.to_csv(out / "word_lifecycle.csv", index=False)
    movers_df.to_csv(out / "top_movers.csv", index=False)

    # 5. Export Summary JSON
    mean_jac = float(overlap_df["jaccard"].mean()) if not overlap_df.empty else None
    mean_rho = float(overlap_df["rank_corr_on_overlap"].dropna().mean()) if not overlap_df.empty and not overlap_df["rank_corr_on_overlap"].dropna().empty else None
    
    summary = {
        "target": target_token,
        "years_analysed": sorted(df["time"].unique().tolist()),
        "n_years": len(df["time"].unique()),
        "min_word_df_filter": min_word_df,
        "mean_jaccard": round(mean_jac, 4) if mean_jac is not None else None,
        "mean_spearman": round(mean_rho, 4) if mean_rho is not None else None,
        "most_persistent_words": lifecycle_df.head(10)[["word", "years_present", "coverage_pct"]].to_dict("records") if not lifecycle_df.empty else [],
        "top_rising_words": movers_df[movers_df["slope_pmi_per_year"] > 0].nlargest(10, "slope_pmi_per_year")[["word", "slope_pmi_per_year"]].to_dict("records") if not movers_df.empty else [],
        "top_falling_words": movers_df[movers_df["slope_pmi_per_year"] < 0].nsmallest(10, "slope_pmi_per_year")[["word", "slope_pmi_per_year"]].to_dict("records") if not movers_df.empty else [],
    }
    (out / "drift_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # 6. Export plots
    plot_heatmap(df, out / "pmi_heatmap.png", target_token, top_n)
    plot_overlap(overlap_df, out / "overlap_over_time.png", target_token)
    plot_movers(movers_df, out / "top_movers.png", k=15)

    # 7. Summary report
    print(f"\n{'='*70}")
    print(f"Semantic Drift Analysis for '{target_token.upper()}'")
    print(f"{'='*70}")
    if mean_jac is not None: print(f"Mean Jaccard overlap (year-to-year stability): {mean_jac:.4f}")
    print("  (1.0 = identical sets, 0.0 = completely different words each year)\n")
    
    print("Year-by-year changes (Top words to enter and leave):")
    for _, row in overlap_df.iterrows():
        print(f"  {row['year_from']} -> {row['year_to']} (Jaccard: {row['jaccard']:.2f}):")
        if row['new_in_y2']: print(f"    [+] New: {', '.join(row['new_in_y2'][:5])}")
        if row['dropped_from_y1']: print(f"    [-] Dropped: {', '.join(row['dropped_from_y1'][:5])}")

    if summary["most_persistent_words"]:
        print("\nMost Persistent Words:")
        for rec in summary["most_persistent_words"][:5]:
            print(f"  - {rec['word']} ({rec['years_present']} years)")
            
    print(f"\nArtifacts written to {out.resolve()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", default=DEFAULT_REVIEWS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-cooc", type=int, default=DEFAULT_MIN_COOC)
    parser.add_argument("--min-word-df", type=int, default=DEFAULT_MIN_WORD_DF)
    parser.add_argument("--max-reviews", type=int, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    run_drift_analysis(
        reviews_path=args.reviews,
        target=args.target,
        out_dir=args.out,
        top_n=args.top_n,
        min_word_df=args.min_word_df,
        min_cooc=args.min_cooc,
        max_reviews=args.max_reviews,
        workers=args.workers
    )

if __name__ == "__main__":
    main()