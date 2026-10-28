#!/usr/bin/env python3
"""Trend Analysis (VADER + PMI drift) for Assignment 3.4."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re as std_re
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterator

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
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ==========================================================================
# ▼▼▼ DEFAULT CONFIG ▼▼▼
# ==========================================================================
DEFAULT_REVIEWS      = "yelp_academic_dataset_review.json"
DEFAULT_BUSINESS     = "yelp_academic_dataset_business.json"
DEFAULT_OUT          = "outputs/trends"
DEFAULT_TARGET       = "price"    # 3.4.2 target aspect
DEFAULT_TOP_N        = 15
DEFAULT_MIN_COOC     = 5
DEFAULT_TIME_UNIT    = "year"
DEFAULT_MAX_REVIEWS  = None
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
    if not text:
        return []
    out = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if _URL_RE.match(raw):
            out.append("<url>")
        else:
            out.append(raw)
    return out

def load_business_map(path: str | Path) -> dict[str, dict[str, Any]]:
    businesses = {}
    with open(path, "rb") as f:
        for line in f:
            if not line.strip(): continue
            try: row = _json_loads(line)
            except Exception: continue
            bid = row.get("business_id")
            if not bid: continue
            categories = {c.strip().casefold() for c in (row.get("categories") or "").split(",") if c.strip()}
            businesses[bid] = {"categories": categories}
    return businesses

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

def _parse_year(d: Any) -> int | None:
    if not isinstance(d, str) or len(d) < 4: return None
    try: return int(d[:4])
    except ValueError: return None

def _parse_month(d: Any) -> int | None:
    if not isinstance(d, str) or len(d) < 7: return None
    try: return int(d[5:7])
    except ValueError: return None

_SEASON = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
_SEASON_ORDER = {"Winter":0, "Spring":1, "Summer":2, "Fall":3}

def _time_key_year(d): 
    y = _parse_year(d); return f"{y:04d}" if y else None
def _time_key_quarter(d):
    y,m = _parse_year(d), _parse_month(d)
    return f"{y:04d}-Q{(m-1)//3+1}" if y and m else None
def _time_key_season(d):
    y,m = _parse_year(d), _parse_month(d)
    return f"{y:04d}-{_SEASON_ORDER[_SEASON[m]]:02d}-{_SEASON[m]}" if y and m else None

_TIME_KEY_GETTERS = {"year": _time_key_year, "quarter": _time_key_quarter, "season": _time_key_season}

STOPWORDS = set("""a an the and or but if then than to of for from in on at by with as is are was were be been being it its this that these those i me my we our you your he she they them their us not no do does did have has had am can could should would will just very really so too only also there here what when where who why how all any some many much more most less other another into out up down over under again about after before between while because during without within""".split())

_W_businesses = None
_W_target = None
_W_time_key_of = None
_W_analyzer = None

def _init_worker(bus, tgt, time_func):
    global _W_businesses, _W_target, _W_time_key_of, _W_analyzer
    _W_businesses = bus; _W_target = tgt; _W_time_key_of = time_func
    _W_analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None

def _process_chunk(lines: list[bytes]):
    docs = Counter(); word_docs = defaultdict(Counter); target_docs = Counter()
    cooc = defaultdict(Counter); sent_sum = defaultdict(float); sent_count = Counter()
    sent_pos = Counter(); sent_neg = Counter()

    for raw in lines:
        try: row = _json_loads(raw)
        except Exception: continue
        time_key = _W_time_key_of(row.get("date"))
        if not time_key: continue
        text = row.get("text") or ""
        tokens = custom_tokenize(text)
        
        bus = _W_businesses.get(row.get("business_id"))
        if bus:
            cats = bus["categories"]
            if "fast food" in cats and _W_analyzer:
                score = _W_analyzer.polarity_scores(text)["compound"]
                k = (time_key, "Fast Food")
                sent_sum[k] += score; sent_count[k] += 1
                sent_pos[k] += score >= 0.05; sent_neg[k] += score <= -0.05

        token_set = set(tokens)
        docs[time_key] += 1
        word_docs[time_key].update(token_set)
        if _W_target in token_set:
            target_docs[time_key] += 1
            cooc[time_key].update(token_set - {_W_target})

    return docs, dict(word_docs), target_docs, dict(cooc), dict(sent_sum), dict(sent_count), dict(sent_pos), dict(sent_neg)

def _wrap_progress(it, total, desc):
    return tqdm(it, total=total, desc=desc, unit="chunk") if tqdm else it

def analyze_trends(reviews_path, business_path, output_dir, max_reviews=None, target="price",
                   top_n=15, min_cooc=5, time_unit="year", workers=None, chunk_size=DEFAULT_CHUNK_SIZE):
    if SentimentIntensityAnalyzer is None:
        raise ImportError("vaderSentiment required.")
    target_token = custom_tokenize(target)[0]
    if workers is None: workers = min(os.cpu_count() or 1, 8)

    businesses = load_business_map(business_path)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    
    agg = {"docs":Counter(), "word_docs":defaultdict(Counter), "target_docs":Counter(),
           "cooc":defaultdict(Counter), "sent_sum":defaultdict(float), "sent_count":Counter(),
           "sent_pos":Counter(), "sent_neg":Counter()}

    t0 = time.time()
    time_key_func = _TIME_KEY_GETTERS[time_unit]
    total_chunks = max(1, (max_reviews + chunk_size - 1) // chunk_size) if max_reviews else None

    if workers <= 1:
        _init_worker(businesses, target_token, time_key_func)
        for chunk in _wrap_progress(_chunked_lines(reviews_path, max_reviews, chunk_size), total_chunks, "Processing"):
            d,wd,td,c,ss,sc,sp,sn = _process_chunk(chunk)
            agg["docs"].update(d); [agg["word_docs"][k].update(v) for k,v in wd.items()]
            agg["target_docs"].update(td); [agg["cooc"][k].update(v) for k,v in c.items()]
            [agg["sent_sum"].__setitem__(k, agg["sent_sum"][k]+v) for k,v in ss.items()]
            agg["sent_count"].update(sc); agg["sent_pos"].update(sp); agg["sent_neg"].update(sn)
    else:
        with Pool(processes=workers, initializer=_init_worker, initargs=(businesses, target_token, time_key_func)) as pool:
            res = pool.imap_unordered(_process_chunk, _chunked_lines(reviews_path, max_reviews, chunk_size), chunksize=1)
            for d,wd,td,c,ss,sc,sp,sn in _wrap_progress(res, total_chunks, f"Processing ({workers} workers)"):
                agg["docs"].update(d); [agg["word_docs"][k].update(v) for k,v in wd.items()]
                agg["target_docs"].update(td); [agg["cooc"][k].update(v) for k,v in c.items()]
                [agg["sent_sum"].__setitem__(k, agg["sent_sum"][k]+v) for k,v in ss.items()]
                agg["sent_count"].update(sc); agg["sent_pos"].update(sp); agg["sent_neg"].update(sn)

    elapsed = time.time() - t0
    print(f"[done] Processed in {elapsed:.1f}s", flush=True)

    # --- 3.4.1 Sentiment CSV & Plot ---
    sent_rows = []
    for (t_k, sec), cnt in sorted(agg["sent_count"].items()):
        sent_rows.append({
            "time": t_k, "sector": sec, "review_count": cnt,
            "mean_vader": agg["sent_sum"][(t_k, sec)] / cnt,
            "pos_share": agg["sent_pos"][(t_k, sec)] / cnt,
            "neg_share": agg["sent_neg"][(t_k, sec)] / cnt
        })
    with open(out/"sentiment_by_time.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time","sector","review_count","mean_vader","pos_share","neg_share"])
        w.writeheader(); w.writerows(sent_rows)

    times = sorted({r["time"] for r in sent_rows})
    t_idx = {t:i for i,t in enumerate(times)}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    rows = [r for r in sent_rows if r["sector"] == "Fast Food"]
    if rows:
        ax.plot([t_idx[r["time"]] for r in rows], [r["mean_vader"] for r in rows], marker="o", label="Fast Food")
    
    ax.axhline(0, color="black", lw=1, alpha=0.5); ax.set_xticks(range(len(times)))
    ax.set_xticklabels(times, rotation=45, ha="right")
    ax.set_ylabel("Mean VADER score"); ax.set_title(f"Mean VADER Sentiment for Fast Food by {time_unit.title()}")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(out/"sentiment_trends.png", dpi=160); plt.close(fig)

    # --- 3.4.2 PMI CSV ---
    pmi_rows = []
    for t_k in sorted(agg["docs"]):
        tot = agg["docs"][t_k]
        tgt_df = agg["target_docs"][t_k]
        if not tot or not tgt_df: continue
        cand = []
        for w, c in agg["cooc"][t_k].items():
            if w == target_token or w in STOPWORDS or c < min_cooc: continue
            wdf = agg["word_docs"][t_k][w]
            if not wdf: continue
            pmi = math.log2((c * tot) / (tgt_df * wdf))
            p_xy = c / tot
            npmi = pmi / (-math.log2(p_xy)) if 0 < p_xy < 1 else 0.0
            cand.append((w, c, wdf, pmi, npmi))
        cand.sort(key=lambda x: x[3], reverse=True)
        for rank, (w, c, wdf, pmi, npmi) in enumerate(cand[:top_n], start=1):
            pmi_rows.append({"time":t_k,"target":target_token,"rank":rank,"word":w,
                             "cooccurrence_reviews":c,"target_review_frequency":tgt_df,
                             "word_review_frequency":wdf,"pmi":pmi,"npmi":npmi})

    with open(out/"pmi_associations_by_time.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time","target","rank","word","cooccurrence_reviews","target_review_frequency","word_review_frequency","pmi","npmi"])
        w.writeheader(); w.writerows(pmi_rows)

    summary = {
        "reviews_processed": sum(agg["docs"].values()), "time_unit": time_unit, "workers": workers,
        "sentiment_sectors": ["Fast Food"],
        "target_term": target_token, "pmi_rows_written": len(pmi_rows)
    }
    with open(out/"trend_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", default=DEFAULT_REVIEWS)
    parser.add_argument("--business", default=DEFAULT_BUSINESS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--max-reviews", type=int, default=DEFAULT_MAX_REVIEWS)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-cooc", type=int, default=DEFAULT_MIN_COOC)
    parser.add_argument("--time-unit", default=DEFAULT_TIME_UNIT, choices=["year","quarter","season"])
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    summary = analyze_trends(args.reviews, args.business, args.out, args.max_reviews,
                             args.target, args.top_n, args.min_cooc, args.time_unit, args.workers)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()