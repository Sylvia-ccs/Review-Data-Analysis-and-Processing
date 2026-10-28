{\rtf1\ansi\ansicpg936\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 HelveticaNeue;\f1\fnil\fcharset134 PingFangSC-Regular;}
{\colortbl;\red255\green255\blue255;\red13\green14\blue17;\red255\green255\blue255;\red13\green14\blue17;
}
{\*\expandedcolortbl;;\cssrgb\c5882\c6667\c8235;\cssrgb\c100000\c100000\c100000;\cssrgb\c5882\c6667\c8235;
}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sa320\partightenfactor0

\f0\fs32 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 # Yelp Trend Analysis & PMI Drift (Assignment 3.4)\
\
This repository contains the code for **Section 3.4 (Trend Analysis)** of the Text Data Management assignment. It implements Lexicon-Based Sentiment Mining and Term Co-occurrence Drift (PMI) on the Yelp Open Dataset.\
\
The pipeline is split into three scripts to offer flexibility between full-scale data extraction, strict unfiltered PMI (to demonstrate long-tail noise), and filtered PMI (to reveal meaningful semantic trends).\
\
## Script Overview\
\
### 1. `trend_analysis.py`\
This is the main extraction script for **Section 3.4.1** and **Section 3.4.2**. \
*   **3.4.1 (Sentiment)**: Calculates VADER sentiment scores per year specifically for the **"Fast Food"** sector.\
*   **3.4.2 (PMI)**: Computes the Pointwise Mutual Information (PMI) for the target term **"price"**. It uses a custom tokenizer, multiprocessing, and fast JSON parsing for high performance.\
\
**Run Command:**\
```bash\
python trend_analysis.py \\\
    --reviews yelp_academic_dataset_review.json \\\
    --business yelp_academic_dataset_business.json \\\
    --out outputs/trends \\\
    --target price \\\
    --workers 8\
```\
\
**Key Outputs (`outputs/trends/`):**\
*   `sentiment_trends.png`: Line chart of Fast Food sentiment over time.\
*   `pmi_associations_by_time.csv`: Raw PMI calculations per year (unfiltered).\
\
---\
\
### 2. `pmi_drift_analysis.py` *(Unfiltered Version)*\
This script reads the output CSV from `trend_analysis.py` (or recomputes from raw) and performs drift analysis **without filtering rare terms**. \
*   **Why use this?** This is useful for the report to demonstrate the exact mathematical flaw of standard PMI (where rare terms dominate the top-N lists, causing Jaccard overlap to collapse to 0). It outputs the same suite of charts and CSVs as the filtered version but will show highly fragmented results.\
\
**Run Command:**\
```bash\
# Automatically loads CSV if present, otherwise computes from raw\
python pmi_drift_analysis.py \\\
    --pmi-csv outputs/trends/pmi_associations_by_time.csv \\\
    --target price \\\
    --out outputs/unfiltered_pmi_drift\
```\
\
**Key Outputs (`outputs/unfiltered_pmi_drift/`):**\
*   `pmi_heatmap.png`, `overlap_over_time.png`, `top_movers.png`.\
*   `drift_summary.json`.\
\pard\pardeftab720\sa320\partightenfactor0
\cf4 \cb3 \outl0\strokewidth0 ---\
\
### 3. `term_cooccurrence_drift.py` *(Recommended for final analysis)*\
This is a standalone script that re-computes PMI from raw JSON but **filters out rare/low-frequency words** (`--min-word-df`, default 20). \
*   **Why use this?** Pure PMI is heavily biased towards rare words (e.g., typos that only appear 5 times). Without filtering, your year-to-year overlap will be zero. This script filters the noise to reveal actual semantic shifts.\
*   **Features**: Calculates year-to-year Jaccard overlap, Spearman rank correlation, word lifecycles, and top rising/falling words.\
\
**Run Command:**\
```bash\
python term_cooccurrence_drift.py \\\
    --reviews yelp_academic_dataset_review.json \\\
    --target price \\\
    --out outputs/filtered_pmi_drift \\\
    --min-word-df 20 \\\
    --workers 8\
```\
\
**Key Outputs (`outputs/filtered_pmi_drift/`):**\
*   `pmi_heatmap.png`: A beautiful, continuous heatmap showing how word associations shift over time.\
*   `overlap_over_time.png`: Jaccard and Spearman metrics to quantify semantic drift.\
*   `top_movers.png`: Bar charts showing which words became more/less associated with "price".\
*   `top_movers.csv`, `word_lifecycle.csv`, `year_overlap.csv`, `drift_summary.json`.\
\pard\pardeftab720\sa320\partightenfactor0
\cf2 \cb3 \outl0\strokewidth0 \strokec2 \
---\
\
## How to Answer "Did the words change from year to year?"\
When you run \cf4 \cb3 \outl0\strokewidth0 `pmi_drift_analysis.py` or \cf2 \cb3 \outl0\strokewidth0 \strokec2 `term_cooccurrence_drift.py`, the terminal will automatically print a semantic drift report answering the assignment question directly:\
\
1.  **Mean Jaccard overlap**: `1.0` means the top words are identical every year. `0.0` means completely different words.\
2.  **Year-by-year changes**: It lists `
\f1 [
\f0 +
\f1 ]
\f0  New` (words entering the top-N) and `
\f1 [
\f0 -
\f1 ]
\f0  Dropped` (words leaving the top-N).\
3.  **Most Persistent Words**: Words that survived across the most years.\
4.  **Top Rising/Falling Words**: Words whose PMI slope over time is significantly positive or negative.\
\
## Visualizations Used in the Report\
*   **`sentiment_trends.png`**: Use this for 3.4.1 to show how Fast Food sentiment shifted over the years.\
*   **`pmi_heatmap.png`**: (From the **filtered** version) Use this to visualize how the context of "price" changed (e.g., shifts from "cheap" to "value" or "expensive").\
*   **`overlap_over_time.png`**: Use this to prove whether semantic drift occurred (Jaccard) and if the surviving words reshuffled in importance (Spearman).}