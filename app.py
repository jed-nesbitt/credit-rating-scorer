from pathlib import Path
import time
import random
import math

import pandas as pd
import streamlit as st

import main  # uses your existing functions: fetch_borrower_from_yahoo, evaluate_borrower
from credit_engine.scorecard import load_scorecard, evaluate_borrower_cfg
import sys
from pathlib import Path

def resource_path(relative_path: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative_path


st.set_page_config(page_title="Credit Risk Engine", layout="wide")
st.title("Credit Risk Engine (Yahoo Finance)")

# -----------------------------
# Helpers
# -----------------------------
def clean_tickers(tickers):
    out = []
    for t in tickers:
        if t is None:
            continue
        t = str(t).strip().upper()
        if t and t != "NAN":
            out.append(t)
    # de-dupe, preserve order
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def load_tickers_from_upload(uploaded_file) -> list[str]:
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    # CSV
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        if "ticker" in df.columns:
            tickers = df["ticker"].tolist()
        else:
            tickers = df.iloc[:, 0].tolist()
        return clean_tickers(tickers)

    # TXT fallback
    text = data.decode("utf-8", errors="ignore")
    tickers = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return clean_tickers(tickers)


def build_outputs(df: pd.DataFrame):
    expected = ["Approve", "Review", "Decline", "Unknown", "Error"]
    counts = (
        df["decision"]
        .fillna("Unknown")
        .value_counts(dropna=False)
        .reindex(expected, fill_value=0)
    )
    summary = counts.rename_axis("decision").reset_index(name="count")
    summary["pct"] = (summary["count"] / summary["count"].sum() * 100).round(1)

    watchlist = df[df["decision"].isin(["Review", "Unknown"])].copy()
    errors = df[df["decision"].isin(["Error"])].copy()

    return summary, watchlist, errors


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Run settings")

engine = st.sidebar.selectbox("Engine", ["yaml", "legacy"], index=0)
scorecard_path = st.sidebar.text_input(
    "Scorecard path (YAML)",
    value=str(resource_path("config/scorecard.yaml"))
)

throttle = st.sidebar.slider("Throttle between tickers (seconds)", 0.0, 2.0, 0.3, 0.1)

source = st.sidebar.radio("Tickers source", ["Upload file", "Use input/tickers.csv", "Paste tickers"], index=0)

tickers = []
uploaded = None

if source == "Upload file":
    uploaded = st.sidebar.file_uploader("Upload tickers.csv or tickers.txt", type=["csv", "txt"])
    if uploaded is not None:
        tickers = load_tickers_from_upload(uploaded)

elif source == "Use input/tickers.csv":
    p = Path("input/tickers.csv")
    if p.exists():
        tickers = main.load_tickers(str(p))
    else:
        st.sidebar.warning("Couldn't find input/tickers.csv")

else:
    pasted = st.sidebar.text_area("Paste tickers (one per line)", value="")
    if pasted.strip():
        tickers = clean_tickers(pasted.splitlines())

st.sidebar.write(f"Tickers loaded: **{len(tickers)}**")

run = st.sidebar.button("Run credit scoring", type="primary")


# -----------------------------
# Run
# -----------------------------
if run:
    if len(tickers) == 0:
        st.error("No tickers loaded. Add tickers then run.")
        st.stop()

    # Load YAML scorecard only if needed
    scorecard = None
    if engine == "yaml":
        if not Path(scorecard_path).exists():
            st.error(f"Scorecard file not found: {scorecard_path}")
            st.stop()
        scorecard = load_scorecard(scorecard_path)

    st.info("Running…")
    progress = st.progress(0)
    status = st.empty()

    results = []
    n = len(tickers)

    start = time.time()

    for i, tkr in enumerate(tickers, start=1):
        status.write(f"{i}/{n} — {tkr}")
        try:
            borrower = main.fetch_borrower_from_yahoo(tkr)

            if engine == "legacy":
                res = main.evaluate_borrower(borrower)
            else:
                res = evaluate_borrower_cfg(borrower, scorecard)

            res["ticker"] = tkr
            results.append(res)

        except Exception as e:
            results.append({"ticker": tkr, "name": "ERROR", "decision": "Error", "reason_codes": str(e)})

        progress.progress(i / n)
        time.sleep(throttle + random.uniform(0.0, 0.2))

    df = pd.DataFrame(results)

    # Round key numeric columns if present
    for col in ["dscr", "interest_coverage", "debt_ebitda", "current_ratio", "total_score", "market_cap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    duration = round(time.time() - start, 2)
    st.success(f"Done in {duration}s")

    summary, watchlist, errors = build_outputs(df)

    # -----------------------------
    # Display
    # -----------------------------
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Decisions")
        st.dataframe(df, use_container_width=True)

    with c2:
        st.subheader("Summary")
        st.dataframe(summary, use_container_width=True)

    st.subheader("Downloads")
    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.download_button("Download decisions CSV", df_to_csv_bytes(df), file_name="credit_decisions.csv")
    with d2:
        st.download_button("Download summary CSV", df_to_csv_bytes(summary), file_name="portfolio_summary.csv")
    with d3:
        st.download_button("Download watchlist CSV", df_to_csv_bytes(watchlist), file_name="watchlist.csv")
    with d4:
        st.download_button("Download errors CSV", df_to_csv_bytes(errors), file_name="errors.csv")
