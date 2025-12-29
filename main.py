import math
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import yfinance as yf
import argparse
import json
import time
from datetime import datetime
from credit_engine.ratios import calculate_ratios
from credit_engine.scorecard import load_scorecard, evaluate_borrower_cfg
import os
import random
import sys
import re

def resource_path(relative_path: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative_path

# =========================================
# POLICY (industry-style decision rules)
# =========================================
APPROVE_MIN_SCORE = 3.5
REVIEW_MIN_SCORE = 2.8

HARD_BREACHES = [
    {"metric": "dscr", "op": "<", "value": 1.0, "reason": "DSCR below 1.0x"},
    {"metric": "interest_coverage", "op": "<", "value": 1.5, "reason": "Interest coverage below 1.5x"},
]


def _breach(op: str, actual: float, threshold: float) -> bool:
    if op == "<":
        return actual < threshold
    if op == "<=":
        return actual <= threshold
    if op == ">":
        return actual > threshold
    if op == ">=":
        return actual >= threshold
    raise ValueError(f"Unsupported op: {op}")


def driver_reasons(ratios: Dict[str, float], scores: Dict[str, int]) -> List[str]:
    """Soft reasons based on weak metric scores (useful for Review/Decline explainability)."""
    reasons: List[str] = []

    dscr = ratios.get("dscr", math.nan)
    icr = ratios.get("interest_coverage", math.nan)
    de = ratios.get("debt_ebitda", math.nan)
    cr = ratios.get("current_ratio", math.nan)

    if scores.get("dscr_score", 0) <= 2 and not math.isnan(dscr):
        reasons.append(f"Weak cashflow coverage: DSCR {dscr:.2f}x")
    if scores.get("interest_coverage_score", 0) <= 2 and not math.isnan(icr):
        reasons.append(f"Low interest cover: ICR {icr:.2f}x")
    if scores.get("debt_ebitda_score", 0) <= 2 and not math.isnan(de):
        reasons.append(f"High leverage: Debt/EBITDA {de:.2f}x")
    if scores.get("current_ratio_score", 0) <= 2 and not math.isnan(cr):
        reasons.append(f"Tight liquidity: Current ratio {cr:.2f}x")

    return reasons[:3]


def apply_policy(ratios: Dict[str, float], scores: Dict[str, int], total_score: float) -> Dict[str, Any]:
    reasons: List[str] = []

    # -----------------------------
    # 1) Data sufficiency check
    # -----------------------------
    valid_scores = sum(
        1 for k in ["dscr_score", "interest_coverage_score", "debt_ebitda_score", "current_ratio_score"]
        if scores.get(k, 0) > 0
    )

    # If we can't compute a score (or almost everything is missing), mark as Unknown
    if math.isnan(total_score) or valid_scores < 2:
        return {
            "decision": "Unknown",
            "reason_codes": "Insufficient financial data to score (missing key line items/ratios).",
        }

    # -----------------------------
    # 2) Hard breaches (only if metric is available)
    # -----------------------------
    for rule in HARD_BREACHES:
        metric = rule["metric"]
        actual = ratios.get(metric, math.nan)
        if math.isnan(actual):
            continue
        if _breach(rule["op"], actual, rule["value"]):
            reasons.append(rule["reason"])

    if reasons:
        decision = "Decline"
    elif total_score >= APPROVE_MIN_SCORE:
        decision = "Approve"
    elif total_score >= REVIEW_MIN_SCORE:
        decision = "Review"
        reasons.extend(driver_reasons(ratios, scores))
    else:
        decision = "Decline"
        reasons.extend(driver_reasons(ratios, scores))

    return {
        "decision": decision,
        "reason_codes": "; ".join(reasons) if reasons else "",
    }



# =========================================
# 1) Scorecard logic
# =========================================
def score_dscr(dscr: float) -> int:
    if math.isnan(dscr):
        return 0
    if dscr >= 2.0:
        return 5
    elif dscr >= 1.5:
        return 4
    elif dscr >= 1.2:
        return 3
    elif dscr >= 1.0:
        return 2
    else:
        return 1


def score_interest_coverage(icr: float) -> int:
    if math.isnan(icr):
        return 0
    if icr >= 5.0:
        return 5
    elif icr >= 4.0:
        return 4
    elif icr >= 2.5:
        return 3
    elif icr >= 1.5:
        return 2
    else:
        return 1


def score_debt_ebitda(debt_ebitda: float) -> int:
    if math.isnan(debt_ebitda):
        return 0
    if debt_ebitda <= 1.0:
        return 5
    elif debt_ebitda <= 2.0:
        return 4
    elif debt_ebitda <= 3.5:
        return 3
    elif debt_ebitda <= 5.0:
        return 2
    else:
        return 1


def score_current_ratio(cr: float) -> int:
    if math.isnan(cr):
        return 0
    if cr >= 2.0:
        return 5
    elif cr >= 1.5:
        return 4
    elif cr >= 1.2:
        return 3
    elif cr >= 1.0:
        return 2
    else:
        return 1


def calculate_weighted_score(scores: Dict[str, int]) -> float:
    weights = {
        "dscr_score": 0.35,
        "debt_ebitda_score": 0.30,
        "current_ratio_score": 0.20,
        "interest_coverage_score": 0.15,
    }

    total = 0.0
    weight_sum = 0.0

    for key, w in weights.items():
        s = scores.get(key, 0)
        if s > 0:
            total += s * w
            weight_sum += w

    if weight_sum == 0:
        return math.nan

    return total / weight_sum


def score_to_rating(score: float) -> str:
    if math.isnan(score):
        return "NR"
    if score >= 4.5:
        return "AAA"
    elif score >= 4.0:
        return "AA"
    elif score >= 3.5:
        return "A"
    elif score >= 3.0:
        return "BBB"
    elif score >= 2.5:
        return "BB"
    elif score >= 2.0:
        return "B"
    else:
        return "CCC"


def rating_to_risk_band(rating: str) -> str:
    if rating in ("AAA", "AA", "A"):
        return "Low"
    elif rating in ("BBB", "BB"):
        return "Medium"
    elif rating in ("B", "CCC"):
        return "High"
    else:
        return "Unknown"


def evaluate_borrower(borrower: Dict[str, Any]) -> Dict[str, Any]:
    ratios = calculate_ratios(borrower)

    dscr_score = score_dscr(ratios["dscr"])
    icr_score = score_interest_coverage(ratios["interest_coverage"])
    de_score = score_debt_ebitda(ratios["debt_ebitda"])
    cr_score = score_current_ratio(ratios["current_ratio"])

    scores = {
        "dscr_score": dscr_score,
        "interest_coverage_score": icr_score,
        "debt_ebitda_score": de_score,
        "current_ratio_score": cr_score,
    }

    total_score = calculate_weighted_score(scores)
    rating = score_to_rating(total_score)
    risk_band = rating_to_risk_band(rating)

    result: Dict[str, Any] = {
        "name": borrower.get("name", "Unknown"),
        "currency": borrower.get("currency", ""),
        "sector": borrower.get("sector", ""),
        "industry": borrower.get("industry", ""),
        "market_cap": borrower.get("market_cap", math.nan),
        **ratios,
        **scores,
        "total_score": total_score,
        "rating": rating,
        "risk_band": risk_band,
    }

    policy_out = apply_policy(ratios, scores, total_score)
    result.update(policy_out)

    return result


# =========================================
# 2) Yahoo Finance helpers
# =========================================
def _norm_label(s: str) -> str:
    # Lowercase + strip everything except letters/numbers so we can match
    # e.g. "Total Current Assets" == "TotalCurrentAssets"
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _latest_period_col(df: pd.DataFrame):
    """Pick the most recent column for Yahoo financial statements."""
    if df is None or df.empty or len(df.columns) == 0:
        return None
    cols = list(df.columns)
    # yfinance usually uses datetime-like columns; max() is the most recent date.
    try:
        return max(cols)
    except Exception:
        return cols[0]


def get_line_item(df: pd.DataFrame, labels) -> float:
    """Try multiple label names (robust to case/spacing); returns latest-period value."""
    if df is None or df.empty:
        return math.nan

    if isinstance(labels, str):
        labels = [labels]

    latest_col = _latest_period_col(df)
    if latest_col is None:
        return math.nan

    # Build normalised index lookup
    norm_to_row = {}
    for idx in df.index:
        key = _norm_label(idx)
        # keep first occurrence (avoid overwriting if duplicates)
        norm_to_row.setdefault(key, idx)

    def _to_float(v) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return math.nan

    # 1) Exact normalised match
    for label in labels:
        key = _norm_label(label)
        if key in norm_to_row:
            return _to_float(df.loc[norm_to_row[key], latest_col])

    # 2) Fuzzy contains match (last resort)
    for label in labels:
        key = _norm_label(label)
        if not key:
            continue
        for k, row in norm_to_row.items():
            if key in k:
                return _to_float(df.loc[row, latest_col])

    return math.nan

    if isinstance(labels, str):
        labels = [labels]

    for label in labels:
        if label in df.index:
            latest_col = df.columns[0]  # most recent period
            value = df.loc[label, latest_col]
            try:
                return float(value)
            except (TypeError, ValueError):
                return math.nan

    return math.nan

CACHE_VERSION = 2

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days


def _cache_path_for_ticker(ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace("\\", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}.json"


def _is_cache_fresh(path: Path, ttl_seconds: int) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < ttl_seconds


def load_cached_borrower(ticker: str) -> Dict[str, Any] | None:
    p = _cache_path_for_ticker(ticker)
    if not _is_cache_fresh(p, CACHE_TTL_SECONDS):
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        # Invalidate cache across code changes (prevents stale NaNs for items like current assets/liabilities)
        if obj.get("_cache_version") != CACHE_VERSION:
            return None
        return obj
    except Exception:
        return None


def save_cached_borrower(ticker: str, borrower: Dict[str, Any]) -> None:
    borrower = dict(borrower)
    borrower["_cache_version"] = CACHE_VERSION
    p = _cache_path_for_ticker(ticker)
    p.write_text(json.dumps(borrower, indent=2), encoding="utf-8")

def fetch_borrower_from_yahoo(ticker: str) -> Dict[str, Any]:
        # 1) Try cache
    cached = load_cached_borrower(ticker)
    if cached is not None:
        return cached

    # 2) Fetch with retries (Yahoo can be flaky / rate limited)
    last_err = None
    for attempt in range(1, 4):  # 3 attempts
        try:
            t = yf.Ticker(ticker)
            # proceed with your existing code below...
            break
        except Exception as e:
            last_err = e
            # exponential-ish backoff with jitter
            time.sleep((1.5 ** attempt) + random.uniform(0.0, 0.8))

    if last_err is not None and "t" not in locals():
        raise last_err


    fs = t.financials
    bs = t.balance_sheet
    cf = t.cashflow
    info = t.info or {}

    ebitda = get_line_item(fs, ["Ebitda", "EBITDA"])
    ebit = get_line_item(fs, ["Ebit", "EBIT"])
    interest_expense = get_line_item(
        fs,
        ["Interest Expense", "InterestExpense", "Interest Expense Non Operating", "Interest Expense Non-Operating"],
    )
    total_debt = get_line_item(bs, ["Total Debt", "TotalDebt"])
    current_assets = get_line_item(bs, [
        "Total Current Assets",
        "Current Assets",
        "CurrentAssets",
        "TotalCurrentAssets",
    ])
    current_liabilities = get_line_item(bs, [
        "Total Current Liabilities",
        "Current Liabilities",
        "CurrentLiabilities",
        "TotalCurrentLiabilities",
    ])

    debt_repayment = get_line_item(cf, ["Repayment Of Debt", "Debt Repayment", "Repayment of Debt"])

    # Yahoo often returns cash outflows/expenses as negative numbers
    if not math.isnan(interest_expense):
        interest_expense = abs(interest_expense)
    if not math.isnan(debt_repayment):
        debt_repayment = abs(debt_repayment)

    # If we can't find repayment, assume simple 5% amortisation of total debt
    if math.isnan(debt_repayment) and not math.isnan(total_debt):
        debt_repayment = 0.05 * total_debt

    # Annual debt service = interest + principal
    if math.isnan(interest_expense) and math.isnan(debt_repayment):
        annual_debt_service = math.nan
    else:
        annual_debt_service = max(
            0.0,
            (0.0 if math.isnan(interest_expense) else float(interest_expense))
            + (0.0 if math.isnan(debt_repayment) else float(debt_repayment)),
        )

    name = info.get("longName") or info.get("shortName") or ticker

    borrower = {
        "name": name,
        "currency": info.get("currency", ""),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "market_cap": info.get("marketCap", math.nan),
        "ebitda": ebitda,
        "ebit": ebit,
        "interest_expense": interest_expense,
        "total_debt": total_debt,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "annual_debt_service": annual_debt_service,
    }
    save_cached_borrower(ticker, borrower)
    return borrower


# =========================================
# 3) Portfolio / batch runner
# =========================================
def load_tickers(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Ticker file not found: {path}")

    # Try CSV first (supports either 1st column or a 'ticker' column)
    try:
        df = pd.read_csv(p)
        if "ticker" in df.columns:
            s = df["ticker"]
        else:
            s = df.iloc[:, 0]
        tickers = s.astype(str).str.strip().str.upper().tolist()
        tickers = [t for t in tickers if t and t != "NAN"]
        return tickers
    except Exception:
        # Fallback: plain text (one per line)
        lines = p.read_text(encoding="utf-8").splitlines()
        tickers = [ln.strip().upper() for ln in lines if ln.strip()]
        return tickers


def run_portfolio_from_tickers(tickers: List[str], scorecard: Dict[str, Any] | None = None, engine: str = "yaml") -> pd.DataFrame:

    results: List[Dict[str, Any]] = []

    for tkr in tqdm(tickers, desc="Fetching Yahoo financials", unit="ticker"):
        try:
            borrower = fetch_borrower_from_yahoo(tkr)

            if engine == "legacy":
                # uses your existing evaluate_borrower() in main.py
                res = evaluate_borrower(borrower)
            else:
                if scorecard is None:
                    raise ValueError("scorecard is required when engine='yaml'")
                res = evaluate_borrower_cfg(borrower, scorecard)

            res["ticker"] = tkr
            results.append(res)

        except Exception as e:
            results.append(
                {"ticker": tkr, "name": "ERROR", "decision": "Error", "reason_codes": str(e)}
            )

    df = pd.DataFrame(results)

    for col in ["dscr", "interest_coverage", "debt_ebitda", "current_ratio", "total_score", "market_cap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit risk batch runner (Yahoo Finance)")
    parser.add_argument("--tickers", default="input/tickers.csv", help="Path to tickers file (csv or txt)")
    parser.add_argument("--out", default="outputs", help="Base output folder")
    parser.add_argument("--run-name", default=None, help="Optional run folder name (default: timestamp)")
    parser.add_argument(
    "--scorecard",
    default=str(resource_path("config/scorecard.yaml")),
    help="Path to YAML scorecard config",
)

    parser.add_argument("--engine", choices=["yaml", "legacy"], default="yaml", help="Scoring engine to use")
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = Path(args.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers_file = args.tickers
    if not Path(tickers_file).exists():
        print(f"Couldn't find {tickers_file}. Create it with one ticker per line (or a 'ticker' column).")
        print("Example:")
        print("CBA.AX\nBHP.AX\nWBC.AX\nTLS.AX")
        raise SystemExit(1)

    # Only require/load scorecard when using the YAML engine
    scorecard = None
    if args.engine == "yaml":
        if not Path(args.scorecard).exists():
            print(f"Couldn't find scorecard file: {args.scorecard}")
            raise SystemExit(1)
        scorecard = load_scorecard(args.scorecard)

    start = time.time()

    tickers = load_tickers(tickers_file)
    df = run_portfolio_from_tickers(tickers, scorecard=scorecard, engine=args.engine)

    # Save outputs
    decisions_path = out_dir / "credit_decisions.csv"
    summary_path = out_dir / "portfolio_summary.csv"
    watchlist_path = out_dir / "watchlist.csv"
    errors_path = out_dir / "errors.csv"
    meta_path = out_dir / "run_metadata.json"

    df.to_csv(decisions_path, index=False)

    # Summary counts + %
    expected = ["Approve", "Review", "Decline", "Unknown", "Error"]
    counts = (
        df["decision"]
        .fillna("Unknown")
        .value_counts(dropna=False)
        .reindex(expected, fill_value=0)
    )
    summary = counts.rename_axis("decision").reset_index(name="count")
    summary["pct"] = (summary["count"] / summary["count"].sum() * 100).round(1)
    summary.to_csv(summary_path, index=False)

    # Watchlist: what needs manual follow-up
    watchlist = df[df["decision"].isin(["Review", "Unknown"])].copy()
    watchlist.to_csv(watchlist_path, index=False)

    # Errors only
    errors = df[df["decision"].isin(["Error"])].copy()
    errors.to_csv(errors_path, index=False)

    duration = round(time.time() - start, 2)
    meta = {
        "run_name": run_name,
        "tickers_file": tickers_file,
        "scorecard_file": args.scorecard,
        "tickers_count": len(tickers),
        "rows_out": int(len(df)),
        "duration_seconds": duration,
        "outputs": {
            "decisions": str(decisions_path),
            "summary": str(summary_path),
            "watchlist": str(watchlist_path),
            "errors": str(errors_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nExported:")
    print(f" - {decisions_path}")
    print(f" - {summary_path}")
    print(f" - {watchlist_path}")
    print(f" - {errors_path}")
    print(f" - {meta_path}")

    print("\nDecision counts:")
    print(summary)

    print(f"\nRun duration: {duration}s")

