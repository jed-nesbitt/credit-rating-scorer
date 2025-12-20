import math
from typing import Dict, Any, List

import yaml

from credit_engine.ratios import calculate_ratios


def load_scorecard(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Very light validation (enough to avoid silly crashes)
    if "metrics" not in cfg or "weights" not in cfg or "policy" not in cfg or "ratings" not in cfg:
        raise ValueError("Scorecard config missing one of: metrics, weights, policy, ratings")

    return cfg


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


def score_metric(value: float, metric_cfg: Dict[str, Any]) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0

    direction = metric_cfg["direction"]
    bands = metric_cfg["bands"]

    if direction == "higher_is_better":
        # Pick first band whose min is satisfied (bands should be ordered high->low mins)
        for b in bands:
            if value >= float(b["min"]):
                return int(b["score"])
        return 0

    if direction == "lower_is_better":
        # Pick first band whose max is satisfied (bands should be ordered low->high max)
        for b in bands:
            if value <= float(b["max"]):
                return int(b["score"])
        return 0

    raise ValueError(f"Unknown direction: {direction}")


def weighted_total(scores_by_metric: Dict[str, int], weights: Dict[str, float]) -> float:
    total = 0.0
    weight_sum = 0.0
    for metric, w in weights.items():
        s = scores_by_metric.get(metric, 0)
        if s > 0:
            total += s * float(w)
            weight_sum += float(w)

    if weight_sum == 0:
        return math.nan
    return total / weight_sum


def map_rating(total_score: float, ratings_table: List[Dict[str, Any]]) -> Dict[str, str]:
    if total_score is None or (isinstance(total_score, float) and math.isnan(total_score)):
        return {"rating": "NR", "risk_band": "Unknown"}

    # ratings_table should be ordered high->low mins
    for row in ratings_table:
        if total_score >= float(row["min"]):
            return {"rating": row["rating"], "risk_band": row["band"]}

    return {"rating": "NR", "risk_band": "Unknown"}


def driver_reasons(ratios: Dict[str, float], scores: Dict[str, int]) -> List[str]:
    reasons: List[str] = []

    dscr = ratios.get("dscr", math.nan)
    icr = ratios.get("interest_coverage", math.nan)
    de = ratios.get("debt_ebitda", math.nan)
    cr = ratios.get("current_ratio", math.nan)

    if scores.get("dscr", 0) <= 2 and not math.isnan(dscr):
        reasons.append(f"Weak cashflow coverage: DSCR {dscr:.2f}x")
    if scores.get("interest_coverage", 0) <= 2 and not math.isnan(icr):
        reasons.append(f"Low interest cover: ICR {icr:.2f}x")
    if scores.get("debt_ebitda", 0) <= 2 and not math.isnan(de):
        reasons.append(f"High leverage: Debt/EBITDA {de:.2f}x")
    if scores.get("current_ratio", 0) <= 2 and not math.isnan(cr):
        reasons.append(f"Tight liquidity: Current ratio {cr:.2f}x")

    return reasons[:3]


def apply_policy_cfg(ratios: Dict[str, float], scores_by_metric: Dict[str, int], total_score: float, cfg: Dict[str, Any]) -> Dict[str, Any]:
    policy = cfg["policy"]
    reasons: List[str] = []

    # Data sufficiency => Unknown (separate from Decline)
    valid_metrics = sum(1 for s in scores_by_metric.values() if s > 0)
    min_needed = int(policy.get("min_metrics_for_decision", 2))
    if math.isnan(total_score) or valid_metrics < min_needed:
        return {"decision": "Unknown", "reason_codes": "Insufficient financial data to score (missing key line items/ratios)."}

    # Hard breaches
    for rule in policy.get("hard_breaches", []):
        metric = rule["metric"]
        actual = ratios.get(metric, math.nan)
        if math.isnan(actual):
            continue
        if _breach(rule["op"], actual, float(rule["value"])):
            reasons.append(rule["reason"])

    if reasons:
        decision = "Decline"
    else:
        approve_min = float(policy["approve_min_score"])
        review_min = float(policy["review_min_score"])

        if total_score >= approve_min:
            decision = "Approve"
        elif total_score >= review_min:
            decision = "Review"
            reasons.extend(driver_reasons(ratios, scores_by_metric))
        else:
            decision = "Decline"
            reasons.extend(driver_reasons(ratios, scores_by_metric))

    return {"decision": decision, "reason_codes": "; ".join(reasons) if reasons else ""}


def evaluate_borrower_cfg(borrower: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    ratios = calculate_ratios(borrower)

    # Score each configured metric
    metrics_cfg = cfg["metrics"]
    scores_by_metric: Dict[str, int] = {}
    for metric_name, mcfg in metrics_cfg.items():
        scores_by_metric[metric_name] = score_metric(ratios.get(metric_name, math.nan), mcfg)

    total_score = weighted_total(scores_by_metric, cfg["weights"])
    rating_out = map_rating(total_score, cfg["ratings"])
    policy_out = apply_policy_cfg(ratios, scores_by_metric, total_score, cfg)

    # Flatten scores into old-style columns if you want easy reading
    result: Dict[str, Any] = {
        "name": borrower.get("name", "Unknown"),
        "currency": borrower.get("currency", ""),
        "sector": borrower.get("sector", ""),
        "industry": borrower.get("industry", ""),
        "market_cap": borrower.get("market_cap", math.nan),

        **ratios,

        # metric scores (by metric name)
        "dscr_score": scores_by_metric.get("dscr", 0),
        "interest_coverage_score": scores_by_metric.get("interest_coverage", 0),
        "debt_ebitda_score": scores_by_metric.get("debt_ebitda", 0),
        "current_ratio_score": scores_by_metric.get("current_ratio", 0),

        "total_score": total_score,
        **rating_out,
        **policy_out,
    }

    return result
