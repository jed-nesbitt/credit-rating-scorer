import math
from typing import Dict, Any

def safe_div(numerator: float, denominator: float) -> float:
    """
    Safely perform division, returning NaN if denominator is 0 or None.
    """
    if denominator is None or denominator == 0 or (isinstance(denominator, float) and math.isnan(denominator)):
        return math.nan
    return math.nan if numerator is None or (isinstance(numerator, float) and math.isnan(numerator)) else numerator / denominator


def calculate_ratios(borrower: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate key credit ratios for a single borrower.
    Expected keys in `borrower`:
        ebitda, ebit, interest_expense,
        total_debt, current_assets, current_liabilities,
        annual_debt_service
    """
    ebitda = borrower.get("ebitda", 0.0)
    ebit = borrower.get("ebit", 0.0)
    interest_expense = borrower.get("interest_expense", 0.0)
    total_debt = borrower.get("total_debt", 0.0)
    current_assets = borrower.get("current_assets", 0.0)
    current_liabilities = borrower.get("current_liabilities", 0.0)
    annual_debt_service = borrower.get("annual_debt_service", 0.0)

    dscr = safe_div(ebitda, annual_debt_service)
    icr = safe_div(ebit, interest_expense)
    debt_ebitda = safe_div(total_debt, ebitda)
    current_ratio = safe_div(current_assets, current_liabilities)

    return {
        "dscr": dscr,
        "interest_coverage": icr,
        "debt_ebitda": debt_ebitda,
        "current_ratio": current_ratio,
    }
