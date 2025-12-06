📊 Credit Risk Scorecard (Python + Yahoo Finance)

A lightweight Credit Risk Analysis Tool that automatically retrieves real company financial data from Yahoo Finance and generates a full credit scorecard.
The model computes key credit ratios, applies a weighted scoring system, assigns a credit rating (AAA–CCC), and exports results to CSV for reporting and analysis.

This project demonstrates practical skills in Python, financial modelling, credit analysis, data extraction, and automation—ideal for business banking, lending, credit risk, and investment roles.

🔍 Features

Fetches income statement, balance sheet, and cash flow data using yfinance

Calculates essential credit metrics:

DSCR – Debt Service Coverage Ratio

Interest Coverage

Debt / EBITDA

Current Ratio

Converts ratios into 1–5 credit scores

Applies a weighted scoring model:

DSCR (35%)

Debt/EBITDA (30%)

Current Ratio (20%)

Interest Coverage (15%)

Generates:

Overall Credit Score

Letter Rating (AAA–CCC)

Risk Band (Low / Medium / High)

Exports results to CSV for easy review or comparison
