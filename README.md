## Credit Risk Scorecard (Python + Yahoo Finance)

A lightweight Credit Risk Analysis Tool that automatically retrieves real company financial data from Yahoo Finance and generates a full credit scorecard.
The model computes key credit ratios, applies a weighted scoring system, assigns a credit rating (AAA–CCC), and exports results to CSV for reporting and analysis.

This project demonstrates practical skills in Python, financial modelling, credit analysis, data extraction, and automation—ideal for business banking, lending, credit risk, and investment roles.

## Features

1. Fetches income statement, balance sheet, and cash flow data using yfinance
2. Calculates essential credit metrics:
    a. DSCR – Debt Service Coverage Ratio
    b. Interest Coverage
    c. Debt / EBITDA
    d. Current Ratio

3. Converts ratios into 1–5 credit scores
4. Applies a weighted scoring model: (Or the weighting the user can choose)
DSCR (35%)
Debt/EBITDA (30%)
Current Ratio (20%)
Interest Coverage (15%)

5. Does discard if any of the key metrics falls below or above a certain value regradless of the others

6. Generates:
  a. Overall Credit Score
  b. Letter Rating (AAA–CCC)
  c. Risk Band (Low / Medium / High)
  d. Whether to decline accept or review

## How To Run

1. Install Requirements.txt
2. python main.py

## Inputs
Simply need a csv file (tickers.csv) with a single column name 'ticker' and all the tickers of the companies found on yahoo finance you want to rate

## Outputs
1. A CSV file that contains all the errors the program had (errors.csv)
2. A CSV file that contains a summary of the decisions of all the stocks (portfolio_summary.csv)
<img width="241" height="145" alt="image" src="https://github.com/user-attachments/assets/4dca3a22-cdf3-4082-a192-aa697c43e617" />

3. A CSV file that contains all the decisions and ratings of all the stocks and if decline or review the reason why (credit_decisions.csv)
<img width="1521" height="193" alt="image" src="https://github.com/user-attachments/assets/b3712a4b-8347-4ba0-b39c-3b003828c925" />

4. A CSV file that contains only the stocks that require further review also including those stocks which didn't have financial ratios
<img width="1973" height="241" alt="image" src="https://github.com/user-attachments/assets/8f556d63-1c30-4d1a-82ba-04b2e7c6f8ba" />


