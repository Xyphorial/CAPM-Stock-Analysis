> **Data provenance warning:** The included `demo_outputs/` results are **synthetic demonstrations only**, not backtests or observed stock-market results. Run `capm_project.py` without `--demo` to generate real market-data outputs. Internet access and upstream data availability are required.

# CAPM Quant Project: 10 U.S. Stocks

This project follows the supplied assignment: it estimates each stock's market exposure with

`stock excess return = alpha + beta * market excess return + error`.

The default universe covers 10 selected GICS sectors (not a complete list of all GICS sectors):
AAPL, JPM, JNJ, XOM, WMT, CAT, NEE, AMZN, PLD, and T. The S&P 500 (`^GSPC`) is the market proxy,
and FRED's 3-month Treasury bill series (`DTB3`) is the risk-free rate.

## Run on your Mac

Easy method: double-click `run_on_mac.command`. If macOS blocks it, right-click the file,
choose **Open**, then confirm. If it says permission denied, run
`chmod +x run_on_mac.command` once in Terminal.

Terminal method:

Open Terminal, enter this project folder, and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 capm_project.py
```

Internet access is required for Yahoo Finance and FRED. The first run may take a minute.
Yahoo occasionally returns HTTP 429 when it is busy; wait a few minutes and run the command again.

To verify the whole pipeline without downloading data:

```bash
python3 capm_project.py --demo
```

## Outputs

A new execution writes results to `outputs/`. The checked-in examples are stored separately under `demo_outputs/` and were generated from **synthetic demo data**, not real historical market observations. The sample coefficients, p-values, and figures must not be presented as empirical findings.

Output files produced by a run include:

- `capm_results.csv`: alpha, annualized alpha, beta, R-squared, p-values, residual statistics.
- `capm_scatter_regressions.png`: ten scatter plots with fitted CAPM lines.
- `residual_plots.png`: fitted-value-versus-residual diagnostics (regenerated when the pipeline runs; no valid archived example was available).
- `rolling_beta.png` and `rolling_betas.csv`: 126-observation rolling betas.
- `beta_comparison.png`: cross-stock beta comparison.
- `regression_summaries.txt`: complete statsmodels OLS summaries.
- `analysis_summary.md`: concise interpretation of the main findings.
- `clean_returns.csv`: aligned prices converted to returns and excess returns.

## Customize it

Edit `config.json` to change dates, stock tickers, sectors, or the rolling window. For monthly
returns, set `return_frequency` to `monthly`. A useful extension is to run once with daily data,
rename the output folder, then rerun with monthly data and compare beta stability and p-values.

## How the calculation works

1. Adjusted closing prices are downloaded and aligned by trading date.
2. Simple return is computed as `price_today / price_yesterday - 1`.
3. The annualized T-bill yield is converted to a daily or monthly effective rate.
4. The risk-free rate is subtracted from both stock and market returns.
5. OLS estimates alpha (intercept) and beta (slope) separately for each stock.
6. Residual normality is checked with the Jarque-Bera p-value.
7. Rolling beta is computed as rolling covariance(stock, market) / rolling variance(market).

This is an educational research project, not investment advice.
