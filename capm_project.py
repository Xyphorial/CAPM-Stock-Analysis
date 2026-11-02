"""End-to-end CAPM analysis for ten U.S. stocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
from scipy import stats


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    if config["return_frequency"] not in {"daily", "monthly"}:
        raise ValueError("return_frequency must be 'daily' or 'monthly'")
    return config


def download_prices(config: dict) -> pd.DataFrame:
    tickers = list(config["stocks"]) + [config["market_ticker"]]
    raw = yf.download(
        tickers,
        start=config["start_date"],
        end=config["end_date"],
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no price data")
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if len(tickers) == 1:
        prices.columns = tickers
    prices = prices.reindex(columns=tickers).dropna(how="all")
    missing = prices.columns[prices.isna().all()].tolist()
    if missing:
        raise RuntimeError(f"No usable Yahoo Finance data for: {missing}")
    return prices.ffill().dropna()


def download_risk_free(config: dict) -> pd.Series:
    series_id = config["risk_free_series"]
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?"
        f"id={series_id}&cosd={config['start_date']}&coed={config['end_date']}"
    )
    with urlopen(url, timeout=30) as response:
        rf = pd.read_csv(response)
    rf.columns = ["date", "annual_percent"]
    rf["date"] = pd.to_datetime(rf["date"])
    rf["annual_percent"] = pd.to_numeric(rf["annual_percent"], errors="coerce")
    return rf.set_index("date")["annual_percent"].sort_index().ffill()


def make_demo_data(config: dict, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """Deterministic synthetic data, used only to verify the pipeline offline."""
    dates = pd.bdate_range(config["start_date"], config["end_date"])
    rng = np.random.default_rng(seed)
    market_r = rng.normal(0.00035, 0.011, len(dates))
    prices = {config["market_ticker"]: 4000 * np.cumprod(1 + market_r)}
    betas = np.linspace(0.55, 1.55, len(config["stocks"]))
    for (ticker, _), beta in zip(config["stocks"].items(), betas):
        noise = rng.normal(0, 0.009, len(dates))
        stock_r = 0.00003 + beta * market_r + noise
        prices[ticker] = 100 * np.cumprod(1 + stock_r)
    return pd.DataFrame(prices, index=dates), pd.Series(4.5, index=dates)


def prepare_returns(prices: pd.DataFrame, annual_rf_percent: pd.Series, config: dict) -> pd.DataFrame:
    frequency = config["return_frequency"]
    if frequency == "monthly":
        prices = prices.resample("ME").last()
        periods = 12
    else:
        periods = 252
    returns = prices.pct_change(fill_method=None).dropna()

    rf = annual_rf_percent.reindex(returns.index, method="ffill")
    rf = rf.bfill()
    periodic_rf = (1 + rf / 100) ** (1 / periods) - 1
    returns["RF"] = periodic_rf
    returns["MKT_EXCESS"] = returns[config["market_ticker"]] - returns["RF"]
    for ticker in config["stocks"]:
        returns[f"{ticker}_EXCESS"] = returns[ticker] - returns["RF"]
    return returns.dropna()


def run_regressions(returns: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rows, models = [], {}
    periods = 252 if config["return_frequency"] == "daily" else 12
    x = sm.add_constant(returns["MKT_EXCESS"])
    for ticker, sector in config["stocks"].items():
        model = sm.OLS(returns[f"{ticker}_EXCESS"], x, missing="drop").fit()
        models[ticker] = model
        residuals = model.resid
        jb = stats.jarque_bera(residuals)
        rows.append({
            "ticker": ticker,
            "sector": sector,
            "alpha_periodic": model.params["const"],
            "alpha_annualized": (1 + model.params["const"]) ** periods - 1,
            "beta": model.params["MKT_EXCESS"],
            "r_squared": model.rsquared,
            "alpha_p_value": model.pvalues["const"],
            "beta_p_value": model.pvalues["MKT_EXCESS"],
            "residual_std": residuals.std(),
            "jarque_bera_p_value": jb.pvalue,
            "observations": int(model.nobs),
        })
    return pd.DataFrame(rows).set_index("ticker"), models


def save_scatter_plots(returns: pd.DataFrame, models: Dict[str, object], config: dict) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(14, 24), sharex=True)
    x = returns["MKT_EXCESS"]
    grid = np.linspace(x.min(), x.max(), 100)
    for ax, ticker in zip(axes.flat, config["stocks"]):
        y = returns[f"{ticker}_EXCESS"]
        model = models[ticker]
        ax.scatter(x, y, s=9, alpha=0.28)
        ax.plot(grid, model.params["const"] + model.params["MKT_EXCESS"] * grid,
                color="crimson", linewidth=2)
        ax.set_title(f"{ticker}: beta={model.params['MKT_EXCESS']:.2f}, R²={model.rsquared:.2f}")
        ax.set_xlabel("Market excess return")
        ax.set_ylabel("Stock excess return")
    fig.suptitle("CAPM regressions", fontsize=18, y=1.002)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "capm_scatter_regressions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_residual_plots(models: Dict[str, object]) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(14, 22))
    for ax, (ticker, model) in zip(axes.flat, models.items()):
        ax.scatter(model.fittedvalues, model.resid, s=9, alpha=0.3)
        ax.axhline(0, color="crimson", linewidth=1.5)
        ax.set_title(ticker)
        ax.set_xlabel("Fitted excess return")
        ax.set_ylabel("Residual")
    fig.suptitle("CAPM residual diagnostics", fontsize=18, y=1.002)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "residual_plots.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_rolling_betas(returns: pd.DataFrame, config: dict) -> pd.DataFrame:
    window = int(config["rolling_window"])
    market = returns["MKT_EXCESS"]
    rolling = pd.DataFrame(index=returns.index)
    market_var = market.rolling(window).var()
    for ticker in config["stocks"]:
        rolling[ticker] = returns[f"{ticker}_EXCESS"].rolling(window).cov(market) / market_var
    ax = rolling.plot(figsize=(15, 8), linewidth=1.25, colormap="tab10")
    ax.axhline(1, color="black", linestyle="--", linewidth=1, label="Market beta = 1")
    ax.set(title=f"Rolling beta ({window} observations)", xlabel="Date", ylabel="Beta")
    ax.legend(ncol=3, fontsize=9)
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUT_DIR / "rolling_beta.png", dpi=180)
    plt.close(ax.figure)
    return rolling


def save_beta_chart(results: pd.DataFrame) -> None:
    ordered = results.sort_values("beta")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=ordered.reset_index(), x="ticker", y="beta", hue="sector", dodge=False, ax=ax)
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set(title="CAPM beta comparison", xlabel="Stock", ylabel="Beta")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Sector")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "beta_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(results: pd.DataFrame, models: Dict[str, object], config: dict, demo: bool) -> None:
    significant = results.index[results["alpha_p_value"] < 0.05].tolist()
    low_beta = results.index[results["beta"] < 1].tolist()
    lines = [
        "# CAPM analysis summary",
        "",
        f"Data mode: {'synthetic demo data' if demo else 'Yahoo Finance + FRED'}",
        f"Window: {config['start_date']} to {config['end_date']}",
        f"Frequency: {config['return_frequency']}",
        "",
        f"Highest beta: {results['beta'].idxmax()} ({results['beta'].max():.3f})",
        f"Lowest beta: {results['beta'].idxmin()} ({results['beta'].min():.3f})",
        f"Stocks with beta below 1: {', '.join(low_beta) or 'None'}",
        f"Statistically significant alpha at 5%: {', '.join(significant) or 'None'}",
        "",
        "Important: statistical significance is not the same as economic significance, and an",
        "in-sample alpha is not evidence that the same alpha will persist out of sample.",
    ]
    (OUTPUT_DIR / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")
    with (OUTPUT_DIR / "regression_summaries.txt").open("w", encoding="utf-8") as file:
        for ticker, model in models.items():
            file.write(f"\n{'=' * 80}\n{ticker}\n{'=' * 80}\n")
            file.write(model.summary().as_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ten-stock CAPM project")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--demo", action="store_true", help="Use reproducible synthetic data")
    args = parser.parse_args()
    config = load_config(args.config)
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    if args.demo:
        prices, rf = make_demo_data(config)
    else:
        prices = download_prices(config)
        rf = download_risk_free(config)

    returns = prepare_returns(prices, rf, config)
    results, models = run_regressions(returns, config)
    results.to_csv(OUTPUT_DIR / "capm_results.csv")
    returns.to_csv(OUTPUT_DIR / "clean_returns.csv")
    save_scatter_plots(returns, models, config)
    save_residual_plots(models)
    rolling = save_rolling_betas(returns, config)
    rolling.to_csv(OUTPUT_DIR / "rolling_betas.csv")
    save_beta_chart(results)
    write_summary(results, models, config, args.demo)
    print(f"Finished. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

