# Kalshi Market Making — Avellaneda–Stoikov Backtest on BTC Hourly Markets

A research pipeline for market making on [Kalshi](https://kalshi.com) prediction markets, focused on the **KXBTC** series (hourly "Bitcoin price at X" markets). It collects public market data, screens markets for maker edge, fits Avellaneda–Stoikov model parameters, and backtests a quoting strategy against real trade prints.

## Pipeline

The scripts run in order, each feeding the next through parquet files:

1. **`collect_data.py`** — pulls settled markets, trade prints, and 1-minute candles for a series from Kalshi's public API (no auth needed) into `<SERIES>_markets/trades/candles.parquet`. Also builds `ranked_data.parquet`: per-market liquidity stats (trades/hour, median spread, mid volatility, post-trade adverse-selection drift) combined into an estimated **edge per fill** (half-spread − maker fee − 1-minute drift) and **edge flow** (edge × fill rate), used to rank markets.
2. **`get_params.py`** — fits Avellaneda–Stoikov inputs on the top-ranked market:
   - **σ** — 60-minute rolling std of 1-minute mid changes
   - **k, A** — exponential fill-intensity fit: trades/hour vs. distance from mid, `λ(δ) = A·e^(−kδ)`, via log-linear regression
   - Writes `params.parquet` and diagnostic plots (`sigma.html`, `k_fit.html`).
3. **`backtest.py`** — minute-by-minute replay over all ranked markets. Each minute, quotes come from the A–S reservation price + optimal spread:
   - reservation price `r = mid − q·γσ²T`, spread `δ = γσ²T + (2/γ)·ln(1 + γ/k)`, capped at 4¢
   - a quote fills when a real trade prints through it (max 2 fills/min/side, size 1)
   - maker fee `0.0175·p·(1−p)` per fill, inventory capped at ±100, quoting stops 90 min before close, leftover inventory settles at the market's resolution
   - Outputs a per-market P&L table, summary stats, `pnl_hist.html` (P&L distribution) and `pnl_curve.html` (cumulative equity curve).

Helper scripts: `experiment.py` (API exploration — series info, orderbooks), `see_parquet.py` (quick parquet inspection).

## Data in this repo

Collected from Kalshi's public API for the KXBTC series, roughly **2026-07-24 to 2026-08-28**:

| File | Contents |
|---|---|
| `KXBTC_markets.parquet` | settled market metadata |
| `KXBTC_trades.parquet` | ~616k trade prints |
| `KXBTC_candles.parquet` | ~846k 1-minute candles |
| `ranked_data.parquet` | 344 markets ranked by estimated maker edge |
| `params.parquet` | fitted A–S parameters (k ≈ 0.033/¢, A ≈ 2.9 trades/hr at mid) |

The `.html` files are interactive Plotly outputs of the fits and backtest results.

## Running it

```bash
pip install polars numpy plotly requests tqdm
python collect_data.py   # or reuse the parquet files included here
python get_params.py
python backtest.py
```

## Caveats

This is a research backtest, not a live P&L claim:

- **Optimistic fills** — any print crossing the quote is assumed to fill; real queue position and priority are ignored.
- Quotes are static within each minute; candle-close mids are used as the reference price.
- Parameters (γ = 0.1, 90-min cutoff) were chosen with an in-sample grid search plus a rough old/new out-of-sample split — limited data, so overfitting risk remains.
- Kalshi fee schedules change; the 1.75% maker fee rate is hardcoded.
