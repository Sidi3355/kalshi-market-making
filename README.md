# Kalshi Market Making — Avellaneda–Stoikov Backtest on BTC Hourly Markets

A research pipeline for market making on [Kalshi](https://kalshi.com) prediction markets, focused on the **KXBTC** series (hourly "Bitcoin price at X" markets). It collects public market data, screens markets for maker edge, fits Avellaneda–Stoikov model parameters, and backtests a quoting strategy against real trade prints.

## Pipeline

The scripts run in order, each feeding the next through parquet files:

1. **`src/collect_data.py`** — pulls settled markets, trade prints, and 1-minute candles for a series from Kalshi's public API (no auth needed) into `data/<SERIES>_markets/trades/candles.parquet`. Also builds `data/ranked_data.parquet`: per-market liquidity stats (trades/hour, median spread, mid volatility, post-trade adverse-selection drift) combined into an estimated **edge per fill** (half-spread − maker fee − 1-minute drift) and **edge flow** (edge × fill rate), used to rank markets.
2. **`src/get_params.py`** — fits Avellaneda–Stoikov inputs on the top-ranked market:
   - **σ** — 60-minute rolling std of 1-minute mid changes
   - **k, A** — exponential fill-intensity fit: trades/hour vs. distance from mid, `λ(δ) = A·e^(−kδ)`, via log-linear regression
   - Writes `data/params.parquet` and diagnostic plots (`plots/sigma.html`, `plots/k_fit.html`).
3. **`src/backtest.py`** — minute-by-minute replay over all ranked markets. Each minute, quotes come from the A–S reservation price + optimal spread:
   - reservation price `r = mid − q·γσ²T`, spread `δ = γσ²T + (2/γ)·ln(1 + γ/k)`, capped at 4¢
   - a quote fills when a real trade prints through it (max 2 fills/min/side, size 1)
   - maker fee `0.0175·p·(1−p)` per fill, inventory capped at ±100, quoting stops 90 min before close, leftover inventory settles at the market's resolution
   - Outputs a per-market P&L table, summary stats, `plots/pnl_hist.html` (P&L distribution) and `plots/pnl_curve.html` (cumulative equity curve).

Helper scripts: `src/experiment.py` (API exploration — series info, orderbooks), `src/see_parquet.py` (quick parquet inspection).

## Key results

```
================ BACKTEST SUMMARY ================
Series:              KXBTC (hourly BTC markets)
Period:              2026-07-29 to 2026-08-27
Markets traded:      344
Model:               Avellaneda-Stoikov, delta cap 4c
Parameters:          gamma=0.1, cutoff=90 min, k=3.3/$
Assumptions:         size 1, max 2 fills/min/side, maker fee 1.75% rate
                     fills = any print crossing quote (optimistic)

Starting capital:    $100.00
Final capital:       $148.41
Total P&L:           $48.41  (+48.4%)
Max drawdown:        $-15.93  (-15.9%)

Fills:               5477
P&L per fill:        0.884c
Profitable markets:  228 (66%)
Losing markets:      84
Avg win / avg loss:  $0.57 / $-0.96
Best market:         KXBTC-26JUL2917-B63375  $12.17
Worst market:        KXBTC-26AUG2017-B72625  $-15.93
Max inventory:       20 contracts
Avg fills/market:    16
==================================================
```

See `plots/pnl_curve.html` and `plots/pnl_hist.html` for the equity curve and per-market P&L distribution.

## Data in this repo

Collected from Kalshi's public API for the KXBTC series, roughly **2026-07-24 to 2026-08-28**:

| File | Contents |
|---|---|
| `data/KXBTC_markets.parquet` | settled market metadata |
| `data/KXBTC_trades.parquet` | ~616k trade prints |
| `data/KXBTC_candles.parquet` | ~846k 1-minute candles |
| `data/ranked_data.parquet` | 344 markets ranked by estimated maker edge |
| `data/params.parquet` | fitted A–S parameters (k ≈ 0.033/¢, A ≈ 2.9 trades/hr at mid) |

The `.html` files in `plots/` are interactive Plotly outputs of the fits and backtest results.

## Layout

```
src/     scripts (run them from the repo root)
data/    collected market data and fitted parameters (parquet)
plots/   interactive Plotly outputs
```

## Running it

```bash
pip install polars numpy plotly requests tqdm
python src/collect_data.py   # or reuse the parquet files included here
python src/get_params.py
python src/backtest.py
```

## Limitations

- **No quote queue modeling** — a quote is assumed to fill whenever any trade prints through it. Real order books have queue position and priority, so actual fill rates (and P&L) would be lower.
- **In-sample fitting** — γ and the quoting cutoff were selected by a grid search on the same data they were evaluated on, with only a rough old/new date split as an out-of-sample check. One month of data leaves real overfitting risk.
- Quotes are static within each minute and reference candle-close mids, ignoring intraminute price movement.
- The 1.75% maker fee rate is hardcoded; Kalshi's fee schedule can change.
