import polars as pl
import numpy as np
import math
from datetime import datetime
import plotly.graph_objects as go

candles = pl.read_parquet("data/KXBTC_candles.parquet").with_columns(
    pl.col("yes_bid_close_dollars").cast(pl.Float64),
    pl.col("yes_ask_close_dollars").cast(pl.Float64),
).filter(
    (pl.col("yes_bid_close_dollars") > 0)
    & (pl.col("yes_ask_close_dollars") < 1)
    & ((pl.col("yes_ask_close_dollars") - pl.col("yes_bid_close_dollars")) <= 0.10)
).with_columns(
    mid=(pl.col("yes_bid_close_dollars") + pl.col("yes_ask_close_dollars")) / 2
).sort("ticker", "end_period_ts")

trades = pl.read_parquet("data/KXBTC_trades.parquet").with_columns(
    pl.col("created_time").str.to_datetime()
)

markets = pl.read_parquet("data/KXBTC_markets.parquet").with_columns(
    close_ts=pl.col("close_time").str.to_datetime().dt.replace_time_zone(None).dt.epoch("s")
)

ranked = pl.read_parquet("data/ranked_data.parquet")
top_ticker = ranked["ticker"][0]
params = pl.read_parquet("data/params.parquet")
k_fit = params.filter(pl.col("ticker") == top_ticker)["k"][0] * 100


def quote(mid, q, sigma, k, gamma, mins_left,
          cutoff=90, max_inv=100, tick=0.01):
    if sigma is None or mins_left <= cutoff:
        return None

    var_t = gamma * sigma**2 * mins_left
    r = mid - q * var_t
    delta = var_t + (2 / gamma) * math.log(1 + gamma / k)
    delta = min(delta, 0.04)

    bid = min(mid - tick, r - delta / 2)
    ask = max(mid + tick, r + delta / 2)

    bid = max(0.01, min(0.99, round(bid / tick) * tick))
    ask = max(0.01, min(0.99, round(ask / tick) * tick))
    if ask <= bid:
        return None

    if q >= max_inv:
        bid = None
    if q <= -max_inv:
        ask = None
    return bid, ask


def replay(ticker, gamma, cutoff=30, max_inv=100, size=1, fee_rate=0.0175):
    m = markets.filter(pl.col("ticker") == ticker)
    close_ts = m["close_ts"][0]
    result = 1.0 if str(m["result"][0]).lower() == "yes" else 0.0

    c = (candles.filter(pl.col("ticker") == ticker)
         .sort("end_period_ts")
         .with_columns(sigma_1m=pl.col("mid").diff().rolling_std(window_size=60)))

    t = (trades.filter(pl.col("ticker") == ticker)
         .with_columns(
             minute=pl.col("created_time").dt.replace_time_zone(None).dt.epoch("s") // 60 * 60,
             price=pl.col("yes_price_dollars").cast(pl.Float64),
         ))
    trades_by_minute = {key[0]: g for key, g in t.group_by("minute")}

    q, cash, n_fills, max_q = 0, 0.0, 0, 0

    rows = c.iter_rows(named=True)
    prev = next(rows)
    for row in rows:
        mins_left = (close_ts - prev["end_period_ts"]) / 60
        quotes = quote(prev["mid"], q, prev["sigma_1m"], k_fit, gamma,
                       mins_left, cutoff, max_inv)
        if quotes:
            bid, ask = quotes
            minute_trades = trades_by_minute.get(prev["end_period_ts"] )
            if minute_trades is not None:
                buys = sells = 0
                for tr in minute_trades.iter_rows(named=True):
                    fee = fee_rate * tr["price"] * (1 - tr["price"])
                    if bid is not None and buys < 2 and tr["price"] <= bid:
                        q += size; cash -= bid * size + fee; n_fills += 1; buys += 1
                    if ask is not None and sells < 2 and tr["price"] >= ask:
                        q -= size; cash += ask * size - fee; n_fills += 1; sells += 1
                    max_q = max(max_q, abs(q))
        prev = row

    cash += q * result
    return {"ticker": ticker, "pnl": round(cash, 4), "fills": n_fills,
            "max_q": max_q, "end_q": q}


# all-markets backtest

results = pl.DataFrame([replay(t, gamma=0.1) for t in ranked["ticker"]])
#print(results.sort("pnl"))
#print(results.select(pl.col("pnl").sum(), pl.col("fills").sum(), pl.col("max_q").max()))

# find optimal params

# 1. extend the grid where it was still rising
def test_params():
    for gamma in (0.1, 0.3):
        for cutoff in (90, 120):
            r = pl.DataFrame([replay(t, gamma=gamma, cutoff=cutoff) for t in ranked["ticker"]])
            print({"gamma": gamma, "cutoff": cutoff,
                "pnl": round(r["pnl"].sum(), 2),
                "worst_mkt": round(r["pnl"].min(), 2),
                "worst_q": r["max_q"].max()})

# 2. out-of-sample split for the leading cells
dated = ranked.with_columns(
    date=pl.col("ticker").str.extract(r"-(\d{2}[A-Z]{3}\d{2})").str.to_date("%y%b%d", strict=False)
).sort("date")
half = dated.height // 2
old = dated["ticker"][:half].to_list()
new = dated["ticker"][half:].to_list()

def out_of_sample_test():

    for gamma, cutoff in ((0.03, 60), (0.01, 60), (0.1, 90)):
        for label, group in (("old", old), ("new", new)):
            r = pl.DataFrame([replay(t, gamma=gamma, cutoff=cutoff) for t in group])
            #print(gamma, cutoff, label, round(r["pnl"].sum(), 2), r["fills"].sum())



# final backtesting

results = pl.DataFrame([replay(t, gamma=0.1, cutoff=90) for t in ranked["ticker"]])
print(results.sort("pnl"))
print(results.select(pl.col("pnl").sum(), pl.col("fills").sum(), pl.col("max_q").max()))

total = results["pnl"].sum()
n_pos = (results["pnl"] > 0).sum()
avg_fill = total / results["fills"].sum()

# draw two graphs

# --- histogram: pnl per market ---
fig = go.Figure(go.Histogram(
    x=results["pnl"], nbinsx=50,
    marker=dict(color="#3b6ea5", line=dict(color="white", width=0.5)),
    hovertemplate="pnl %{x:$.2f}<br>%{y} markets<extra></extra>",
))
fig.add_vline(x=0, line_dash="dot", line_color="grey")
fig.update_layout(
    template="plotly_white",
    title=dict(
        text=f"P&L per market — KXBTC, γ=0.1, cutoff=90 min, size 1"
             f"<br><sup>{results.height} markets · total ${total:.2f} · "
             f"{n_pos} profitable · {avg_fill*100:.2f}¢ per fill</sup>",
    ),
    xaxis_title="P&L per market ($)",
    yaxis_title="number of markets",
    xaxis_tickprefix="$",
    bargap=0.02,
)
fig.write_html("plots/pnl_hist.html", auto_open=True)

# --- equity curve: cumulative pnl by market date ---
dated = results.with_columns(
    date=pl.col("ticker").str.extract(r"-(\d{2}[A-Z]{3}\d{2})").str.to_date("%y%b%d", strict=False)
).sort("date").with_columns(cum_pnl=pl.col("pnl").cum_sum())

peak = dated["cum_pnl"].cum_max()
drawdown = (dated["cum_pnl"] - peak).min()

fig = go.Figure()
fig.add_scatter(
    x=dated["date"], y=dated["cum_pnl"], mode="lines",
    line=dict(color="#3b6ea5", width=2),
    fill="tozeroy", fillcolor="rgba(59,110,165,0.08)",
    hovertemplate="%{x|%d %b}<br>cumulative %{y:$.2f}<extra></extra>",
)
fig.add_hline(y=0, line_dash="dot", line_color="grey")
fig.update_layout(
    template="plotly_white",
    title=dict(
        text="Cumulative backtest P&L — KXBTC hourly markets"
             f"<br><sup>γ=0.1 · quoting stops 90 min pre-close · "
             f"max drawdown ${drawdown:.2f} · max inventory {results['max_q'].max()}</sup>",
    ),
    xaxis_title="market date",
    yaxis_title="cumulative P&L ($)",
    yaxis_tickprefix="$",
)
fig.write_html("plots/pnl_curve.html", auto_open=True)

initial_capital = 100.0  # working capital assumption; size-1 quoting rarely uses more

total = results["pnl"].sum()
wins = results.filter(pl.col("pnl") > 0)
losses = results.filter(pl.col("pnl") < 0)
dd = (dated["cum_pnl"] - dated["cum_pnl"].cum_max()).min()

print(f"""
================ BACKTEST SUMMARY ================
Series:              KXBTC (hourly BTC markets)
Period:              {dated["date"].min()} to {dated["date"].max()}
Markets traded:      {results.height}
Model:               Avellaneda-Stoikov, delta cap 4c
Parameters:          gamma=0.1, cutoff=90 min, k={k_fit:.1f}/$
Assumptions:         size 1, max 2 fills/min/side, maker fee 1.75% rate
                     fills = any print crossing quote (optimistic)

Starting capital:    ${initial_capital:.2f}
Final capital:       ${initial_capital + total:.2f}
Total P&L:           ${total:.2f}  ({total / initial_capital * 100:+.1f}%)
Max drawdown:        ${dd:.2f}  ({dd / initial_capital * 100:.1f}%)

Fills:               {results["fills"].sum()}
P&L per fill:        {total / results["fills"].sum() * 100:.3f}c
Profitable markets:  {wins.height} ({wins.height / results.height * 100:.0f}%)
Losing markets:      {losses.height}
Avg win / avg loss:  ${wins["pnl"].mean():.2f} / ${losses["pnl"].mean():.2f}
Best market:         {results.sort("pnl", descending=True)["ticker"][0]}  ${results["pnl"].max():.2f}
Worst market:        {results.sort("pnl")["ticker"][0]}  ${results["pnl"].min():.2f}
Max inventory:       {results["max_q"].max()} contracts
Avg fills/market:    {results["fills"].mean():.0f}
==================================================
""")