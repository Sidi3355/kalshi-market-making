import polars as pl, requests
import numpy as np
import plotly.graph_objects as go

ranked = pl.read_parquet("data/ranked_data.parquet")
top_ticker = ranked["ticker"][0]

# sigma (volatility)

candles = pl.read_parquet("data/KXBTC_candles.parquet").with_columns(
    pl.col("yes_bid_close_dollars").cast(pl.Float64),
    pl.col("yes_ask_close_dollars").cast(pl.Float64),
).filter(
    (pl.col("yes_bid_close_dollars") > 0)
    & (pl.col("yes_ask_close_dollars") < 1)
    & ((pl.col("yes_ask_close_dollars") - pl.col("yes_bid_close_dollars")) <= 0.10)
).with_columns(
    mid=(pl.col("yes_bid_close_dollars") + pl.col("yes_ask_close_dollars")) / 2
)

sigma = (
    candles
    .filter(pl.col("ticker") == top_ticker)
    .sort("end_period_ts")
    .with_columns(sigma_1m=pl.col("mid").diff().rolling_std(window_size=60))
    .select("end_period_ts", "mid", "sigma_1m")
)

# print(sigma[:62])   

fig = go.Figure()
fig.add_scatter(x=sigma["end_period_ts"], y=sigma["sigma_1m"], mode="lines", name="sigma")
fig.add_scatter(x=sigma["end_period_ts"], y=sigma["mid"], mode="lines", name="mid", yaxis="y2")
fig.update_layout(
    yaxis_title="sigma (dollars/min)",
    yaxis2=dict(title="mid", overlaying="y", side="right"),
    xaxis_title="time",
)

#fig.write_html("plots/sigma.html", auto_open=True)        

# A (trades per hour at mid-price) and k (decay rate from mid-price), modelled together exponentially

trades = pl.read_parquet("data/KXBTC_trades.parquet").with_columns(
    pl.col("created_time").str.to_datetime()
)

candles_mid = candles.filter(pl.col("ticker") == top_ticker).select(
    ts=pl.from_epoch("end_period_ts"),
    mid="mid",
).sort("ts")

trade_dist = (
    trades.filter(pl.col("ticker") == top_ticker)
    .select(
        ts=pl.col("created_time").dt.replace_time_zone(None),
        price=pl.col("yes_price_dollars").cast(pl.Float64),
    )
    .sort("ts")
    .join_asof(candles_mid, on="ts")
    .drop_nulls()
    .with_columns(distance=((pl.col("price") - pl.col("mid")).abs() * 100).round(0))
)

hours = (
    trade_dist["ts"].max() - trade_dist["ts"].min()
).total_seconds() / 3600

buckets = (
    trade_dist.group_by("distance").agg(count=pl.len())
    .filter(pl.col("count") >= 5)
    .with_columns(rate=pl.col("count") / hours)
    .sort("distance")
)
#print(buckets)

d = buckets["distance"].to_numpy()
log_rate = np.log(buckets["rate"].to_numpy())
k_slope, log_A = np.polyfit(d, log_rate, 1)
k, A = -k_slope, np.exp(log_A)
#print(f"k = {k:.3f} per cent, A = {A:.1f} trades/hour at mid")

fig = go.Figure()
fig.add_scatter(x=d, y=log_rate, mode="markers", name="buckets")
fig.add_scatter(x=d, y=k_slope * d + log_A, mode="lines", name=f"fit: k={k:.2f}")
fig.update_layout(xaxis_title="distance from mid (cents)",
                  yaxis_title="log trades/hour")

#fig.write_html("plots/k_fit.html", auto_open=True)

params = pl.DataFrame({
    "ticker": [top_ticker],
    "k": [k],
    "A": [A],
    "sigma_med": [sigma["sigma_1m"].drop_nulls().median()],
})
params.write_parquet("data/params.parquet")