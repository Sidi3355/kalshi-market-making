import sys, time
from datetime import datetime
import polars as pl, requests
from tqdm import tqdm

BASE = "https://api.elections.kalshi.com/trade-api/v2"


def get(path, **params):
    while True:
        r = requests.get(BASE + path, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(1); continue
        r.raise_for_status()
        time.sleep(0.1)
        return r.json()


def pages(path, key, **params):
    cursor = None
    while True:
        d = get(path, limit=1000, cursor=cursor, **params)
        yield from d.get(key, [])
        cursor = d.get("cursor")
        if not cursor:
            return


def ts(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def flat(d, prefix=""):
    """Flatten nested dicts: {"price": {"close": 1}} -> {"price_close": 1}"""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}_"))
        else:
            out[prefix + k] = v
    return out


def collect(series_list, days=30, min_volume=50, out_dir="."):
    for series in series_list:
        markets = list(pages("/markets", "markets", series_ticker=series, status="settled",
                             min_close_ts=int(time.time()) - days * 86400))
        trades, candles = [], []
        for m in (bar := tqdm([m for m in markets if float(m["volume_fp"]) > min_volume], desc=series)):
            t = m["ticker"]
            bar.set_postfix_str(t)
            trades += [flat(dict(x, ticker=t)) for x in pages("/markets/trades", "trades", ticker=t)]
            start, end, step = ts(m["open_time"]), ts(m["close_time"]), 5000 * 60
            for a in range(start, end, step):
                d = get(f"/series/{series}/markets/{t}/candlesticks",
                        start_ts=a, end_ts=min(a + step - 60, end), period_interval=1)
                candles += [flat(dict(x, ticker=t)) for x in d.get("candlesticks", [])]
        pl.DataFrame([flat(m) for m in markets]).write_parquet(f"{out_dir}/{series}_markets.parquet")
        pl.DataFrame(trades).write_parquet(f"{out_dir}/{series}_trades.parquet")
        pl.DataFrame(candles).write_parquet(f"{out_dir}/{series}_candles.parquet")

# sorted trading volume per hour

trades = pl.read_parquet(source="KXBTC_trades.parquet").with_columns(
    pl.col("created_time").str.to_datetime()
)
per_market_trades = trades.group_by("ticker").agg(
    n=pl.len(),
    first=pl.col("created_time").min(),
    last=pl.col("created_time").max(),
    trades_per_hour=(pl.len() / ((pl.col("created_time").max() - pl.col("created_time").min()).dt.total_seconds() / 3600)),
)
per_market_trades = per_market_trades.filter(pl.col("n") > 20)
per_market_trades = per_market_trades.with_columns(
    tph_z=(pl.col("trades_per_hour") - pl.col("trades_per_hour").mean())
        / pl.col("trades_per_hour").std()
)

# print(per_market_trades.shape)
# print(per_market_trades.sort("trades_per_hour", descending=True))

# sorted median spread and mid-price volatility

candles = pl.read_parquet("KXBTC_candles.parquet").with_columns(
    pl.col("yes_bid_close_dollars").cast(pl.Float64),
    pl.col("yes_ask_close_dollars").cast(pl.Float64),
).filter(
    (pl.col("yes_bid_close_dollars") > 0)
    & (pl.col("yes_ask_close_dollars") < 1)
    & ((pl.col("yes_ask_close_dollars") - pl.col("yes_bid_close_dollars")) <= 0.10)
).with_columns(
    mid=(pl.col("yes_bid_close_dollars") + pl.col("yes_ask_close_dollars")) / 2
).sort("ticker", "end_period_ts")

per_market_candles = candles.group_by("ticker").agg(
    n_candles=pl.len(),
    med_spread=(pl.col("yes_ask_close_dollars") - pl.col("yes_bid_close_dollars")).median(),
    sigma_1m=pl.col("mid").diff().std(),
    avg_mid=pl.col("mid").mean(),
).with_columns(
    vol_vs_spread=pl.col("sigma_1m") / pl.col("med_spread"),
)



# print(per_market_candles.sort("med_spread", descending=True))

# post trade mid-drift

candles_mid = candles.select(
    "ticker",
    ts=pl.from_epoch("end_period_ts"),
    mid="mid",
).sort("ts")

trade_times = trades.select(
    "ticker", "taker_side",
    ts=pl.col("created_time").dt.replace_time_zone(None),
).sort("ts")

trades_with_mids = (
    trade_times.join_asof(candles_mid, on="ts", by="ticker")
    .rename({"mid": "mid_at_trade"})
    .with_columns(
        ts_plus_1m=pl.col("ts") + pl.duration(minutes=1),
        ts_plus_5m=pl.col("ts") + pl.duration(minutes=5),
    )
    .sort("ts_plus_1m")
    .join_asof(candles_mid.rename({"ts": "ts_plus_1m"}), on="ts_plus_1m", by="ticker")
    .rename({"mid": "mid_after_1m"})
    .sort("ts_plus_5m")
    .join_asof(candles_mid.rename({"ts": "ts_plus_5m"}), on="ts_plus_5m", by="ticker")
    .rename({"mid": "mid_after_5m"})
)

taker_sign = pl.when(pl.col("taker_side") == "yes").then(1).otherwise(-1)

per_market_drift = trades_with_mids.with_columns(
    drift_1m_signed=taker_sign * (pl.col("mid_after_1m") - pl.col("mid_at_trade")),
    drift_5m_signed=taker_sign * (pl.col("mid_after_5m") - pl.col("mid_at_trade")),
).group_by("ticker").agg(
    drift_1m=pl.col("drift_1m_signed").mean(),
    drift_5m=pl.col("drift_5m_signed").mean(),
    n_trades=pl.len(),
)

#summary df

summary = (
    per_market_trades
    .join(per_market_candles, on="ticker")
    .join(per_market_drift, on="ticker")
)
# print(summary.sort("drift_1m"))
# print(summary.columns)

# ranked df with fees introduced

MAKER_FEE_RATE = 0.0175  # set 0.0 only if your series is absent from the Maker Fees section

ranked = summary.with_columns(
    fee=MAKER_FEE_RATE * pl.col("avg_mid") * (1 - pl.col("avg_mid")),
).with_columns(
    edge_per_fill=pl.col("med_spread") / 2 - pl.col("fee") - pl.col("drift_1m"),
).filter(
    (pl.col("vol_vs_spread") < pl.col("vol_vs_spread").median() * 2)
    & (pl.col("n_candles") >= 100)
).with_columns(
    edge_flow=pl.col("edge_per_fill") * pl.col("trades_per_hour"),
).select(
    "ticker", "first", "last",
    "trades_per_hour", "med_spread", "vol_vs_spread", "drift_1m", "avg_mid",
    "fee", "edge_per_fill", "edge_flow",
).sort("edge_flow", descending=True)

print(ranked)

ranked.write_parquet(f"./ranked_data.parquet")