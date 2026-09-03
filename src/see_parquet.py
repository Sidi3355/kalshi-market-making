import polars as pl

btc_trades = pl.read_parquet(source="data/KXBTC_trades.parquet")
btc_markets = pl.read_parquet(source="data/KXBTC_markets.parquet")
btc_candles = pl.read_parquet(source="data/KXBTC_candles.parquet")

print(btc_candles)