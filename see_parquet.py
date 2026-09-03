import polars as pl

btc_trades = pl.read_parquet(source="KXBTC_trades.parquet")
btc_markets = pl.read_parquet(source="KXBTC_markets.parquet")
btc_candles = pl.read_parquet(source="KXBTC_candles.parquet")

print(btc_candles)