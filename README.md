# Nifty 500 Momentum & Breakout Scanner

## Deploy on Streamlit Community Cloud
1. Extract this ZIP.
2. Upload `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
3. In Streamlit Community Cloud, create an app and select `app.py` as the entry point.

## Universe
The app loads the official Nifty 500 constituent CSV and caches it for 24 hours. It displays the number of symbols loaded and refuses to scan if fewer than 450 constituents are returned. If the official source is unavailable, it can use `nifty500_symbols.csv` placed beside `app.py`.

## Strategy
Preserves the supplied filter logic: Close > EMA9 > EMA20, Close > EMA50, MACD > 0, MACD histogram > 0, ADX threshold, volume multiple, relative strength vs Nifty 50, RSI range, Stochastic K threshold, and price within 1.5% of the prior 20-day high.

Yahoo Finance may rate-limit or omit individual symbols. The app reports processed count and data issues. It does not guarantee all 500 have usable quotes on every run.
