import io
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Nifty 500 Momentum Breakout Scanner", page_icon="📈", layout="wide")
st.title("📈 Nifty 500 Momentum & Breakout Scanner")
st.caption("Daily bullish momentum scan using EMA, MACD, ADX, volume, relative strength, RSI and Stochastic.")

CONSTITUENTS_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]
FALLBACK_FILE = "nifty500_symbols.csv"

@st.cache_data(ttl=86400, show_spinner=False)
def load_universe():
    errors = []
    for url in CONSTITUENTS_URLS:
        try:
            response = __import__("requests").get(
                url, timeout=20,
                headers={"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
            )
            response.raise_for_status()
            frame = pd.read_csv(io.BytesIO(response.content))
            symbol_col = next((c for c in frame.columns if c.strip().lower() in ("symbol", "symbol name")), None)
            if symbol_col is None:
                raise ValueError("CSV does not contain a Symbol column")
            symbols = sorted(set(frame[symbol_col].dropna().astype(str).str.strip().str.upper()))
            symbols = [s + ".NS" for s in symbols if s and s.lower() != "nan"]
            if len(symbols) < 450:
                raise ValueError(f"Only {len(symbols)} symbols found; refusing incomplete universe")
            pd.DataFrame({"Symbol": [s.removesuffix(".NS") for s in symbols]}).to_csv(FALLBACK_FILE, index=False)
            return symbols, f"Official Nifty Indices CSV ({len(symbols)} symbols)", None
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    # A local copy can be supplied alongside app.py to keep scanning during source outages.
    try:
        local = pd.read_csv(FALLBACK_FILE)
        col = "Symbol" if "Symbol" in local.columns else local.columns[0]
        symbols = sorted(set(local[col].dropna().astype(str).str.strip().str.upper()))
        symbols = [s if s.endswith(".NS") else s + ".NS" for s in symbols]
        if len(symbols) >= 450:
            return symbols, f"Bundled/local Nifty 500 list ({len(symbols)} symbols)", "Official list refresh failed; using local constituent snapshot."
    except Exception:
        pass
    return [], "Unavailable", "Could not load a complete Nifty 500 constituent list. " + " | ".join(errors)

def normalize_download(raw, symbol):
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance may arrange ticker at either level depending on version.
        if symbol in raw.columns.get_level_values(0):
            frame = raw[symbol].copy()
        elif symbol in raw.columns.get_level_values(-1):
            frame = raw.xs(symbol, axis=1, level=-1).copy()
        else:
            return pd.DataFrame()
    else:
        frame = raw.copy()
    frame.columns = [str(c).title() for c in frame.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in frame.columns for c in required):
        return pd.DataFrame()
    for c in required:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame[required].dropna(subset=["Close", "High", "Low"])

def compute_indicators(df, bench_close=None):
    out = df.copy()
    close, high, low, vol = out["Close"], out["High"], out["Low"], out["Volume"]
    out["EMA9"] = close.ewm(span=9, adjust=False).mean()
    out["EMA20"] = close.ewm(span=20, adjust=False).mean()
    out["EMA50"] = close.ewm(span=50, adjust=False).mean()
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    out["ATR14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=out.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=out.index)
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / out["ATR14"].replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / out["ATR14"].replace(0, np.nan)
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = (100 * (plus_di-minus_di).abs()/denom)
    out["ADX14"] = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12-ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"]-out["MACD_Signal"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI14"] = 100 - 100/(1+rs)
    out.loc[(loss == 0) & (gain > 0), "RSI14"] = 100
    out["RSI14"] = out["RSI14"].fillna(50)
    lo14, hi14 = low.rolling(14, min_periods=14).min(), high.rolling(14, min_periods=14).max()
    raw_k = 100*(close-lo14)/(hi14-lo14).replace(0, np.nan)
    out["Stoch_K"] = raw_k.rolling(3, min_periods=3).mean()
    out["Stoch_D"] = out["Stoch_K"].rolling(3, min_periods=3).mean()
    out["Breakout_Level"] = high.shift(1).rolling(20, min_periods=20).max()
    out["Vol_SMA20"] = vol.rolling(20, min_periods=20).mean()
    out["Vol_Mult"] = vol/out["Vol_SMA20"].replace(0, np.nan)
    stock_ret20 = (close/close.shift(20)-1)*100
    if bench_close is not None and not bench_close.empty:
        bench = bench_close.copy()
        bench.index = pd.to_datetime(bench.index).tz_localize(None) if getattr(bench.index, "tz", None) else pd.to_datetime(bench.index)
        idx = pd.to_datetime(out.index)
        idx = idx.tz_localize(None) if getattr(idx, "tz", None) else idx
        aligned = bench.reindex(idx).ffill()
        bench_ret20 = (aligned/aligned.shift(20)-1)*100
        out["RS_Pct"] = stock_ret20.to_numpy() - bench_ret20.to_numpy()
    else:
        out["RS_Pct"] = stock_ret20
    return out

def download_one(symbol, start, end):
    try:
        raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        return symbol, normalize_download(raw, symbol), None
    except Exception as exc:
        return symbol, pd.DataFrame(), str(exc)

def run_scan(symbols, lookback_days, min_vol_mult, min_rs_pct, min_adx, min_rsi, max_rsi, min_stoch_k, max_results, workers):
    end = datetime.now().date() + timedelta(days=1)
    start = end - timedelta(days=lookback_days)
    bench_raw = yf.download("^NSEI", start=start, end=end, auto_adjust=True, progress=False, threads=False)
    bench = normalize_download(bench_raw, "^NSEI")
    bench_close = bench["Close"] if not bench.empty else None
    rows, failures = [], []
    progress = st.progress(0)
    status = st.empty()
    total = len(symbols)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, s, start, end): s for s in symbols}
        for future in as_completed(futures):
            sym, df, err = future.result()
            done += 1
            if err:
                failures.append((sym, err))
            if not df.empty and len(df) >= 60:
                try:
                    ind = compute_indicators(df, bench_close)
                    latest = ind.iloc[-1]
                    vals = latest[["Close","EMA9","EMA20","EMA50","MACD","MACD_Hist","ADX14","Vol_Mult","RS_Pct","RSI14","Stoch_K","Breakout_Level","ATR14"]]
                    if vals.isna().any() or latest["ATR14"] <= 0:
                        pass
                    else:
                        passes = (
                            latest["Close"] > latest["EMA9"] and latest["EMA9"] > latest["EMA20"]
                            and latest["Close"] > latest["EMA50"] and latest["MACD"] > 0
                            and latest["MACD_Hist"] > 0 and latest["ADX14"] >= min_adx
                            and latest["Vol_Mult"] >= min_vol_mult and latest["RS_Pct"] >= min_rs_pct
                            and min_rsi <= latest["RSI14"] <= max_rsi
                            and latest["Stoch_K"] >= min_stoch_k
                            and latest["Close"] >= latest["Breakout_Level"] * 0.985
                        )
                        if passes:
                            close_px, bo, atr = float(latest["Close"]), float(latest["Breakout_Level"]), float(latest["ATR14"])
                            trigger = round(max(close_px, bo*1.002), 2)
                            stop = round(trigger-1.5*atr, 2)
                            risk = max(trigger-stop, 0.01)
                            score = round(0.35*(latest["RS_Pct"]/10)+0.30*(latest["Vol_Mult"]/2)+0.20*(latest["ADX14"]/30)+0.15*(latest["RSI14"]/70), 3)
                            rows.append({"Symbol":sym,"Close":round(close_px,2),"Breakout_Level":round(bo,2),"Trigger":trigger,"Stop_Loss":stop,"Target_1":round(trigger+2*risk,2),"Target_2":round(trigger+3*risk,2),"Vol_Mult":round(float(latest["Vol_Mult"]),2),"RS_%":round(float(latest["RS_Pct"]),2),"ADX14":round(float(latest["ADX14"]),1),"RSI14":round(float(latest["RSI14"]),1),"Stoch_K":round(float(latest["Stoch_K"]),1),"Score":score})
                except Exception as exc:
                    failures.append((sym, str(exc)))
            if done % 10 == 0 or done == total:
                progress.progress(done/total)
                status.caption(f"Processed {done} / {total} symbols · qualifying setups: {len(rows)}")
    return pd.DataFrame(rows).sort_values(["Score","RS_%","Vol_Mult"], ascending=False).head(max_results).reset_index(drop=True) if rows else pd.DataFrame(), done, failures

with st.sidebar:
    st.header("⚙️ Scan settings")
    lookback = st.slider("History (calendar days)", 120, 365, 220, 10)
    min_vol = st.number_input("Minimum volume multiple", 1.0, 5.0, 1.6, 0.1)
    min_rs = st.number_input("Minimum relative strength (%)", -10.0, 30.0, 4.0, 0.5)
    min_adx = st.number_input("Minimum ADX", 10.0, 60.0, 26.0, 1.0)
    min_rsi = st.number_input("Minimum RSI", 30.0, 80.0, 55.0, 1.0)
    max_rsi = st.number_input("Maximum RSI", 40.0, 100.0, 78.0, 1.0)
    min_stoch = st.number_input("Minimum Stochastic K", 30.0, 100.0, 65.0, 1.0)
    max_results = st.slider("Maximum results", 1, 50, 8)
    workers = st.slider("Parallel downloads", 1, 8, 4)

symbols, universe_source, universe_warning = load_universe()
if universe_warning:
    st.warning(universe_warning)
if not symbols:
    st.error("Nifty 500 universe could not be loaded. Add a valid `nifty500_symbols.csv` beside app.py (column: Symbol), then retry.")
    st.stop()

c1,c2,c3 = st.columns(3)
c1.metric("Universe", "Nifty 500")
c2.metric("Constituent symbols loaded", len(symbols))
c3.metric("Universe source", "Official / local list")
st.caption(f"Universe source: {universe_source}. The scan uses all loaded constituents, not the 12-symbol sample list in the original script.")
if st.button("🔄 Refresh constituent list", help="Clear cached list and reload the official Nifty 500 file"):
    load_universe.clear()
    st.rerun()

if st.button("🚀 Scan all Nifty 500 stocks", type="primary"):
    if min_rsi > max_rsi:
        st.error("Minimum RSI must be less than or equal to maximum RSI.")
    else:
        with st.spinner("Downloading daily market data and calculating indicators..."):
            results, scanned, failures = run_scan(symbols, lookback, min_vol, min_rs, min_adx, min_rsi, max_rsi, min_stoch, max_results, workers)
        st.session_state["scan_results"] = results
        st.session_state["scan_scanned"] = scanned
        st.session_state["scan_failures"] = failures
        st.session_state["scan_time"] = datetime.now().strftime("%d-%b-%Y %H:%M:%S")

if "scan_results" in st.session_state:
    results = st.session_state["scan_results"]
    st.subheader("Scan results")
    st.caption(f"Last scan: {st.session_state['scan_time']} · Processed: {st.session_state['scan_scanned']} / {len(symbols)} constituents · Matching setups: {len(results)}")
    if results.empty:
        st.info("No stocks currently meet every selected condition. Try again after the next completed daily candle or relax filters.")
    else:
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download results CSV", results.to_csv(index=False).encode("utf-8"), "nifty500_momentum_results.csv", "text/csv")
    failures = st.session_state.get("scan_failures", [])
    if failures:
        with st.expander(f"Data issues ({len(failures)} symbols)"):
            st.write("Some symbols may be unavailable or rate-limited by Yahoo Finance. They are counted as processed but not treated as passing.")
            st.dataframe(pd.DataFrame(failures, columns=["Symbol","Error"]), use_container_width=True, hide_index=True)

st.divider()
st.caption("Data source: Yahoo Finance (daily adjusted OHLCV). Signals are screening outputs, not guaranteed trades. Trigger, stop and targets are illustrative levels derived from the latest available daily candle and ATR.")
