# feature_engineering.py
import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except Exception:
    ta = None


# ---------------- LABELING: symmetric triple-barrier ----------------
def get_triple_barrier_labels(
    df: pd.DataFrame,
    atr_len: int = 14,
    atr_mult_tp: float = 2.0,
    atr_mult_sl: float = 2.0,
    time_barrier: int = 30,           # candles
    price_col: str = "close",
) -> pd.Series:
    """
    Output in {-1,0,+1} on the SAME DatetimeIndex as df.
    0 means neither barrier hit within time_barrier.
    """
    assert {"high", "low", price_col}.issubset(df.columns), "DF must have high, low, close"

    # ATR
    if ta is not None:
        atr = df.ta.atr(length=atr_len)
    else:
        prev_close = df[price_col].shift(1)
        tr = pd.concat(
            [(df["high"] - df["low"]),
             (df["high"] - prev_close).abs(),
             (df["low"] - prev_close).abs()],
            axis=1
        ).max(axis=1)
        atr = tr.ewm(alpha=1 / atr_len, adjust=False).mean()

    c = df[price_col].to_numpy()
    a = atr.to_numpy()
    tp = c + atr_mult_tp * a
    sl = c - atr_mult_sl * a

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    labels = np.zeros(n, dtype=np.int8)
    end_idx = np.minimum(np.arange(n) + time_barrier, n - 1)

    for i in range(n):
        j = end_idx[i]
        lo = i + 1
        if lo > j:
            continue
        wh = highs[lo : j + 1].max()
        wl = lows[lo : j + 1].min()
        hit_tp = wh >= tp[i]
        hit_sl = wl <= sl[i]
        if hit_tp and not hit_sl:
            labels[i] = 1
        elif hit_sl and not hit_tp:
            labels[i] = -1
        elif hit_tp and hit_sl:
            # tie-break by closer barrier distance (keeps symmetry)
            d_tp = tp[i] - c[i]
            d_sl = c[i] - sl[i]
            labels[i] = 1 if d_tp < d_sl else -1
        else:
            labels[i] = 0

    return pd.Series(labels, index=df.index, name="label")


# ---------------- FEATURES: symmetric, volume-aware, leakage-safe ----------------
def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Set a UTC DatetimeIndex from 'timestamp' or 'time' if not already set."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df.copy()
    col = "timestamp" if "timestamp" in out.columns else ("time" if "time" in out.columns else None)
    if col is None:
        return out
    out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    out = out.dropna(subset=[col]).set_index(col).sort_index()
    return out


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create symmetric features. Returns float32 matrix aligned to df.index with no NaNs.
    Requires columns: open, high, low, close. Uses 'tick_volume' if present.
    """
    assert {"open", "high", "low", "close"}.issubset(df.columns), "DF must have OHLC"
    df = _ensure_time_index(df).copy()

    # unify volume
    if "volume" not in df.columns and "tick_volume" in df.columns:
        df["volume"] = df["tick_volume"].astype(float)

    # Core TA
    if ta is not None:
        if "EMA_200" not in df.columns:
            df["EMA_200"] = df.ta.ema(length=200)
        df["RSI_7"] = df.ta.rsi(length=7)
        df["RSI_14"] = df.ta.rsi(length=14)
        stoch = df.ta.stoch(k=7, d=3, smooth_k=3)
        if stoch is not None:
            df = pd.concat([df, stoch], axis=1)
        df["ATRr_14"] = df.ta.atr(length=14)
        if "EMA_50" not in df.columns:
            df["EMA_50"] = df.ta.ema(length=50)
    else:
        df["EMA_200"] = df["close"].ewm(span=200, adjust=False).mean()
        d = df["close"].diff()
        up7 = (d.clip(lower=0)).ewm(alpha=1 / 7, adjust=False).mean()
        dn7 = (-d.clip(upper=0)).ewm(alpha=1 / 7, adjust=False).mean()
        rs7 = up7 / (dn7 + 1e-12)
        df["RSI_7"] = 100 - (100 / (1 + rs7))
        up14 = (d.clip(lower=0)).ewm(alpha=1 / 14, adjust=False).mean()
        dn14 = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs14 = up14 / (dn14 + 1e-12)
        df["RSI_14"] = 100 - (100 / (1 + rs14))
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [(df["high"] - df["low"]),
             (df["high"] - prev_close).abs(),
             (df["low"] - prev_close).abs()],
            axis=1
        ).max(axis=1)
        df["ATRr_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
        low_n = df["low"].rolling(7).min()
        high_n = df["high"].rolling(7).max()
        df["STOCHk_7_3_3"] = 100 * (df["close"] - low_n) / (high_n - low_n + 1e-12)
        df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # Symmetric distances to trend
    df["dist_ema200"] = (df["close"] / df["EMA_200"]) - 1.0
    df["dist_ema50"] = (df["close"] / df["EMA_50"]) - 1.0

    # Returns and volatility
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_15"] = df["close"].pct_change(15)
    df["hl_range"] = (df["high"] - df["low"]) / (df["close"].shift(1) + 1e-12)

    # Candle shape
    df["body"] = (df["close"] - df["open"]) / (df["open"].shift(1) + 1e-12)
    df["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (df["open"].shift(1) + 1e-12)
    df["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (df["open"].shift(1) + 1e-12)

    # Breakout flags
    df["hh_20"] = (df["high"] >= df["high"].rolling(20).max()).astype(np.int8)
    df["ll_20"] = (df["low"] <= df["low"].rolling(20).min()).astype(np.int8)

    # Time-of-day (UTC)
    t = df.index
    sec = (t.hour * 3600 + t.minute * 60 + t.second).astype(float)
    df["tod_sin"] = np.sin(2 * np.pi * sec / 86400.0)
    df["tod_cos"] = np.cos(2 * np.pi * sec / 86400.0)

    # Day-of-week
    dow = t.weekday.astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # Volume imbalance
    if "volume" in df.columns:
        v = df["volume"].astype(float)
        df["vol_imb_5"] = (v - v.rolling(5).mean()) / (v.rolling(5).std() + 1e-12)
        df["vol_imb_20"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-12)

    # Rename compactly
    df.rename(
        columns={
            "RSI_7": "rsi_7",
            "RSI_14": "rsi_14",
            "STOCHk_7_3_3": "stoch_k",
            "STOCHd_7_3_3": "stoch_d",
            "ATRr_14": "atr",
            "EMA_200": "ema_200",
            "EMA_50": "ema_50",
        },
        inplace=True,
    )

    # Post-rename features depending on renamed columns
    df["ema200_slope"] = df["ema_200"].pct_change(5)  # 5-min slope
    atr_rolling = df["atr"].rolling(1440, min_periods=60)
    df["atr_pctile"] = atr_rolling.rank(pct=True)

    keep_cols = [
        "rsi_7", "rsi_14", "stoch_k", "atr", "ema_200",
        "dist_ema200", "dist_ema50", "ema200_slope", "atr_pctile",
        "ret_1", "ret_5", "ret_15", "hl_range",
        "body", "upper_wick", "lower_wick",
        "hh_20", "ll_20", "tod_sin", "tod_cos", "dow_sin", "dow_cos",
        "vol_imb_5", "vol_imb_20",
    ]
    X = df[[c for c in keep_cols if c in df.columns]].copy()

    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    X = X.astype(np.float32)

    print(f"Feature engineering complete. Shape of feature set: {X.shape}")
    return X
