# src/ml/backtester.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from feature_engineering import create_features, _ensure_time_index

MODEL_DEFAULT = "models/best_balanced_model.pkl"
CSV_DEFAULT = "data/processed/xauusd_m1_processed.csv"


def _load_bundle(model_path: Path):
    bundle = joblib.load(model_path)
    model = bundle["model"]
    calibrator = bundle.get("calibrator")
    thr_global = float(bundle["threshold"])
    ema_edges = bundle.get("ema200_slope_edges")
    atr_edges = bundle.get("atr_pctile_edges")
    thr_grid = bundle.get("thr_grid")
    return model, calibrator, thr_global, ema_edges, atr_edges, thr_grid


def _resolve_paths():
    here = Path(__file__).resolve()
    project_root = here.parents[2] if len(here.parents) >= 3 else here.parent.parent
    csv_env = os.environ.get("XAUUSD_CSV", "")
    csv_path = Path(csv_env) if csv_env else (project_root / CSV_DEFAULT)
    model_env = os.environ.get("MODEL_PATH", "")
    model_path = Path(model_env) if model_env else (project_root / MODEL_DEFAULT)
    return project_root, csv_path, model_path


def _prepare_data(csv_path: Path) -> pd.DataFrame:
    read_kwargs = dict(
        parse_dates=["timestamp"],
        dayfirst=False,
        dtype={
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "tick_volume": "float64",
        },
        engine="c",
    )
    df = pd.read_csv(csv_path, **read_kwargs)
    df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)
    if "timestamp" not in df.columns or not {"open", "high", "low", "close"}.issubset(df.columns):
        raise ValueError("CSV must contain timestamp, open, high, low, close")
    return df


def _compute_threshold_series(
    X: pd.DataFrame,
    proba: np.ndarray,
    thr_global: float,
    ema_edges,
    atr_edges,
    thr_grid,
):
    if ema_edges and atr_edges and thr_grid and "ema200_slope" in X and "atr_pctile" in X:
        e_bin = np.digitize(X["ema200_slope"].to_numpy(), np.array(ema_edges, dtype=float), right=True)
        a_bin = np.digitize(X["atr_pctile"].to_numpy(), np.array(atr_edges, dtype=float), right=True)
        thr_mat = np.array(thr_grid, dtype=float)
        thr = thr_mat[e_bin, a_bin]
    else:
        thr = np.full_like(proba, thr_global, dtype=float)
    return thr


def _atr_series(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / length, adjust=False).mean()
    return atr


def backtest(
    spread: float = 0.05,        # one-way cost in price units
    commission: float = 0.0,     # per-side fixed fee in price units
    atr_len: int = 14,
    atr_mult_tp: float = 2.0,
    atr_mult_sl: float = 2.0,
    time_barrier: int = 45,      # candles, match training
    max_concurrent: int = 1,     # positions at once
    backtest_start: str | None = None,  # optional ISO date
):
    project_root, csv_path, model_path = _resolve_paths()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not model_path.exists():
        raise SystemExit(f"Model bundle not found: {model_path}")
    print(f"Data:  {csv_path}")
    print(f"Model: {model_path}")

    # --- Data + features (past-only transforms) ---
    df_raw = _prepare_data(csv_path)
    dfi = _ensure_time_index(df_raw).copy().sort_index()
    X = create_features(dfi)
    px = dfi.loc[X.index, ["open", "high", "low", "close"]].copy()

    if backtest_start is not None:
        start_ts = pd.Timestamp(backtest_start, tz="UTC")
        mask = X.index >= start_ts
        X, px = X.loc[mask], px.loc[mask]

    # --- Predictions with saved calibrator and regime thresholds ---
    model, calibrator, thr_global, ema_edges, atr_edges, thr_grid = _load_bundle(model_path)
    proba = calibrator.predict_proba(X)[:, 1] if calibrator is not None else model.predict_proba(X)[:, 1]
    thr = _compute_threshold_series(X, proba, thr_global, ema_edges, atr_edges, thr_grid)
    sig = (proba >= thr).astype(np.int8)     # 1=buy, 0=sell
    side = np.where(sig == 1, 1, -1)         # +1/-1 for trading

    # --- No lookahead entry: act on next bar open ---
    side_shifted = pd.Series(side, index=X.index).shift(1).dropna().astype(int)
    px = px.loc[side_shifted.index]

    # ATR known at decision time: use ATR[t-1] for entry at t
    atr = _atr_series(dfi, atr_len).reindex(px.index)
    atr_entry = atr.shift(1).dropna()

    # Align all streams
    common = px.index.intersection(atr_entry.index).intersection(side_shifted.index)
    px = px.loc[common]
    atr_entry = atr_entry.loc[common]
    side_shifted = side_shifted.loc[common]

    # --- Simulation core (one position by default, FIFO) ---
    trades = []
    open_positions = []

    opens = px["open"].to_numpy()
    highs = px["high"].to_numpy()
    lows = px["low"].to_numpy()
    closes = px["close"].to_numpy()
    atrv = atr_entry.to_numpy()
    sides = side_shifted.to_numpy()
    times = px.index.to_numpy()

    n = len(px)
    for i in range(n - 1):
        # Update existing positions with bar i data
        still_open = []
        for pos in open_positions:
            j = pos["i_open"]
            i_now = i

            # Time barrier hit -> close at close[i]
            if i_now - j >= time_barrier:
                exit_price = closes[i]
                pnl = (exit_price - pos["entry_price"]) * pos["side"]
                pnl -= spread + 2 * commission
                trades.append(
                    {
                        **{k: pos[k] for k in ["entry_time", "entry_price", "side", "atr", "tp", "sl", "i_open"]},
                        "exit_time": times[i],
                        "exit_price": exit_price,
                        "bars_held": i_now - j,
                        "reason": "time",
                        "pnl": pnl,
                    }
                )
                continue

            # Barrier checks within current bar
            hit_tp = (highs[i] >= pos["tp"]) if pos["side"] > 0 else (lows[i] <= pos["tp"])
            hit_sl = (lows[i] <= pos["sl"]) if pos["side"] > 0 else (highs[i] >= pos["sl"])

            if hit_tp and not hit_sl:
                exit_price = pos["tp"]
                reason = "tp"
            elif hit_sl and not hit_tp:
                exit_price = pos["sl"]
                reason = "sl"
            elif hit_tp and hit_sl:
                # Tie-break by closer barrier distance (consistent with labeling)
                d_tp = abs(pos["tp"] - pos["entry_price"])
                d_sl = abs(pos["entry_price"] - pos["sl"])
                if d_tp <= d_sl:
                    exit_price, reason = pos["tp"], "tp_tie"
                else:
                    exit_price, reason = pos["sl"], "sl_tie"
            else:
                still_open.append(pos)
                continue

            pnl = (exit_price - pos["entry_price"]) * pos["side"]
            pnl -= spread + 2 * commission
            trades.append(
                {
                    **{k: pos[k] for k in ["entry_time", "entry_price", "side", "atr", "tp", "sl", "i_open"]},
                    "exit_time": times[i],
                    "exit_price": exit_price,
                    "bars_held": i_now - j,
                    "reason": reason,
                    "pnl": pnl,
                }
            )

        open_positions = still_open

        # Open new position at this bar if capacity allows
        if len(open_positions) < max_concurrent:
            s = sides[i]
            atr_i = atrv[i]
            if not np.isfinite(atr_i) or atr_i <= 0:
                continue
            entry_price = opens[i]  # next-bar open already enforced by shift
            if s > 0:
                tp = entry_price + atr_mult_tp * atr_i
                sl = entry_price - atr_mult_sl * atr_i
            else:
                tp = entry_price - atr_mult_tp * atr_i
                sl = entry_price + atr_mult_sl * atr_i

            open_positions.append(
                {
                    "i_open": i,
                    "entry_time": times[i],
                    "entry_price": entry_price,
                    "side": int(s),
                    "atr": float(atr_i),
                    "tp": float(tp),
                    "sl": float(sl),
                }
            )

    # Force-close remaining positions at final close
    last_i = n - 1
    for pos in open_positions:
        exit_price = closes[last_i]
        pnl = (exit_price - pos["entry_price"]) * pos["side"]
        pnl -= spread + 2 * commission
        trades.append(
            {
                **{k: pos[k] for k in ["entry_time", "entry_price", "side", "atr", "tp", "sl", "i_open"]},
                "exit_time": times[last_i],
                "exit_price": exit_price,
                "bars_held": last_i - pos["i_open"],
                "reason": "eod",
                "pnl": pnl,
            }
        )

    if len(trades) == 0:
        print("No trades executed.")
        return None

    # --- Performance summary ---
    trades_df = pd.DataFrame(trades)
    trades_df["r_multiple"] = trades_df["pnl"] / (atr_mult_sl * trades_df["atr"])
    trades_df["win"] = (trades_df["pnl"] > 0).astype(int)
    trades_df["side_name"] = trades_df["side"].map({1: "BUY", -1: "SELL"})

    equity = trades_df["pnl"].cumsum()
    dd = equity - equity.cummax()
    max_dd = dd.min()

    win_rate = trades_df["win"].mean() if len(trades_df) else 0.0
    avg_r = trades_df["r_multiple"].mean()
    total_pnl = trades_df["pnl"].sum()

    side_stats = trades_df.groupby("side_name").agg(
        n=("win", "count"),
        win_rate=("win", "mean"),
        avg_r=("r_multiple", "mean"),
        pnl=("pnl", "sum"),
    )

    print("\n--- Backtest Performance ---")
    print(f"Trades: {len(trades_df)} | Win rate: {win_rate*100:.2f}% | Avg R: {avg_r:.3f}")
    print(f"Total PnL (price units): {total_pnl:.2f} | Max Drawdown: {max_dd:.2f}")
    print("\nBy side:")
    print(side_stats)

    # Save
    project_root, _, _ = _resolve_paths()
    results_dir = project_root / "backtest_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    trades_path = results_dir / "trades.csv"
    equity_path = results_dir / "equity.csv"
    trades_df.to_csv(trades_path, index=False)
    pd.DataFrame({"timestamp": trades_df["exit_time"], "equity": equity}).to_csv(equity_path, index=False)

    print(f"\nSaved trades to:  {trades_path}")
    print(f"Saved equity to:  {equity_path}")
    return trades_df


if __name__ == "__main__":
    backtest()
