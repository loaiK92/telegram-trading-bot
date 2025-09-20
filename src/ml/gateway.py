# gateway.py
import os
from pathlib import Path
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

from feature_engineering import create_features, _ensure_time_index

# -------- config --------
here = Path(__file__).resolve()
project_root = here.parents[2] if len(here.parents) >= 3 else here.parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", project_root / "models" / "best_balanced_model.pkl"))
MIN_BARS = 2000   # enough for EMA200 and ATR percentile stability

# -------- load model bundle --------
bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
CAL = bundle.get("calibrator", None)
THR_GLOBAL = float(bundle["threshold"])
E_EDGES = bundle.get("ema200_slope_edges", None)
A_EDGES = bundle.get("atr_pctile_edges", None)
THR_GRID = bundle.get("thr_grid", None)

app = Flask(__name__)

@app.get("/ping")
def ping():
    return {"ok": True}

def _predict_one(df: pd.DataFrame) -> Dict:
    """Compute features on provided history, then return last-bar decision to be executed on next bar open."""
    dfi = _ensure_time_index(df)
    X = create_features(dfi)
    if X.empty:
        return {"ok": False, "error": "insufficient_bars_for_features"}

    # last fully-computed row
    x_last = X.iloc[[-1]]
    if CAL is not None:
        proba = CAL.predict_proba(X)[:, 1]
    else:
        proba = MODEL.predict_proba(X)[:, 1]
    p_last = float(proba[-1])

    # regime threshold
    if (E_EDGES and A_EDGES and THR_GRID
        and "ema200_slope" in X.columns and "atr_pctile" in X.columns):
        e_bin = np.digitize(x_last["ema200_slope"].to_numpy(), np.array(E_EDGES, dtype=float), right=True)[0]
        a_bin = np.digitize(x_last["atr_pctile"].to_numpy(),   np.array(A_EDGES, dtype=float), right=True)[0]
        thr_mat = np.array(THR_GRID, dtype=float)
        thr = float(thr_mat[e_bin, a_bin])
    else:
        thr = THR_GLOBAL

    signal = 1 if p_last >= thr else 0  # 1=BUY, 0=SELL
    ts = x_last.index[-1].isoformat()

    return {"ok": True, "timestamp": ts, "proba": p_last, "threshold": thr, "signal": int(signal)}

@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON input:
    {
      "bars": [
        {"ts": 1704067200, "open":..., "high":..., "low":..., "close":..., "tick_volume": ...},
        ...
      ]
    }
    Send only CLOSED bars. Do NOT include the forming bar.
    """
    body = request.get_json(force=True, silent=True)
    if not body or "bars" not in body:
        return jsonify({"ok": False, "error": "missing_bars"}), 400
    bars: List[Dict] = body["bars"]
    if len(bars) < MIN_BARS:
        return jsonify({"ok": False, "error": f"need_at_least_{MIN_BARS}_bars"}), 400

    df = pd.DataFrame(bars)
    # normalize columns
    cols_map = {
        "ts": "timestamp", "time": "timestamp",
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "tick_volume"
    }
    df.rename(columns=cols_map, inplace=True)
    for c in ["open", "high", "low", "close", "tick_volume"]:
        if c not in df.columns:
            return jsonify({"ok": False, "error": f"missing_{c}"}), 400

    # parse epoch seconds to UTC
    if np.issubdtype(df["timestamp"].dtype, np.number):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    out = _predict_one(df)
    code = 200 if out.get("ok") else 422
    return jsonify(out), code

if __name__ == "__main__":
    print(f"Loaded model: {MODEL_PATH}")
    app.run(host="127.0.0.1", port=3000, debug=False)
