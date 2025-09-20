# model_training.py
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    fbeta_score,
    confusion_matrix,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

from feature_engineering import create_features, get_triple_barrier_labels


def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df.copy()
    col = "timestamp" if "timestamp" in out.columns else ("time" if "time" in out.columns else None)
    if col is None:
        raise ValueError("No 'timestamp' or 'time' column found.")
    out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    out = out.dropna(subset=[col]).set_index(col).sort_index()
    return out


def _align_Xy(X: pd.DataFrame, y: pd.Series):
    common = X.index.intersection(y.index)
    return X.loc[common], y.loc[common]


def _regime_thresholds(valid_proba, y_valid, X_valid, global_t, sell_floor=0.93):
    ema = X_valid["ema200_slope"].to_numpy()
    atrp = X_valid["atr_pctile"].to_numpy()

    e_edges = np.quantile(ema, [1 / 3, 2 / 3]).astype(float)
    a_edges = np.quantile(atrp, [1 / 3, 2 / 3]).astype(float)

    e_bin = np.digitize(ema, e_edges, right=True)  # 0..2
    a_bin = np.digitize(atrp, a_edges, right=True)  # 0..2

    grid = np.concatenate([np.linspace(0.30, 0.70, 81), np.linspace(0.05, 0.95, 19)])
    grid = np.unique(np.clip(grid, 0.01, 0.99))

    thr = np.full((3, 3), np.nan, dtype=float)

    for i in range(3):
        for j in range(3):
            m = (e_bin == i) & (a_bin == j)
            if m.sum() < 2000:
                thr[i, j] = global_t
                continue
            best, best_t = -1.0, global_t
            yv = y_valid[m]
            pv = valid_proba[m]
            for t in grid:
                yp = (pv >= t).astype(int)
                _, rec, _, _ = precision_recall_fscore_support(
                    yv, yp, average=None, labels=[0, 1], zero_division=0
                )
                sell_recall = rec[0]
                buy_f2 = fbeta_score(yv, yp, beta=2.0, pos_label=1, zero_division=0)
                if sell_recall >= sell_floor and buy_f2 > best:
                    best, best_t = buy_f2, float(t)
            thr[i, j] = best_t
    return e_edges.tolist(), a_edges.tolist(), thr.tolist()


def train_lightgbm_binary(
    X: pd.DataFrame,
    y: pd.Series,
    valid_size: float = 0.2,
    random_state: int = 42,
    keep_sell_recall: float = 0.93,
):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=valid_size, random_state=random_state, stratify=y
    )

    # class weights
    neg, pos = np.bincount(y_train)
    pos_weight = float(neg) / float(max(pos, 1))
    pos_weight *= 1.8  # boost buys slightly

    clf = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=6000,
        learning_rate=0.03,
        num_leaves=127,
        min_data_in_leaf=150,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        class_weight={0: 1.0, 1: pos_weight},
        n_jobs=-1,
        random_state=random_state,
    )

    # custom eval that tracks buy AP (PR-AUC)
    def _ap_buy_eval(y_true, y_pred_raw):
        p = 1.0 / (1.0 + np.exp(-y_pred_raw))
        ap = average_precision_score(y_true, p)
        return "ap_buy", ap, True

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=lambda y_true, y_pred: [_ap_buy_eval(y_true, y_pred)],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(50)],
    )

    # probability calibration on validation
    cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
    cal.fit(X_valid, y_valid)
    valid_proba = cal.predict_proba(X_valid)[:, 1]

    # threshold: maximize F2 for buys with sell recall floor
    grid = np.concatenate([
        np.linspace(0.30, 0.70, 81),
        np.linspace(0.05, 0.95, 19),
    ])
    grid = np.unique(np.clip(grid, 0.01, 0.99))

    best_t, best_score = 0.5, -1.0
    for t in grid:
        y_pred = (valid_proba >= t).astype(int)
        _, rec, _, _ = precision_recall_fscore_support(
            y_valid, y_pred, average=None, labels=[0, 1], zero_division=0
        )
        sell_recall = rec[0]
        buy_f2 = fbeta_score(y_valid, y_pred, beta=2.0, pos_label=1, zero_division=0)
        if sell_recall >= keep_sell_recall and buy_f2 > best_score:
            best_score, best_t = buy_f2, float(t)

    if best_score < 0:
        # fallback: maximize macro-F1
        macro_best, macro_t = -1.0, 0.5
        for t in grid:
            y_pred = (valid_proba >= t).astype(int)
            _, _, f1s, _ = precision_recall_fscore_support(
                y_valid, y_pred, average=None, labels=[0, 1], zero_division=0
            )
            macro = float(np.mean(f1s))
            if macro > macro_best:
                macro_best, macro_t = macro, float(t)
        best_t = macro_t

    # regime-conditioned thresholds
    e_edges, a_edges, thr_grid = _regime_thresholds(
        valid_proba, y_valid, X_valid, best_t, sell_floor=keep_sell_recall
    )

    # final validation report
    y_valid_pred = (valid_proba >= best_t).astype(int)
    report = classification_report(
        y_valid, y_valid_pred, target_names=["Sell (0)", "Buy (1)"], zero_division=0
    )
    cm = confusion_matrix(y_valid, y_valid_pred, labels=[0, 1])

    return clf, cal, best_t, e_edges, a_edges, thr_grid, report, cm


def train_and_save_model(
    df: pd.DataFrame,
    model_path: str = "models/best_balanced_model.pkl",
    keep_sell_recall: float = 0.93,
):
    # unify to a single DatetimeIndex ONCE
    dfi = _ensure_time_index(df)

    # labels on SAME index, slightly longer horizon
    labels = get_triple_barrier_labels(dfi, time_barrier=45)  # was 30
    mask = labels != 0
    y = labels.loc[mask].map({-1: 0, 1: 1}).astype(int)

    # features on SAME dataframe
    X_full = create_features(dfi)

    # align after feature dropna and neutral drop
    X, y = _align_Xy(X_full, y)
    assert len(X) == len(y) and len(X) > 0, "Empty aligned dataset. Check time index and NaNs."
    print("Class balance:", y.value_counts(normalize=True).to_dict())

    # train
    model, calibrator, threshold, e_edges, a_edges, thr_grid, report, cm = train_lightgbm_binary(
        X, y, keep_sell_recall=keep_sell_recall
    )
    print("\n--- Validation report (threshold tuned) ---")
    print(report)
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    # save bundle
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "calibrator": calibrator,
            "threshold": float(threshold),          # global fallback
            "ema200_slope_edges": e_edges,          # list[2]
            "atr_pctile_edges": a_edges,            # list[2]
            "thr_grid": thr_grid,                   # 3x3 list of floats
        },
        model_path,
    )
    print(f"\nFinal model saved to: {Path(model_path).resolve()}")
    print(f"Chosen threshold: {threshold:.3f}")


def load_and_predict(df: pd.DataFrame, model_path: str):
    bundle = joblib.load(model_path)
    model = bundle["model"]
    calibrator = bundle.get("calibrator")
    thr_global = float(bundle["threshold"])
    e_edges = bundle.get("ema200_slope_edges")
    a_edges = bundle.get("atr_pctile_edges")
    thr_grid = bundle.get("thr_grid")

    X = create_features(_ensure_time_index(df))
    proba = (
        calibrator.predict_proba(X)[:, 1] if calibrator is not None else model.predict_proba(X)[:, 1]
    )

    if e_edges and a_edges and thr_grid and "ema200_slope" in X and "atr_pctile" in X:
        e_bin = np.digitize(X["ema200_slope"].to_numpy(), np.array(e_edges, dtype=float), right=True)
        a_bin = np.digitize(X["atr_pctile"].to_numpy(), np.array(a_edges, dtype=float), right=True)
        thr_mat = np.array(thr_grid, dtype=float)
        thr = thr_mat[e_bin, a_bin]
    else:
        thr = np.full_like(proba, thr_global)

    preds = (proba >= thr).astype(int)  # 0=sell, 1=buy
    return preds, proba


if __name__ == "__main__":
    # CSV with columns: timestamp,open,high,low,close,tick_volume
    here = Path(__file__).resolve()
    project_root = here.parents[2] if len(here.parents) >= 3 else here.parent.parent

    csv_env = os.environ.get("XAUUSD_CSV", "")
    csv_path = Path(csv_env) if csv_env else (project_root / "data" / "processed" / "xauusd_m1_processed.csv")

    if not csv_path.exists():
        raise SystemExit(
            f"CSV not found at: {csv_path}\n"
            "Set XAUUSD_CSV env var or place file at data/processed/xauusd_m1_processed.csv.\n"
            "Expected columns: timestamp,open,high,low,close,tick_volume"
        )

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
    raw = pd.read_csv(csv_path, **read_kwargs)

    # normalize headers to lowercase; keep timestamp as a column
    raw.rename(columns={c: c.lower() for c in raw.columns}, inplace=True)

    # basic checks
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(raw.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    if "tick_volume" not in raw.columns:
        print("Note: 'tick_volume' not found. Volume-based features will be skipped.")

    # train
    model_out = project_root / "models" / "best_balanced_model.pkl"
    train_and_save_model(raw, model_path=str(model_out), keep_sell_recall=0.93)
