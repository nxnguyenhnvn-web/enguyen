"""Price forecasting with Kronos (NeoQuasar/Kronos-*) plus a statistical fallback and rolling backtest."""
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
KRONOS_SRC =os.path.join(os.path.dirname(os.path.abspath(__file__)), "kronos_src", "Kronos-master")
TOKENIZERS = {"NeoQuasar/Kronos-mini": "NeoQuasar/Kronos-Tokenizer-2k",
              "NeoQuasar/Kronos-small": "NeoQuasar/Kronos-Tokenizer-base",
              "NeoQuasar/Kronos-base": "NeoQuasar/Kronos-Tokenizer-base"}
MAX_CONTEXT = {"NeoQuasar/Kronos-mini": 2048}


def load_kronos(model_name="NeoQuasar/Kronos-small"):
    """Returns (predictor, error_message)."""
    try:
        if KRONOS_SRC not in sys.path:
            sys.path.insert(0, KRONOS_SRC)
        import torch
        from model import Kronos, KronosTokenizer, KronosPredictor
        tok = KronosTokenizer.from_pretrained(TOKENIZERS[model_name])
        mdl = Kronos.from_pretrained(model_name)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return KronosPredictor(mdl, tok, device=device, max_context=MAX_CONTEXT.get(model_name, 512)), None
    except Exception as e:  # missing torch / no internet / repo not downloaded
        return None, f"{type(e).__name__}: {e}"


def future_days(last: pd.Timestamp, n: int) -> pd.Series:
    return pd.Series(pd.bdate_range(last + pd.Timedelta(days=1), periods=n))  # HOSE/HNX trade Mon–Fri


def kronos_paths(predictor, df: pd.DataFrame, horizon: int, lookback: int, n_paths: int) -> np.ndarray:
    """Runs n_paths stochastic Kronos samples; returns array (n_paths, horizon) of closes."""
    x = df.tail(lookback).reset_index(drop=True)
    x_df = x[["open", "high", "low", "close", "volume"]].astype(float)
    x_df["amount"] = x_df["volume"] * x_df[["open", "high", "low", "close"]].mean(axis=1)
    x_ts = pd.Series(pd.to_datetime(x["time"]))
    y_ts = future_days(x_ts.iloc[-1], horizon)
    out = []
    for _ in range(n_paths):
        p = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=horizon,
                              T=1.0, top_p=0.9, sample_count=1, verbose=False)
        out.append(p["close"].values)
    return np.array(out)


def stat_paths(df: pd.DataFrame, horizon: int, lookback: int, n_paths: int, seed=7) -> np.ndarray:
    """Fallback: geometric Brownian motion with shrunk drift from recent log returns."""
    lr = np.log(df["close"].tail(lookback)).diff().dropna()
    mu, sig = lr.ewm(span=60).mean().iloc[-1] * 0.5, lr.tail(120).std()
    rng = np.random.default_rng(seed)
    shocks = rng.normal(mu, sig, size=(max(n_paths, 200), horizon))
    return df["close"].iloc[-1] * np.exp(np.cumsum(shocks, axis=1))


def forecast(df, horizon=20, lookback=400, n_paths=5, predictor=None):
    """Returns dict with dates, mean/low/high paths, expected return and engine name."""
    if predictor is not None:
        paths, engine = kronos_paths(predictor, df, horizon, lookback, n_paths), "Kronos"
    else:
        paths, engine = stat_paths(df, horizon, lookback, n_paths), "Statistical (GBM)"
    last = df["close"].iloc[-1]
    mean = paths.mean(axis=0)
    return {
        "dates": future_days(pd.to_datetime(df["time"].iloc[-1]), horizon),
        "mean": mean, "low": np.percentile(paths, 10, axis=0), "high": np.percentile(paths, 90, axis=0),
        "exp_ret": float(mean[-1] / last - 1), "engine": engine, "paths": paths,
    }


def backtest(df, horizon=20, lookback=400, windows=6, n_paths=3, predictor=None):
    """Rolling-origin test: forecast from past cut-offs, compare with what actually happened."""
    rows = []
    for i in range(windows, 0, -1):
        cut = len(df) - i * horizon
        if cut < 120:
            continue
        hist, actual = df.iloc[:cut], df.iloc[cut:cut + horizon]
        f = forecast(hist, len(actual), lookback, n_paths, predictor)
        a, p, last = actual["close"].values, f["mean"], hist["close"].iloc[-1]
        rows.append({
            "start": pd.to_datetime(hist["time"].iloc[-1]).date(),
            "last_close": last, "pred_end": p[-1], "actual_end": a[-1],
            "pred_ret": p[-1] / last - 1, "actual_ret": a[-1] / last - 1,
            "mape": float(np.mean(np.abs(p - a) / a)),
            "naive_mape": float(np.mean(np.abs(last - a) / a)),  # random-walk benchmark
            "direction_ok": np.sign(p[-1] - last) == np.sign(a[-1] - last),
            "in_band": bool(f["low"][-1] <= a[-1] <= f["high"][-1]),
        })
    bt = pd.DataFrame(rows)
    summary = {} if bt.empty else {
        "hit_rate": float(bt["direction_ok"].mean()), "mape": float(bt["mape"].mean()),
        "naive_mape": float(bt["naive_mape"].mean()), "band_coverage": float(bt["in_band"].mean()),
        "engine": "Kronos" if predictor is not None else "Statistical (GBM)",
    }
    return bt, summary
