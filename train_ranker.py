"""Walk-forward training + backtest of the cross-sectional LightGBM ranker vs Kronos vs momentum.

    python train_ranker.py                    # data + ML/momentum backtest + final model
    python train_ranker.py --kronos 24        # also backtest Kronos on the last 24 monthly dates

Writes models/ranker.txt, models/meta.json, models/backtest.json.
"""
import argparse
import json
import os
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import analysis as A
import data as D
import features as FE
import forecast as F

PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=50, bagging_fraction=0.8,
              bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0, verbose=-1, seed=7, num_threads=4)
ROUNDS = 300


def fit(X, y):
    return lgb.train(PARAMS, lgb.Dataset(X, y), num_boost_round=ROUNDS)
TEST_START = "2023-01-01"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache", "raw.pkl")


def get_raw(refresh=False):
    if os.path.exists(CACHE) and not refresh and time.time() - os.path.getmtime(CACHE) < 86400:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    syms = FE.universe()
    print(f"downloading {len(syms)} stocks …")
    raw = FE.load_all(syms)
    idx = D.history("VNINDEX")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump((raw, idx), f)
    return raw, idx


def build(raw, idx):
    calendar = pd.DatetimeIndex(idx["time"])
    dates = FE.month_ends(calendar)
    P = FE.panel(raw, calendar, dates)
    P = P[P["value20"] >= FE.MIN_VALUE].reset_index(drop=True)  # investable names only
    X = FE.normalise(P)
    P["y"] = P.groupby("date")["fwd"].rank(pct=True)
    return calendar, dates, P, X


def walk_forward(P, X, calendar):
    """Retrain every month on labels that were fully known at that date."""
    pos = {d: i for i, d in enumerate(calendar)}
    test_dates = sorted(d for d in P["date"].unique() if d >= pd.Timestamp(TEST_START) and P.loc[P.date == d, "fwd"].notna().any())
    preds = []
    for d in test_dates:
        known = P["date"].map(pos) + FE.HORIZON <= pos[d]
        tr = known & P["y"].notna()
        m = fit(X[tr], P.loc[tr, "y"])
        te = P["date"] == d
        preds.append(pd.Series(m.predict(X[te]), index=P.index[te]))
    P["ml"] = pd.concat(preds)
    P["momentum"] = X["mom12_1"]
    return test_dates


def kronos_scores(P, raw, dates, lookback=256, model="NeoQuasar/Kronos-small"):
    pred, err = F.load_kronos(model)
    if pred is None:
        print("Kronos unavailable:", err)
        return
    P["kronos"] = np.nan
    for n, d in enumerate(dates):
        rows = P.index[P["date"] == d]
        dfs, xts, yts, keep = [], [], [], []
        for i in rows:
            h = raw[P.at[i, "sym"]][0]
            h = h[h["time"] <= d].tail(lookback)
            if len(h) < lookback:
                continue
            x = h[["open", "high", "low", "close", "volume"]].astype(float).reset_index(drop=True)
            x["amount"] = x["volume"] * x[["open", "high", "low", "close"]].mean(axis=1)
            dfs.append(x)
            xts.append(pd.Series(pd.to_datetime(h["time"].values)))
            yts.append(F.future_days(d, FE.HORIZON))
            keep.append(i)
        t = time.time()
        out = pred.predict_batch(dfs, xts, yts, pred_len=FE.HORIZON, T=1.0, top_p=0.9, sample_count=1, verbose=False)
        for i, df_in, o in zip(keep, dfs, out):
            P.at[i, "kronos"] = o["close"].iloc[-1] / df_in["close"].iloc[-1] - 1
        print(f"  Kronos {n + 1}/{len(dates)} {d.date()} · {len(keep)} stocks · {time.time() - t:.0f}s")


def evaluate(P, method, dates, subset=None):
    rows = []
    for d in dates:
        g = P[(P["date"] == d) & P[method].notna() & P["fwd"].notna()]
        if subset is not None:
            g = g[g["sym"].isin(subset)]
        if len(g) < 8:
            continue
        ic = g[method].rank().corr(g["fwd"].rank())
        q = pd.qcut(g[method].rank(method="first"), 5, labels=False)
        top, bot, allm = g.loc[q == 4, "fwd"].mean(), g.loc[q == 0, "fwd"].mean(), g["fwd"].mean()
        rows.append({"date": str(d.date()), "ic": ic, "top_excess": top - allm, "spread": top - bot, "n": len(g)})
    r = pd.DataFrame(rows)
    if r.empty:
        return {}, r
    s = {"months": len(r), "mean_ic": r["ic"].mean(), "ic_t": r["ic"].mean() / r["ic"].std() * np.sqrt(len(r)),
         "pct_ic_pos": (r["ic"] > 0).mean(), "top_excess_m": r["top_excess"].mean(), "spread_m": r["spread"].mean(),
         "top_hit": (r["top_excess"] > 0).mean()}
    r["cum_top_excess"] = (1 + r["top_excess"]).cumprod() - 1
    return {k: float(v) for k, v in s.items()}, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kronos", type=int, default=0, help="backtest Kronos on the last N monthly dates")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    raw, idx = get_raw(a.refresh)
    calendar, dates, P, X = build(raw, idx)
    print(f"panel: {len(P)} rows · {P['sym'].nunique()} stocks · {P['date'].nunique()} dates")
    test_dates = walk_forward(P, X, calendar)
    methods = ["ml", "momentum"]
    kdates = test_dates[-a.kronos:] if a.kronos else []
    if kdates:
        kronos_scores(P, raw, kdates)
        methods.append("kronos")

    transport = set(A.UNIVERSE)
    report = {"generated": str(pd.Timestamp.today().date()), "horizon": FE.HORIZON,
              "test_start": TEST_START, "universe_size": int(P["sym"].nunique()), "results": {}}
    for scope, sub in (("all", None), ("transport", transport)):
        for mth in methods:
            # compare every method on the same dates as Kronos when it ran, plus ML/momentum on the full period
            for label, ds in (("full", test_dates), ("kronos_window", kdates)):
                if not ds or (mth == "kronos" and label == "full"):
                    continue
                s, r = evaluate(P, mth, ds, sub)
                report["results"][f"{scope}|{mth}|{label}"] = {"summary": s, "series": r.to_dict("records")}
    for k, v in report["results"].items():
        s = v["summary"]
        if s:
            print(f"{k:38s} IC {s['mean_ic']:+.3f} (t {s['ic_t']:+.1f}, {s['pct_ic_pos']:.0%} pos) · "
                  f"top-quintile excess {s['top_excess_m']:+.2%}/mo · Q5-Q1 {s['spread_m']:+.2%}/mo")

    # final model on every label known today
    tr = P["y"].notna()
    final = fit(X[tr], P.loc[tr, "y"])
    os.makedirs(FE.MODEL_DIR, exist_ok=True)
    final.save_model(os.path.join(FE.MODEL_DIR, "ranker.txt"))
    imp = pd.Series(final.feature_importance("gain"), index=list(FE.FEATURES))
    report["importance"] = (imp / imp.sum()).sort_values(ascending=False).round(4).to_dict()
    with open(os.path.join(FE.MODEL_DIR, "backtest.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=float)
    meta = {"features": list(FE.FEATURES), "trained_to": str(P.loc[tr, "date"].max().date()),
            "rows": int(tr.sum()), "universe": sorted(P["sym"].unique()), "params": PARAMS, "rounds": ROUNDS}
    with open(os.path.join(FE.MODEL_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print("saved models/ranker.txt, meta.json, backtest.json")


if __name__ == "__main__":
    main()
