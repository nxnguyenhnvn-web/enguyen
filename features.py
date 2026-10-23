"""Point-in-time features for the cross-sectional ML ranker (shared by train_ranker.py and the app)."""
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import analysis as A
import data as D

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HORIZON = 20          # trading days ahead the ranker predicts
MIN_VALUE = 1e9       # VND/day avg traded value to be investable
FIN_LAG_DAYS = {1: 45, 2: 45, 3: 45, 4: 90}  # days after quarter-end before results are public

FEATURES = {
    # name: label used in reasons
    "r21": "1-month return", "r63": "3-month return", "mom12_1": "12-1 month momentum",
    "vol60": "60-day volatility", "dist_hi": "Distance from 52-week high", "rsi": "RSI(14)",
    "ma50_200": "MA50 / MA200", "px_ma200": "Price vs MA200", "vol_trend": "Volume trend (20d/60d)",
    "log_value": "Liquidity (traded value)", "max_ret21": "Largest daily jump (1M)",
    "fflow20": "Foreign net buying (20d, % of value)", "fflow60": "Foreign net buying (60d, % of value)",
    "fown_chg60": "Change in foreign ownership (60d)",
    "ey": "Earnings yield (TTM E/P)", "bp": "Book-to-price", "sp": "Sales-to-price", "log_mcap": "Market cap",
    "roe": "ROE", "roe_chg": "ROE change YoY", "net_margin": "Net margin", "de": "Debt/Equity",
    "rev_growth": "Revenue growth YoY", "eps_growth": "Earnings growth YoY", "eps_accel": "Earnings acceleration",
}

H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def _json(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or H), timeout=30) as r:
        return json.loads(r.read())


def universe() -> list[str]:
    """VN100 + transport/port names. Falls back to the list saved with the model."""
    try:
        vn100 = [x["symbol"] for x in _json("https://trading.vietcap.com.vn/api/price/symbols/getByGroup?group=VN100",
                                            {**H, "Referer": "https://trading.vietcap.com.vn/"})]
    except Exception:
        vn100 = []
    if not vn100:
        try:
            with open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8") as f:
                vn100 = json.load(f)["universe"]
        except Exception:
            vn100 = []
    return sorted(set(vn100) | set(A.UNIVERSE))


def foreign(symbol: str, start: str = "2020-01-01") -> pd.DataFrame:
    try:
        rows = _json(f"https://api-finfo.vndirect.com.vn/v4/foreigns?q=code:{symbol}~tradingDate:gte:{start}"
                     f"&sort=tradingDate&size=3000&page=1")["data"]
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    f = pd.DataFrame(rows)
    f["time"] = pd.to_datetime(f["tradingDate"])
    f["fown"] = 1 - f["currentRoom"] / f["totalRoom"].replace(0, np.nan)  # share of foreign room used
    return f[["time", "netVal", "fown"]].sort_values("time").drop_duplicates("time")


def load_one(sym):
    return sym, D.history(sym), D.ratios_ttm(sym), foreign(sym)


def load_all(symbols, workers=12) -> dict:
    with ThreadPoolExecutor(workers) as ex:
        return {s: (h, r, f) for s, h, r, f in ex.map(load_one, symbols)}


def _fundamentals_asof(rat: pd.DataFrame) -> pd.DataFrame:
    """Quarterly TTM rows stamped with the date they became public."""
    if rat.empty:
        return pd.DataFrame()
    r = rat.copy()
    qend = pd.to_datetime(dict(year=r["yearReport"], month=3 * r["quarter"], day=1)) + pd.offsets.MonthEnd(0)
    r["avail"] = qend + pd.to_timedelta(r["quarter"].map(FIN_LAG_DAYS), unit="D")
    with np.errstate(divide="ignore", invalid="ignore"):
        r["equity"] = np.where(r["pb"] > 0, r["marketCap"] / r["pb"], np.nan)
    r["roe_chg"] = r["roe"] - r["roe"].shift(4)
    r["eps_accel"] = r["eps_growth"] - r["eps_growth"].shift(1)
    r["net_margin"] = r["afterTaxProfitMargin"]
    r["de"] = r["debtToEquity"]
    keep = ["avail", "ttm_earnings", "ttm_revenue", "equity", "numberOfSharesMktCap", "roe", "roe_chg",
            "net_margin", "de", "rev_growth", "eps_growth", "eps_accel"]
    return r[keep].sort_values("avail")


def daily_features(h: pd.DataFrame, rat: pd.DataFrame, fr: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Raw (un-normalised) features for every day of the market calendar, plus the forward-return label."""
    if h.empty:
        return pd.DataFrame()
    p = h.set_index("time")[["open", "high", "low", "close", "volume"]]
    p = p[~p.index.duplicated()].reindex(calendar)
    p["close"] = p["close"].ffill(limit=15)  # no trade = unchanged price
    p["volume"] = p["volume"].fillna(0)
    c = p["close"]
    lr = np.log(c).diff()
    val = c * 1000 * p["volume"]
    d = pd.DataFrame(index=calendar)
    d["close"] = c
    d["r21"], d["r63"] = c / c.shift(21) - 1, c / c.shift(63) - 1
    d["mom12_1"] = c.shift(21) / c.shift(252) - 1
    d["vol60"] = lr.rolling(60).std() * np.sqrt(250)
    d["dist_hi"] = c / c.rolling(252, min_periods=120).max() - 1
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    d["ma50_200"], d["px_ma200"] = ma50 / ma200 - 1, c / ma200 - 1
    d["vol_trend"] = p["volume"].rolling(20).mean() / p["volume"].rolling(60).mean().replace(0, np.nan)
    d["value20"] = val.rolling(20).mean()
    d["log_value"] = np.log1p(d["value20"])
    d["max_ret21"] = lr.rolling(21).max()
    if not fr.empty:
        f = fr.set_index("time").reindex(calendar)
        nv = f["netVal"].fillna(0)
        d["fflow20"] = nv.rolling(20).sum() / val.rolling(20).sum().replace(0, np.nan)
        d["fflow60"] = nv.rolling(60).sum() / val.rolling(60).sum().replace(0, np.nan)
        d["fown_chg60"] = f["fown"].ffill().diff(60)
    fa = _fundamentals_asof(rat)
    if not fa.empty:
        left = d[["close"]].rename_axis("time").reset_index()
        left["time"] = left["time"].astype("datetime64[ns]")
        fa = fa.assign(avail=fa["avail"].astype("datetime64[ns]"))
        m = pd.merge_asof(left, fa, left_on="time", right_on="avail").set_index("time")
        m.index = d.index
        mcap = m["close"] * 1000 * m["numberOfSharesMktCap"]
        d["ey"], d["bp"], d["sp"] = m["ttm_earnings"] / mcap, m["equity"] / mcap, m["ttm_revenue"] / mcap
        d["log_mcap"] = np.log(mcap)
        for k in ("roe", "roe_chg", "net_margin", "de", "rev_growth", "eps_growth", "eps_accel"):
            d[k] = m[k]
    d["fwd"] = c.shift(-HORIZON) / c - 1
    for k in FEATURES:
        if k not in d:
            d[k] = np.nan
    return d.replace([np.inf, -np.inf], np.nan)


def panel(raw: dict, calendar: pd.DatetimeIndex, dates) -> pd.DataFrame:
    """Long table (date, sym, raw features, fwd, value20) sampled at rebalance dates."""
    parts = []
    for sym, (h, r, f) in raw.items():
        d = daily_features(h, r, f, calendar)
        if d.empty:
            continue
        d = d.loc[d.index.intersection(dates)].copy()
        d["sym"] = sym
        parts.append(d)
    out = pd.concat(parts).rename_axis("date").reset_index()
    return out[out["close"].notna()]


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank of every feature within each date (NaN stays NaN)."""
    X = df.groupby("date")[list(FEATURES)].rank(pct=True)
    return X


def _fmt(k, v):
    if pd.isna(v):
        return "n/a"
    if k in ("rsi",):
        return f"{v:.0f}"
    if k in ("log_value", "log_mcap"):
        return f"{np.exp(v) / 1e9:,.0f} bn"
    if k in ("de", "vol_trend"):
        return f"{v:.2f}"
    return f"{v:+.1%}" if k not in ("vol60", "roe", "net_margin", "ey", "bp", "sp") else f"{v:.1%}"


def score_latest(raw: dict, idx: pd.DataFrame, booster) -> pd.DataFrame:
    """Score every stock on the latest date. Features are ranked against the investable universe,
    exactly like in training; illiquid names are placed within that same distribution."""
    calendar = pd.DatetimeIndex(idx["time"])
    P = panel(raw, calendar, calendar[-1:]).set_index("sym")
    elig = P[P["value20"] >= MIN_VALUE]
    X = pd.DataFrame(index=P.index)
    for k in FEATURES:
        ref = np.sort(elig[k].dropna().values)
        X[k] = [np.nan if pd.isna(v) or not len(ref) else np.searchsorted(ref, v, side="right") / len(ref)
                for v in P[k]]
    pred = booster.predict(X[list(FEATURES)])
    contrib = pd.DataFrame(booster.predict(X[list(FEATURES)], pred_contrib=True)[:, :-1],
                           index=P.index, columns=list(FEATURES))
    investable = (P["value20"] >= MIN_VALUE).values
    ref = np.sort(pred[investable]) if investable.any() else np.sort(pred)
    score = np.searchsorted(ref, pred, side="right") / len(ref) * 100
    # the model never saw illiquid stocks in training, so it does not score them
    out = pd.DataFrame({"pred": pred, "ml_score": np.where(investable, score, np.nan), "investable": investable},
                       index=P.index)
    why = {}
    for s in P.index:
        c = contrib.loc[s][X.loc[s].notna()].sort_values()
        phrase = lambda k: f"{FEATURES[k]} {_fmt(k, P.at[s, k])} (higher than {X.at[s, k]:.0%} of stocks)"
        items = [("+", phrase(k)) for k in c.index[::-1][:3] if c[k] > 0]
        items += [("-", phrase(k)) for k in c.index[:3] if c[k] < 0]
        why[s] = items
    out["why"] = pd.Series(why)
    out["date"] = calendar[-1]
    return out


def month_ends(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(calendar, index=calendar)
    return pd.DatetimeIndex(s.groupby([calendar.year, calendar.month]).max().values)
