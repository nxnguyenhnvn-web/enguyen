"""Technical indicators, fundamental/macro/forecast/liquidity scoring and reason generation."""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- universe
UNIVERSE = {
    # sub-sector drives the macro sensitivity profile
    "GMD": ("Gemadept – port & logistics", "port"),
    "PHP": ("Hai Phong Port", "port"),
    "SCS": ("Saigon Cargo Service (air cargo terminal)", "aircargo"),
    "VSC": ("Viconship – port & logistics", "port"),
    "SGP": ("Saigon Port", "port"),
    "CDN": ("Da Nang Port", "port"),
    "DVP": ("Dinh Vu Port", "port"),
    "TCL": ("Tan Cang Logistics & Stevedoring", "port"),
    "PDN": ("Dong Nai Port", "port"),
    "CLL": ("Cat Lai Port", "port"),
    "DXP": ("Doan Xa Port", "port"),
    "CQN": ("Quang Ninh Port", "port"),
    "VGR": ("Green Port (Vinaconship)", "port"),
    "PSP": ("Dinh Vu Petroleum Port", "port"),
    "MVN": ("VIMC – Vietnam Maritime Corp", "port"),
    "HAH": ("Hai An Transport & Stevedoring", "shipping"),
    "VOS": ("Vietnam Ocean Shipping", "shipping"),
    "VTO": ("Vitaco – petroleum tankers", "tanker"),
    "PVT": ("PV Trans – tankers", "tanker"),
    "PVP": ("PV Pacific Transport", "tanker"),
    "GSP": ("Gas Shipping (PV Gas)", "tanker"),
    "VIP": ("Vipco – petroleum shipping", "tanker"),
    "PJT": ("Petrolimex Waterway Transport", "tanker"),
    "STG": ("Southern Logistics (Sotrans)", "logistics"),
    "TJC": ("Transport & Chartering", "shipping"),
    "ULG": ("Universal Logistics", "logistics"),
    "GIC": ("Industrial transport", "logistics"),
    "ALC": ("Industrial transport", "logistics"),
    "CCR": ("Cam Ranh Port", "port"),
    "VSG": ("Sea transport", "shipping"),
}

GROUPS = {
    "Vận tải công nghiệp (screenshot)": ["PJT", "CQN", "PHP", "GMD", "ALC", "CCR", "SCS", "MVN", "VTO",
                                         "DXP", "PVP", "ULG", "TJC", "GIC", "STG", "VSG"],
    "Cảng biển (seaports)": ["GMD", "PHP", "VSC", "SGP", "CDN", "DVP", "TCL", "PDN", "CLL", "DXP",
                             "CQN", "VGR", "PSP", "MVN", "CCR", "HAH"],
    "Vận tải biển (shipping/tankers)": ["HAH", "VOS", "VTO", "PVT", "PVP", "GSP", "VIP", "PJT", "VSG", "TJC"],
}

# ---------------------------------------------------------------- macro
MACRO_FACTORS = {
    # key: (label, default, help)  values on a -2 (very negative) .. +2 (very positive) scale
    "trade": ("Export/import growth (VN)", 1.0, "Container throughput tracks trade volume"),
    "freight": ("Global freight / charter rates", 0.0, "Higher rates lift shipping lines, mixed for ports"),
    "rates": ("Interest rate trend (+ = easing)", 0.0, "Easing lowers financing cost of capex-heavy ports/fleets"),
    "fx": ("USD/VND (+ = VND weaker)", 0.5, "USD-linked revenue benefits shipping/ports"),
    "oil": ("Oil price (+ = higher)", 0.0, "Higher oil = cost for shippers, demand for tankers"),
    "fdi": ("FDI / industrial-park inflows", 1.0, "Drives logistics & northern port demand"),
    "infra": ("Public investment / infrastructure", 1.0, "New terminals, Lach Huyen, Cai Mep, expressways"),
    "tariff": ("US/global tariff risk (+ = easing)", -0.5, "Tariff escalation hits export volumes"),
}

# sensitivity of each sub-sector to each macro factor (-1..1)
SENSITIVITY = {
    "port":      {"trade": 1.0, "freight": 0.2, "rates": 0.5, "fx": 0.4, "oil": -0.1, "fdi": 0.8, "infra": 0.6, "tariff": 0.8},
    "shipping":  {"trade": 0.8, "freight": 1.0, "rates": 0.6, "fx": 0.6, "oil": -0.5, "fdi": 0.4, "infra": 0.1, "tariff": 0.7},
    "tanker":    {"trade": 0.3, "freight": 0.8, "rates": 0.5, "fx": 0.5, "oil": 0.5, "fdi": 0.2, "infra": 0.1, "tariff": 0.2},
    "logistics": {"trade": 0.8, "freight": 0.1, "rates": 0.5, "fx": 0.2, "oil": -0.4, "fdi": 1.0, "infra": 0.6, "tariff": 0.7},
    "aircargo":  {"trade": 0.9, "freight": 0.3, "rates": 0.3, "fx": 0.3, "oil": -0.4, "fdi": 0.9, "infra": 0.3, "tariff": 0.8},
}


# ---------------------------------------------------------------- indicators
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["close"]
    for n in (20, 50, 200):
        d[f"ma{n}"] = c.rolling(n).mean()
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    std20 = c.rolling(20).std()
    d["bb_up"], d["bb_dn"] = d["ma20"] + 2 * std20, d["ma20"] - 2 * std20
    tr = pd.concat([d["high"] - d["low"], (d["high"] - c.shift()).abs(), (d["low"] - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["vol20"] = d["volume"].rolling(20).mean()
    d["vol60"] = d["volume"].rolling(60).mean()
    d["value20"] = (d["close"] * d["volume"]).rolling(20).mean() * 1000  # price is in 1,000 VND
    return d


def ret(c: pd.Series, n: int):
    return float(c.iloc[-1] / c.iloc[-n - 1] - 1) if len(c) > n else np.nan


def ta_snapshot(d: pd.DataFrame) -> dict:
    last, c = d.iloc[-1], d["close"]
    win = c.tail(250)
    return {
        "close": last["close"], "ma20": last["ma20"], "ma50": last["ma50"], "ma200": last["ma200"],
        "rsi": last["rsi"], "macd": last["macd"], "macd_signal": last["macd_signal"], "macd_hist": last["macd_hist"],
        "atr_pct": last["atr"] / last["close"] if last["close"] else np.nan,
        "ret_1m": ret(c, 21), "ret_3m": ret(c, 63), "ret_6m": ret(c, 126), "ret_1y": ret(c, 250),
        "hi52": win.max(), "lo52": win.min(), "from_hi52": last["close"] / win.max() - 1,
        "vol_ratio": last["vol20"] / last["vol60"] if last["vol60"] else np.nan,
        "value20": last["value20"],
        "volatility": float(np.log(c).diff().tail(60).std() * np.sqrt(250)),
    }


# ---------------------------------------------------------------- scoring
def _ok(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def score_ta(s: dict):
    pts, why = 0.0, []
    c = s["close"]
    for key, w in (("ma20", 10), ("ma50", 15), ("ma200", 15)):
        if _ok(s[key]):
            if c > s[key]:
                pts += w; why.append(("+", f"Price above {key.upper()} ({s[key]:.2f})"))
            else:
                why.append(("-", f"Price below {key.upper()} ({s[key]:.2f})"))
        else:
            pts += w / 2
    if _ok(s["ma50"]) and _ok(s["ma200"]):
        if s["ma50"] > s["ma200"]:
            pts += 10; why.append(("+", "MA50 > MA200 (golden-cross regime)"))
        else:
            why.append(("-", "MA50 < MA200 (death-cross regime)"))
    r = s["rsi"]
    if _ok(r):
        if 45 <= r <= 65: pts += 15; why.append(("+", f"RSI {r:.0f} healthy momentum"))
        elif 30 <= r < 45: pts += 10; why.append(("~", f"RSI {r:.0f} weak but not oversold"))
        elif r < 30: pts += 8; why.append(("~", f"RSI {r:.0f} oversold – rebound possible"))
        elif r <= 75: pts += 8; why.append(("~", f"RSI {r:.0f} strong, nearing overbought"))
        else: pts += 2; why.append(("-", f"RSI {r:.0f} overbought"))
    if _ok(s["macd"]) and s["macd"] > s["macd_signal"]:
        pts += 15; why.append(("+", "MACD above signal line (bullish)"))
    else:
        why.append(("-", "MACD below signal line (bearish)"))
    m3 = s["ret_3m"]
    if _ok(m3):
        pts += float(np.clip(5 + m3 * 50, 0, 10))
        why.append(("+" if m3 > 0 else "-", f"3-month return {m3:+.1%}"))
    if _ok(s["vol_ratio"]) and _ok(s["ret_1m"]):
        if s["vol_ratio"] > 1.1 and s["ret_1m"] > 0:
            pts += 10; why.append(("+", f"Rising volume on up-move (vol20/vol60 = {s['vol_ratio']:.2f})"))
        elif s["vol_ratio"] > 1.1 and s["ret_1m"] < 0:
            why.append(("-", f"Rising volume on down-move (distribution, {s['vol_ratio']:.2f}x)"))
        else:
            pts += 5
    return float(np.clip(pts, 0, 100)), why


FA_RULES = [
    # key, label, max points, scorer
    ("roe", "ROE", 20, lambda v: 20 if v > .20 else 16 if v > .15 else 11 if v > .10 else 6 if v > .05 else 2),
    ("pe", "P/E", 15, lambda v: 1 if v <= 0 else 15 if v < 8 else 12 if v < 12 else 9 if v < 16 else 5 if v < 22 else 1),
    ("pb", "P/B", 12, lambda v: 12 if v < 1 else 10 if v < 1.5 else 7 if v < 2.5 else 4 if v < 4 else 1),
    ("de", "Debt/Equity", 12, lambda v: 12 if v < .3 else 9 if v < .7 else 6 if v < 1.2 else 3 if v < 2 else 0),
    ("net_margin", "Net margin", 12, lambda v: 12 if v > .20 else 9 if v > .10 else 6 if v > .05 else 3 if v > 0 else 0),
    ("rev_growth", "Revenue growth", 12, lambda v: 12 if v > .20 else 9 if v > .10 else 6 if v > 0 else 2),
    ("eps_growth", "EPS growth", 10, lambda v: 10 if v > .20 else 7 if v > .10 else 5 if v > 0 else 1),
    ("div_yield", "Dividend yield", 7, lambda v: 7 if v > .06 else 5 if v > .03 else 3 if v > .01 else 0),
]
PCT_KEYS = {"roe", "net_margin", "rev_growth", "eps_growth", "div_yield", "roa", "gross_margin"}


def fmt_ratio(k, v):
    if not _ok(v):
        return "n/a"
    return f"{v:.1%}" if k in PCT_KEYS else f"{v:.2f}"


def score_fa(fa: dict):
    pts, why, have = 0.0, [], 0
    for key, label, mx, fn in FA_RULES:
        v = fa.get(key)
        if _ok(v):
            have += 1
            p = fn(v)
            pts += p
            sign = "+" if p >= .7 * mx else "-" if p <= .3 * mx else "~"
            why.append((sign, f"{label} {fmt_ratio(key, v)} → {p}/{mx}"))
        else:
            pts += mx / 2
            why.append(("~", f"{label} unavailable → neutral {mx / 2:.0f}/{mx}"))
    if have == 0:
        why = [("~", "No fundamental data returned by data source – FA set to neutral 50")]
    return float(np.clip(pts, 0, 100)), why


def score_macro(subsector: str, macro: dict, market: dict | None):
    sens = SENSITIVITY.get(subsector, SENSITIVITY["logistics"])
    raw = sum(sens[k] * macro.get(k, 0.0) for k in sens)
    norm = sum(abs(v) for v in sens.values()) * 2  # max possible |raw|
    score = 50 + 50 * raw / norm * 1.6
    why = []
    for k, v in sorted(sens.items(), key=lambda kv: -abs(kv[1] * macro.get(kv[0], 0))):
        impact = v * macro.get(k, 0)
        if abs(impact) >= .2:
            why.append(("+" if impact > 0 else "-", f"{MACRO_FACTORS[k][0]}: {macro[k]:+.1f} × sensitivity {v:+.1f}"))
    if market:
        adj = 8 if market["above_ma200"] else -8
        adj += 4 if market["ret_3m"] > 0 else -4
        score += adj
        why.append(("+" if adj > 0 else "-",
                    f"VN-Index {'above' if market['above_ma200'] else 'below'} MA200, 3M {market['ret_3m']:+.1%}"))
    return float(np.clip(score, 0, 100)), why


def score_forecast(exp_ret: float | None, hit_rate: float | None, model_name: str, horizon: int):
    if not _ok(exp_ret):
        return 50.0, [("~", "No forecast run yet – neutral 50")]
    reliability = 0.5 if not _ok(hit_rate) else float(np.clip((hit_rate - 0.4) / 0.4, 0.1, 1.0))
    score = 50 + float(np.clip(exp_ret * 400, -45, 45)) * reliability
    why = [("+" if exp_ret > 0 else "-", f"{model_name}: expected {exp_ret:+.1%} over {horizon} sessions")]
    if _ok(hit_rate):
        why.append(("+" if hit_rate >= .55 else "-" if hit_rate < .5 else "~",
                    f"Backtest direction hit-rate {hit_rate:.0%} → reliability weight {reliability:.2f}"))
    return float(np.clip(score, 0, 100)), why


def score_liquidity(value20: float):
    if not _ok(value20):
        return 30.0, [("-", "No liquidity data")]
    bn = value20 / 1e9
    s = 100 if bn > 50 else 80 if bn > 10 else 60 if bn > 2 else 40 if bn > .5 else 15
    return float(s), [("+" if s >= 60 else "-", f"Avg 20-day traded value ≈ {bn:,.2f} bn VND")]


def rating(score: float) -> str:
    return ("STRONG BUY" if score >= 70 else "BUY" if score >= 60 else "HOLD" if score >= 45
            else "REDUCE" if score >= 35 else "SELL")


DEFAULT_WEIGHTS = {"FA": 30, "TA": 25, "Macro": 15, "Forecast": 20, "Liquidity": 10}


def composite(scores: dict, weights: dict) -> float:
    tot = sum(weights.values()) or 1
    return sum(scores[k] * weights[k] for k in weights) / tot
