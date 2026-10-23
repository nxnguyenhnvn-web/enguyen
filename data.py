"""Data access: Vietcap public APIs (prices, TTM ratios, profile, news) + Google News RSS."""
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Content-Type": "application/json",
     "Referer": "https://trading.vietcap.com.vn/", "Origin": "https://trading.vietcap.com.vn"}
IQ = "https://iq.vietcap.com.vn/api/iq-insight-service/v1"


def _get(url, payload=None, raw=False):
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload else None, headers=H)
    with urllib.request.urlopen(req, timeout=20) as r:
        body = r.read()
    return body if raw else json.loads(body)


UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
      "Accept": "application/json, text/plain, */*"}


def _frame(d, scale=1.0):
    return pd.DataFrame({"time": pd.to_datetime(np.array(d["t"], dtype="int64"), unit="s").normalize(),
                         "open": np.array(d["o"], float) / scale, "high": np.array(d["h"], float) / scale,
                         "low": np.array(d["l"], float) / scale, "close": np.array(d["c"], float) / scale,
                         "volume": np.array(d["v"], float)})


def _vietcap(symbol, days):
    d = _get("https://trading.vietcap.com.vn/api/chart/OHLCChart/gap-chart",
             {"timeFrame": "ONE_DAY", "symbols": [symbol], "to": int(time.time()) + 86400, "countBack": days})[0]
    df = _frame(d, 1 if "INDEX" in symbol else 1000)
    if "INDEX" not in symbol and d.get("accumulatedValue"):
        # value in million VND → session VWAP in 1,000 VND (UPCoM reference price)
        df["avg"] = np.array(d["accumulatedValue"], dtype=float) * 1000 / df["volume"].replace(0, np.nan)
    return df


def _entrade(symbol, days):  # DNSE; reachable from outside Vietnam
    kind = "index" if "INDEX" in symbol else "stock"
    to = int(time.time()) + 86400
    url = (f"https://services.entrade.com.vn/chart-api/v2/ohlcs/{kind}?from={to - days * 2 * 86400}&to={to}"
           f"&symbol={symbol}&resolution=1D")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
        return _frame(json.loads(r.read()))


def _vndirect(symbol, days):
    to = int(time.time()) + 86400
    url = (f"https://dchart-api.vndirect.com.vn/dchart/history?resolution=D&symbol={symbol}"
           f"&from={to - days * 2 * 86400}&to={to}")
    h = {**UA, "Origin": "https://dchart.vndirect.com.vn", "Referer": "https://dchart.vndirect.com.vn/"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=20) as r:
        return _frame(json.loads(r.read()))


SOURCES = [("Vietcap", _vietcap), ("Entrade", _entrade), ("VNDirect", _vndirect)]


def history(symbol: str, days: int = 1500) -> pd.DataFrame:
    """Daily OHLCV from the first source that answers (Vietcap blocks non-VN servers).
    Stock prices in 1,000 VND, index in points; df.attrs['source'] names the source. Never raises."""
    for name, fn in SOURCES:
        try:
            df = fn(symbol, days)
        except Exception:
            continue
        if df.empty:
            continue
        if "INDEX" not in symbol:
            df = df[df["volume"] > 0]
        df = df.drop_duplicates("time").tail(days).reset_index(drop=True)
        df.attrs["source"] = name
        return df
    return pd.DataFrame()


def profile(symbol: str) -> dict:
    try:
        return _get(f"{IQ}/company/{symbol}")["data"] or {}
    except Exception:
        return {}


def ratios_ttm(symbol: str) -> pd.DataFrame:
    """Quarterly TTM ratio history (oldest → newest)."""
    try:
        df = pd.DataFrame(_get(f"{IQ}/company/{symbol}/statistics-financial")["data"])
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df = df[df["ratioType"] == "RATIO_TTM"].copy() if "ratioType" in df else df
    df["period"] = df["yearReport"].astype(str) + "-Q" + df["quarter"].astype(str)
    df = df.sort_values(["yearReport", "quarter"]).reset_index(drop=True)
    # back out TTM revenue & earnings from valuation multiples to compute YoY growth
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ttm_revenue"] = np.where(df["ps"] > 0, df["marketCap"] / df["ps"], np.nan)
        df["ttm_earnings"] = np.where(df["pe"] > 0, df["marketCap"] / df["pe"], np.nan)
    df["rev_growth"] = df["ttm_revenue"] / df["ttm_revenue"].shift(4) - 1
    df["eps_growth"] = df["ttm_earnings"] / df["ttm_earnings"].shift(4) - 1
    return df


def fa_dict(r: pd.DataFrame, prof: dict, price: float) -> dict:
    if r.empty:
        return {}
    last = r.iloc[-1]
    g = lambda k: float(last[k]) if k in last and pd.notna(last[k]) else np.nan
    dps = prof.get("dividendPerShareTsr")
    return {
        "period": last["period"], "pe": g("pe"), "pb": g("pb"), "ps": g("ps"), "ev_ebitda": g("evToEbitda"),
        "roe": g("roe"), "roa": g("roa"), "roic": g("roic"), "gross_margin": g("grossMargin"),
        "net_margin": g("afterTaxProfitMargin"), "de": g("debtToEquity"), "current_ratio": g("currentRatio"),
        "quick_ratio": g("quickRatio"), "cash_cycle": g("cashCycle"), "asset_turnover": g("assetTurnover"),
        "rev_growth": g("rev_growth"), "eps_growth": g("eps_growth"),
        # Vietcap reports 0 when it has no dividend data, so treat 0 as unknown
        "div_yield": (dps / 1000 / price) if dps and price else (g("dividendYield") or np.nan),
        "market_cap_bn": g("marketCap") / 1e9,
    }


def news_vietcap(symbol: str, n: int = 15) -> pd.DataFrame:
    try:
        to = pd.Timestamp.today()
        u = (f"{IQ}/news?ticker={symbol}&page=0&size={n}&languageId=1"
             f"&fromDate={(to - pd.Timedelta(days=365)):%Y%m%d}&toDate={to:%Y%m%d}")
        items = _get(u)["data"]["content"]
        return pd.DataFrame([{"date": i.get("publicDate", "")[:10], "title": i.get("newsTitle"),
                              "source": "Vietcap IQ",
                              "link": "https://www.google.com/search?q=" + urllib.parse.quote(i.get("newsTitle") or "")}
                             for i in items])
    except Exception:
        return pd.DataFrame()


def news_google(symbol: str, name: str = "", n: int = 15) -> pd.DataFrame:
    try:
        q = urllib.parse.quote(f'"{symbol}" cổ phiếu {name}'.strip())
        x = _get(f"https://news.google.com/rss/search?q={q}&hl=vi&gl=VN&ceid=VN:vi", raw=True)
        rows = []
        for it in ET.fromstring(x).iter("item"):
            rows.append({"date": pd.to_datetime(it.findtext("pubDate"), errors="coerce").strftime("%Y-%m-%d"),
                         "title": it.findtext("title"), "source": it.findtext("source") or "Google News",
                         "link": it.findtext("link")})
        return pd.DataFrame(rows).head(n)
    except Exception:
        return pd.DataFrame()


def market_regime(idx: pd.DataFrame) -> dict | None:
    if idx.empty or len(idx) < 200:
        return None
    c = idx["close"]
    return {"close": float(c.iloc[-1]), "above_ma200": bool(c.iloc[-1] > c.tail(200).mean()),
            "ret_3m": float(c.iloc[-1] / c.iloc[-64] - 1)}
