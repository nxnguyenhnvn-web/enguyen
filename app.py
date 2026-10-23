"""VN Transport & Seaport stock tracker – FA / TA / Macro / Kronos forecast scoring.

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analysis as A
import data as D
import forecast as F

st.set_page_config(page_title="enguyen · VN Transport & Ports", page_icon="⚓", layout="wide")
# FireAnt / HOSE board convention: green up, red down, yellow reference, purple ceiling, cyan floor
UP, DOWN, NEUTRAL, REF, CEIL, FLOOR = "#00c076", "#ff3b4e", "#787b86", "#f0b90b", "#d05cf2", "#1fc8e6"
BAND = {"hose": .07, "hnx": .10, "upcom": .15}  # daily price limits

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 100%;}
header[data-testid="stHeader"] {background: #11141c;}
section[data-testid="stSidebar"] {background: #161a23; border-right: 1px solid #262b38;}
h1, h2, h3 {color: #e8eaed; letter-spacing: -.01em;}
h1 {font-size: 1.6rem !important;}
div[data-testid="stMetric"] {background: #1b1f2a; border: 1px solid #262b38; border-radius: 6px; padding: 8px 12px;}
div[data-testid="stMetricLabel"] p {color: #787b86; font-size: .78rem;}
div[data-testid="stMetricValue"] {font-size: 1.25rem; color: #e8eaed;}
.stTabs [data-baseweb="tab-list"] {gap: 2px; border-bottom: 1px solid #262b38;}
.stTabs [data-baseweb="tab"] {background: #1b1f2a; padding: 6px 16px; border-radius: 6px 6px 0 0; color: #9aa0ab;}
.stTabs [aria-selected="true"] {background: #262b38; color: #f26f21 !important;}
div[data-testid="stExpander"] {background: #161a23; border: 1px solid #262b38; border-radius: 6px;}
.brand {display:flex; align-items:center; gap:10px; margin-bottom:4px;}
.brand .logo {background:#f26f21; color:#11141c; font-weight:800; border-radius:6px; padding:2px 8px; font-size:1.1rem;}
.brand .name {color:#e8eaed; font-weight:700; font-size:1.25rem;}
.board {width:100%; border-collapse:collapse; font-size:.8rem; font-variant-numeric: tabular-nums;}
.board th {color:#787b86; font-weight:500; text-align:right; padding:3px 6px; border-bottom:1px solid #262b38;}
.board th:first-child, .board td:first-child {text-align:left;}
.board td {padding:5px 6px; text-align:right; border-bottom:1px solid #1f2430; font-weight:600;}
.board tr:hover td {background:#232836;}
.board .hdr {display:flex; justify-content:space-between; padding:4px 6px; color:#e8eaed; font-weight:600;}
.pill {color:white; padding:3px 10px; border-radius:4px; font-weight:700; font-size:.85rem;}
</style>""", unsafe_allow_html=True)


def tick(price, exch):
    if exch != "hose":
        return .1
    return .01 if price < 10 else .05 if price < 50 else .1


def price_limits(d, exch):
    """Reference / ceiling / floor in 1,000 VND. UPCoM reference = previous session VWAP."""
    prev = d.iloc[-2]
    ref = prev["close"]
    if exch == "upcom" and pd.notna(prev.get("avg")):
        ref = round(prev["avg"] / .1) * .1
    t, b = tick(ref, exch), BAND[exch]
    return ref, np.floor(ref * (1 + b) / t + 1e-6) * t, np.ceil(ref * (1 - b) / t - 1e-6) * t


def board_color(price, ref, ceil=None, floor=None):
    """Price colour by FireAnt convention."""
    if ceil is not None and price >= ceil - 1e-6:
        return CEIL
    if floor is not None and price <= floor + 1e-6:
        return FLOOR
    if abs(price - ref) < 1e-6:
        return REF
    return UP if price > ref else DOWN


def fmt_vol(v):
    return f"{v / 1e6:.1f}m" if v >= 1e6 else f"{v / 1e3:.1f}k" if v >= 1e3 else f"{v:.0f}"


# ------------------------------------------------------------------ cached loaders
@st.cache_data(ttl=3600, show_spinner=False)
def load_stock(sym):
    h = D.history(sym)
    return h, D.profile(sym), D.ratios_ttm(sym)


@st.cache_data(ttl=3600, show_spinner=False)
def load_index():
    return D.history("VNINDEX")


@st.cache_data(ttl=1800, show_spinner=False)
def load_news(sym, name):
    return pd.concat([D.news_vietcap(sym), D.news_google(sym, name)], ignore_index=True)


@st.cache_resource(show_spinner="Loading Kronos model from Hugging Face…")
def get_predictor(model_name):
    return F.load_kronos(model_name)


@st.cache_data(show_spinner=False)
def run_forecast(sym, last_date, horizon, lookback, paths, windows, model_name, use_kronos):
    h, _, _ = load_stock(sym)
    pred = get_predictor(model_name)[0] if use_kronos else None
    fc = F.forecast(h, horizon, lookback, paths, pred)
    bt, summ = F.backtest(h, horizon, lookback, windows, max(3, paths // 2), pred)
    return fc, bt, summ


# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("<div class='brand'><span class='logo'>e</span><span class='name'>enguyen</span></div>"
                    "<div style='color:#787b86;font-size:.8rem;margin-bottom:8px'>VN Transport & Ports</div>",
                    unsafe_allow_html=True)
board_slot = st.sidebar.container()  # filled with the price board once data is loaded
group = st.sidebar.selectbox("Stock group", list(A.GROUPS))
extra = st.sidebar.text_input("Add tickers (comma separated)", "")
symbols = list(dict.fromkeys(A.GROUPS[group] + [s.strip().upper() for s in extra.split(",") if s.strip()]))

with st.sidebar.expander("Score weights", expanded=False):
    weights = {k: st.slider(k, 0, 50, v, 5) for k, v in A.DEFAULT_WEIGHTS.items()}

with st.sidebar.expander("Macro assumptions (-2 … +2)", expanded=False):
    st.caption("Editable views – set them to your own macro outlook. Sensitivities per sub-sector are in the Macro tab.")
    macro = {k: st.slider(lbl, -2.0, 2.0, dv, 0.5, help=hp) for k, (lbl, dv, hp) in A.MACRO_FACTORS.items()}

with st.sidebar.expander("Forecast settings", expanded=True):
    model_name = st.selectbox("Kronos model", list(F.TOKENIZERS), index=1)
    horizon = st.slider("Horizon (trading days)", 5, 60, 20, 5)
    lookback = st.slider("Lookback (days of context)", 120, 512, 400, 20)
    paths = st.slider("Sample paths", 1, 10, 5)
    windows = st.slider("Backtest windows", 2, 10, 4)
    use_kronos = st.toggle("Use Kronos (else statistical model)", True)

if use_kronos:
    predictor, kerr = get_predictor(model_name)
    if predictor is None:
        st.sidebar.error(f"Kronos unavailable → statistical fallback.\n\n{kerr}")
        use_kronos = False

# ------------------------------------------------------------------ build universe table
idx = load_index()
market = D.market_regime(idx)
if "fc" not in st.session_state:
    st.session_state.fc = {}  # sym -> (fc, bt, summary) from explicit Kronos runs


def evaluate(sym):
    h, prof, rat = load_stock(sym)
    if h.empty or len(h) < 30:  # newly listed tickers (e.g. ULG) get partial TA
        return None
    d = A.add_indicators(h)
    snap = A.ta_snapshot(d)
    fa = D.fa_dict(rat, prof, snap["close"])
    sub = A.UNIVERSE.get(sym, ("", "logistics"))[1]
    key = (sym, model_name, horizon, use_kronos)
    if key in st.session_state.fc:
        fc, _, summ = st.session_state.fc[key]
    else:  # quick statistical estimate for screening; run Kronos via buttons
        fc, summ = F.forecast(h, horizon, lookback, 200, None), {}
    scores, why = {}, {}
    scores["FA"], why["FA"] = A.score_fa(fa)
    scores["TA"], why["TA"] = A.score_ta(snap)
    scores["Macro"], why["Macro"] = A.score_macro(sub, macro, market)
    scores["Forecast"], why["Forecast"] = A.score_forecast(fc["exp_ret"], summ.get("hit_rate"), fc["engine"], horizon)
    scores["Liquidity"], why["Liquidity"] = A.score_liquidity(snap["value20"])
    total = A.composite(scores, weights)
    tp = prof.get("targetPrice")
    grp = str(prof.get("comGroupCode", "")).lower()
    exch = "hnx" if "hnx" in grp else "upcom" if "upcom" in grp else "hose"
    ref, ceil, floor = price_limits(d, exch)
    return {"sym": sym, "name": prof.get("enOrganShortName") or A.UNIVERSE.get(sym, (sym,))[0], "sub": sub,
            "exch": exch, "ref": ref, "ceil": ceil, "floor": floor,
            "color": board_color(snap["close"], ref, ceil, floor),
            "d": d, "snap": snap, "fa": fa, "rat": rat, "prof": prof, "fc": fc, "summ": summ,
            "scores": scores, "why": why, "total": total, "rating": A.rating(total),
            "chg_1d": snap["close"] / ref - 1,
            "upside": (tp / 1000 / snap["close"] - 1) if tp else np.nan}


with st.spinner(f"Loading {len(symbols)} stocks from Vietcap…"):
    res = {s: r for s in symbols if (r := evaluate(s))}

missing = [s for s in symbols if s not in res]
if missing:
    st.sidebar.warning("No/short price data: " + ", ".join(missing))


if res:  # FireAnt-style watchlist board, sorted by % change like the original
    rows_html = "".join(
        f"<tr style='color:{r['color']}'><td>{r['sym']}</td>"
        f"<td>{r['snap']['close']:.2f}</td><td>{r['chg_1d']:+.2%}</td>"
        f"<td style='color:#d1d4dc'>{fmt_vol(r['d']['volume'].iloc[-1])}</td></tr>"
        for r in sorted(res.values(), key=lambda r: -r["chg_1d"]))
    avg = np.mean([r["chg_1d"] for r in res.values()])
    board_slot.markdown(
        f"<div class='hdr board' style='display:flex;justify-content:space-between;padding:4px 6px;font-weight:600'>"
        f"<span>{group.split(' (')[0]}</span><span style='color:{board_color(avg, 0)}'>{avg:+.2%}</span></div>"
        f"<table class='board'><tr><th>Mã</th><th>Giá</th><th>+/-</th><th>KL</th></tr>{rows_html}</table>",
        unsafe_allow_html=True)


def rating_color(r):
    return {"STRONG BUY": "#15803d", "BUY": "#22c55e", "HOLD": "#eab308", "REDUCE": "#f97316", "SELL": "#dc2626"}[r]


def reasons_md(items):
    icon = {"+": "🟢", "-": "🔴", "~": "🟡"}
    return "\n".join(f"- {icon[s]} {t}" for s, t in items)


# ------------------------------------------------------------------ header
st.title(group)
c1, c2, c3, c4 = st.columns(4)
if market:
    c1.metric("VN-Index", f"{market['close']:,.2f}", f"{market['ret_3m']:+.1%} 3M")
avg1d = np.mean([r["chg_1d"] for r in res.values()]) if res else 0
c2.metric("Group avg 1D", f"{avg1d:+.2%}")
c3.metric("Stocks loaded", len(res))
c4.metric("Forecast engine", ("Kronos · " + model_name.split("/")[-1]) if use_kronos else "Statistical")

tab_rank, tab_stock, tab_fc, tab_macro = st.tabs(["🏆 Ranking", "🔎 Stock detail", "🔮 Forecast & backtest", "🌏 Macro"])

# ------------------------------------------------------------------ ranking
with tab_rank:
    rows = []
    for r in sorted(res.values(), key=lambda r: -r["total"]):
        s, fa = r["snap"], r["fa"]
        rows.append({"Ticker": r["sym"], "Name": r["name"], "Type": r["sub"], "Price": s["close"],
                     "1D": r["chg_1d"], "1M": s["ret_1m"], "3M": s["ret_3m"], "1Y": s["ret_1y"],
                     "P/E": fa.get("pe"), "P/B": fa.get("pb"), "ROE": fa.get("roe"), "D/E": fa.get("de"),
                     "Div yld": fa.get("div_yield"), "RSI": s["rsi"], "Value20 (bn)": s["value20"] / 1e9,
                     **{k: round(v) for k, v in r["scores"].items()},
                     f"Fcst {horizon}d": r["fc"]["exp_ret"], "Analyst upside": r["upside"],
                     "SCORE": round(r["total"], 1), "Rating": r["rating"]})
    tbl = pd.DataFrame(rows)
    if not tbl.empty:
        pct = ["1D", "1M", "3M", "1Y", "ROE", "Div yld", f"Fcst {horizon}d", "Analyst upside"]
        sty = (tbl.style.format({**{c: "{:+.1%}" for c in pct}, "Price": "{:.2f}", "P/E": "{:.1f}", "P/B": "{:.2f}",
                                 "D/E": "{:.2f}", "RSI": "{:.0f}", "Value20 (bn)": "{:,.1f}", "SCORE": "{:.1f}"},
                                na_rep="–")
               .map(lambda v: f"color:{UP if v >= 60 else REF if v >= 45 else DOWN};font-weight:600",
                    subset=list(A.DEFAULT_WEIGHTS) + ["SCORE"])
               .map(lambda v: f"color:{UP if v > 0 else DOWN if v < 0 else REF}" if isinstance(v, float) else "",
                    subset=["1M", "3M", "1Y", f"Fcst {horizon}d"])
               .apply(lambda col: [f"color:{res[t]['color']};font-weight:600" for t in tbl["Ticker"]],
                      subset=["Price", "1D"])
               .map(lambda v: "color:#e8eaed;font-weight:700", subset=["Ticker"])
               .map(lambda v: f"background-color:{rating_color(v)};color:white;font-weight:600", subset=["Rating"]))
        st.dataframe(sty, use_container_width=True, hide_index=True, height=38 * len(tbl) + 40)

        fig = go.Figure(go.Bar(x=tbl["Ticker"], y=tbl["SCORE"], marker_color=[rating_color(x) for x in tbl["Rating"]],
                               text=tbl["Rating"], textposition="outside"))
        fig.update_layout(height=320, margin=dict(t=20, b=10), yaxis=dict(range=[0, 100], title="Composite score"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Forecast column uses the statistical model for screening until you run Kronos "
                   "(button below or per stock in the Forecast tab). Prices in 1,000 VND.")

        if st.button(f"🔮 Run Kronos forecast + backtest for all {len(res)} stocks", disabled=not use_kronos):
            bar = st.progress(0.0)
            for i, sym in enumerate(res):
                bar.progress(i / len(res), text=f"Kronos → {sym}")
                h = res[sym]["d"]
                st.session_state.fc[(sym, model_name, horizon, use_kronos)] = run_forecast(
                    sym, str(h["time"].iloc[-1]), horizon, lookback, paths, windows, model_name, True)
            bar.progress(1.0, text="Done")
            st.rerun()

# ------------------------------------------------------------------ stock detail
with tab_stock:
    if not res:
        st.stop()
    sym = st.selectbox("Ticker", list(res), key="detail_sym")
    r = res[sym]
    d, s, fa, prof = r["d"], r["snap"], r["fa"], r["prof"]
    st.subheader(f"{sym} — {prof.get('enOrganName') or r['name']}")
    st.markdown(f"<span style='background:{rating_color(r['rating'])};color:white;padding:4px 10px;border-radius:6px;"
                f"font-weight:700'>{r['rating']} · {r['total']:.1f}/100</span> &nbsp; sub-sector **{r['sub']}** · "
                f"sector {prof.get('sector', '–')}", unsafe_allow_html=True)
    m = st.columns(6)
    m[0].metric("Price", f"{s['close']:.2f}", f"{r['chg_1d']:+.2%}")
    m[1].metric("52W range", f"{s['lo52']:.1f} – {s['hi52']:.1f}", f"{s['from_hi52']:+.1%} from high")
    m[2].metric("Market cap", f"{fa.get('market_cap_bn', np.nan):,.0f} bn" if fa else "–")
    m[3].metric("Volatility (ann.)", f"{s['volatility']:.0%}")
    tp = prof.get("targetPrice")
    m[4].metric("Vietcap target", f"{tp / 1000:.1f}" if tp else "–",
                f"{r['upside']:+.1%} ({prof.get('rating', '')})" if tp else None)
    m[5].metric("Foreign own.", f"{prof.get('foreignerPercentage', 0):.1%}" if prof.get("foreignerPercentage") else "–")

    # --- chart
    period = st.radio("Chart period", ["6M", "1Y", "2Y", "All"], index=1, horizontal=True)
    n = {"6M": 126, "1Y": 250, "2Y": 500, "All": len(d)}[period]
    v = d.tail(n)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[.55, .15, .15, .15], vertical_spacing=.02)
    fig.add_candlestick(x=v["time"], open=v["open"], high=v["high"], low=v["low"], close=v["close"], name=sym,
                        increasing_line_color=UP, decreasing_line_color=DOWN, row=1, col=1)
    for col, colr in (("ma20", "#f59e0b"), ("ma50", "#3b82f6"), ("ma200", "#a855f7")):
        fig.add_scatter(x=v["time"], y=v[col], name=col.upper(), line=dict(width=1.2, color=colr), row=1, col=1)
    fig.add_scatter(x=v["time"], y=v["bb_up"], line=dict(width=.6, dash="dot", color=NEUTRAL), name="BB", row=1, col=1)
    fig.add_scatter(x=v["time"], y=v["bb_dn"], line=dict(width=.6, dash="dot", color=NEUTRAL), showlegend=False, row=1, col=1)
    fc = r["fc"]
    fig.add_scatter(x=fc["dates"], y=fc["high"], line=dict(width=0), showlegend=False, row=1, col=1)
    fig.add_scatter(x=fc["dates"], y=fc["low"], fill="tonexty", fillcolor="rgba(14,165,233,.18)", line=dict(width=0),
                    name="Forecast P10–P90", row=1, col=1)
    fig.add_scatter(x=fc["dates"], y=fc["mean"], line=dict(color="#0ea5e9", width=2, dash="dash"),
                    name=f"{fc['engine']} forecast", row=1, col=1)
    fig.add_bar(x=v["time"], y=v["volume"], marker_color=np.where(v["close"] >= v["open"], UP, DOWN), name="Volume",
                row=2, col=1)
    fig.add_scatter(x=v["time"], y=v["rsi"], name="RSI", line=dict(color="#8b5cf6"), row=3, col=1)
    for lv in (30, 70):
        fig.add_hline(y=lv, line_dash="dot", line_color=NEUTRAL, row=3, col=1)
    fig.add_bar(x=v["time"], y=v["macd_hist"], marker_color=np.where(v["macd_hist"] >= 0, UP, DOWN), name="MACD hist",
                row=4, col=1)
    fig.add_scatter(x=v["time"], y=v["macd"], name="MACD", line=dict(color="#0ea5e9", width=1), row=4, col=1)
    fig.add_scatter(x=v["time"], y=v["macd_signal"], name="Signal", line=dict(color="#f97316", width=1), row=4, col=1)
    fig.update_layout(height=780, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10),
                      legend=dict(orientation="h", y=1.02))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(fig, use_container_width=True)

    # --- scorecard
    st.markdown("### Scorecard & reasons")
    cols = st.columns(5)
    for col, (k, sc) in zip(cols, r["scores"].items()):
        col.metric(f"{k} (w={weights[k]})", f"{sc:.0f}/100")
        with col.expander("Why", expanded=True):
            st.markdown(reasons_md(r["why"][k]))
    allwhy = [w for k in r["why"] for w in r["why"][k]]
    pos, neg = [t for s_, t in allwhy if s_ == "+"][:5], [t for s_, t in allwhy if s_ == "-"][:5]
    st.info(f"**Verdict: {r['rating']} ({r['total']:.1f}/100).** "
            f"Main supports: {'; '.join(pos) or 'none'}. Main risks: {'; '.join(neg) or 'none'}.")

    # --- fundamentals
    st.markdown(f"### Key ratios (TTM, {fa.get('period', 'n/a')})")
    if fa:
        labels = {"pe": "P/E", "pb": "P/B", "ps": "P/S", "ev_ebitda": "EV/EBITDA", "roe": "ROE", "roa": "ROA",
                  "roic": "ROIC", "gross_margin": "Gross margin", "net_margin": "Net margin", "de": "Debt/Equity",
                  "current_ratio": "Current ratio", "quick_ratio": "Quick ratio", "cash_cycle": "Cash cycle (days)",
                  "asset_turnover": "Asset turnover", "rev_growth": "Revenue growth YoY (TTM)",
                  "eps_growth": "Earnings growth YoY (TTM)", "div_yield": "Dividend yield"}
        pct_keys = A.PCT_KEYS | {"roic"}
        grid = st.columns(6)
        for i, (k, lb) in enumerate(labels.items()):
            val = fa.get(k)
            grid[i % 6].metric(lb, "n/a" if val is None or pd.isna(val) else f"{val:.1%}" if k in pct_keys else f"{val:.2f}")
        rat = r["rat"].tail(20)
        f2 = make_subplots(rows=1, cols=3, subplot_titles=("ROE / ROA / Net margin", "P/E & P/B", "TTM revenue & earnings (bn)"))
        for c_, nm in (("roe", "ROE"), ("roa", "ROA"), ("afterTaxProfitMargin", "Net margin")):
            f2.add_scatter(x=rat["period"], y=rat[c_], name=nm, row=1, col=1)
        f2.add_scatter(x=rat["period"], y=rat["pe"], name="P/E", row=1, col=2)
        f2.add_scatter(x=rat["period"], y=rat["pb"], name="P/B", row=1, col=2)
        f2.add_bar(x=rat["period"], y=rat["ttm_revenue"] / 1e9, name="Revenue", row=1, col=3)
        f2.add_bar(x=rat["period"], y=rat["ttm_earnings"] / 1e9, name="Earnings", row=1, col=3)
        f2.update_yaxes(tickformat=".0%", row=1, col=1)
        f2.update_layout(height=340, margin=dict(t=40, b=10), legend=dict(orientation="h"))
        st.plotly_chart(f2, use_container_width=True)
    else:
        st.warning("No fundamental data from Vietcap for this ticker.")

    # --- news
    st.markdown("### News")
    nw = load_news(sym, prof.get("viOrganShortName", ""))
    if nw.empty:
        st.write("No news found.")
    else:
        nw = nw.drop_duplicates("title").sort_values("date", ascending=False).head(20)
        st.markdown("\n".join(f"- `{row.date}` [{row.title}]({row.link}) — *{row.source}*" for row in nw.itertuples()))

# ------------------------------------------------------------------ forecast & backtest
with tab_fc:
    sym_f = st.selectbox("Ticker", list(res), key="fc_sym")
    rf = res[sym_f]
    st.caption(f"Engine: **{'Kronos · ' + model_name if use_kronos else 'Statistical GBM'}** · horizon {horizon} · "
               f"lookback {lookback} · {paths} paths · {windows} backtest windows. "
               "Kronos (NeoQuasar) is a foundation model pre-trained on K-line data from 45+ exchanges; "
               "it samples future OHLCV paths from the recent candles.")
    if st.button(f"Run forecast + backtest for {sym_f}", type="primary"):
        with st.spinner("Forecasting… (Kronos on CPU takes a few seconds per path)"):
            out = run_forecast(sym_f, str(rf["d"]["time"].iloc[-1]), horizon, lookback, paths, windows,
                               model_name, use_kronos)
        st.session_state.fc[(sym_f, model_name, horizon, use_kronos)] = out
        st.rerun()

    key = (sym_f, model_name, horizon, use_kronos)
    if key in st.session_state.fc:
        fc, bt, summ = st.session_state.fc[key]
        last = rf["snap"]["close"]
        k = st.columns(5)
        k[0].metric("Expected return", f"{fc['exp_ret']:+.1%}", f"target {fc['mean'][-1]:.2f}", delta_color="off")
        k[1].metric("P10 / P90 at horizon", f"{fc['low'][-1]:.2f} / {fc['high'][-1]:.2f}")
        if summ:
            k[2].metric("Direction hit-rate", f"{summ['hit_rate']:.0%}")
            k[3].metric("MAPE vs naive", f"{summ['mape']:.1%}", f"{summ['mape'] - summ['naive_mape']:+.1%} vs random-walk",
                        delta_color="inverse")
            k[4].metric("P10–P90 coverage", f"{summ['band_coverage']:.0%}")
        hist = rf["d"].tail(120)
        g = go.Figure()
        g.add_candlestick(x=hist["time"], open=hist["open"], high=hist["high"], low=hist["low"], close=hist["close"],
                          name=sym_f, increasing_line_color=UP, decreasing_line_color=DOWN)
        for p in fc["paths"][:10]:
            g.add_scatter(x=fc["dates"], y=p, line=dict(width=.7, color="rgba(14,165,233,.35)"), showlegend=False)
        g.add_scatter(x=fc["dates"], y=fc["mean"], line=dict(width=3, color="#0284c7"), name="Mean forecast")
        g.update_layout(height=460, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10))
        g.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(g, use_container_width=True)

        # reasoning
        s = rf["snap"]
        trend = "up-trend" if s["close"] > s["ma50"] else "down-trend"
        beats = summ and summ["mape"] < summ["naive_mape"]
        st.markdown(f"""**Why this forecast?**
- The model sees the last {lookback} candles: the stock is in a **{trend}** (price {s['close']:.2f} vs MA50 {s['ma50']:.2f}),
  3-month return {s['ret_3m']:+.1%}, annualised volatility {s['volatility']:.0%} → the P10–P90 band spans
  {(fc['high'][-1] - fc['low'][-1]) / last:.0%} of price.
- Mean path ends at **{fc['mean'][-1]:.2f}** ({fc['exp_ret']:+.1%}); {sum(p[-1] > last for p in fc['paths'])}/{len(fc['paths'])} sampled paths finish above today's close.
- Backtest: {'the model **beat** the random-walk benchmark' if beats else 'the model did **not** beat a random-walk benchmark'}
  {'' if not summ else f"(MAPE {summ['mape']:.1%} vs {summ['naive_mape']:.1%}), direction right {summ['hit_rate']:.0%} of the time"} —
  {'its signal gets more weight in the Forecast score.' if summ and summ['hit_rate'] >= .55 else 'so its signal is down-weighted in the Forecast score.'}
""")
        if not bt.empty:
            st.markdown("**Backtest windows** (forecast made at *start*, compared with what actually happened)")
            st.dataframe(bt.style.format({"last_close": "{:.2f}", "pred_end": "{:.2f}", "actual_end": "{:.2f}",
                                          "pred_ret": "{:+.1%}", "actual_ret": "{:+.1%}", "mape": "{:.1%}",
                                          "naive_mape": "{:.1%}"}), use_container_width=True, hide_index=True)
    else:
        st.info("Press the button to run the forecast and its rolling backtest for this ticker.")

# ------------------------------------------------------------------ macro
with tab_macro:
    st.markdown("Macro score = sector sensitivity × your macro view (sidebar) + VN-Index regime (MA200 & 3M trend).")
    sens = pd.DataFrame(A.SENSITIVITY).T.rename(columns={k: v[0] for k, v in A.MACRO_FACTORS.items()})
    st.dataframe(sens.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:+.1f}"),
                 use_container_width=True)
    if not idx.empty:
        iv = idx.tail(500).copy()
        iv["ma200"] = idx["close"].rolling(200).mean().tail(500)
        g = go.Figure()
        g.add_scatter(x=iv["time"], y=iv["close"], name="VN-Index")
        g.add_scatter(x=iv["time"], y=iv["ma200"], name="MA200", line=dict(dash="dash"))
        g.update_layout(height=360, margin=dict(t=10, b=10))
        st.plotly_chart(g, use_container_width=True)
    st.markdown("""**Macro checklist for Vietnamese ports & shipping**
- **Trade volumes** – monthly GSO export/import data; container throughput at Hai Phong / Cai Mep.
- **Freight rates** – Drewry WCI / SCFI (container), Baltic Dirty/Clean Tanker (PVT, VTO, VIP, PJT).
- **Rates & FX** – SBV policy rate, USD/VND; ports/shipping earn partly in USD.
- **Tariffs** – US reciprocal tariffs / trans-shipment rules hit export volumes.
- **FDI & industrial parks** – northern FDI (Hai Phong, Quang Ninh) feeds GMD, PHP, VSC, CQN, DXP.
- **Infrastructure** – Lach Huyen terminals, Cai Mep–Thi Vai, Lien Chieu (CDN), port tariff floor changes.""")

st.caption("Data: Vietcap public APIs & Google News. Scores are rule-based and for research only — not investment advice.")
