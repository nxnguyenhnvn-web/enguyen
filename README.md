# VN Transport & Seaport Stock Tracker

Streamlit app for Vietnamese industrial-transport, seaport and shipping stocks
(GMD, PHP, VSC, SGP, CDN, DVP, TCL, HAH, PVT, VTO, PJT, SCS, …).

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Locally: `streamlit run app.py --server.baseUrlPath enguyen` → http://localhost:8501/enguyen.
Public: https://enguyen.streamlit.app (Streamlit Community Cloud, main file `app.py`, Python 3.12).
The FireAnt-style dark theme is set in `.streamlit/config.toml`.
`kronos_src/` holds the official Kronos model code (github.com/shiyu-coder/Kronos).
Model weights (`NeoQuasar/Kronos-mini|small|base`) download from Hugging Face on first use.

## What it does
| Tab | Content |
|---|---|
| Ranking | All stocks in the group with price, returns, P/E, P/B, ROE, D/E, RSI, liquidity, 5 sub-scores, composite score and rating |
| Stock detail | Candles + MA20/50/200, Bollinger, volume, RSI, MACD, forecast cone; scorecard with reasons; 17 TTM ratios + history; Vietcap analyst target; news |
| Forecast & backtest | Kronos sampled paths, P10–P90 band, rolling-origin backtest (hit-rate, MAPE vs random-walk, band coverage) and a written explanation |
| Macro | Sub-sector × macro-factor sensitivity matrix, VN-Index regime |

**Composite score** = weighted FA (30) + TA (25) + Macro (15) + Forecast (20) + Liquidity (10); weights editable in the sidebar.
Ratings: ≥70 STRONG BUY · ≥60 BUY · ≥45 HOLD · ≥35 REDUCE · else SELL.

- **FA**: ROE, P/E, P/B, D/E, net margin, TTM revenue & earnings growth YoY, dividend yield.
- **TA**: price vs MAs, golden/death cross, RSI zone, MACD, 3M momentum, volume confirmation.
- **Macro**: your views on trade, freight, rates, FX, oil, FDI, infrastructure, tariffs (sidebar) × sub-sector sensitivity, plus VN-Index above/below MA200.
- **Forecast**: expected return from Kronos (or GBM fallback), scaled by backtest direction hit-rate.
- **Liquidity**: 20-day average traded value.

Data: Vietcap public APIs (prices, TTM ratios, profile, news) + Google News RSS. For research only — not investment advice.
