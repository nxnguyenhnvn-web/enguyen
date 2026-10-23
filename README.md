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

| ML ranking | LightGBM cross-sectional ranker over ~125 liquid stocks: today's ML score with drivers, walk-forward backtest vs Kronos and momentum, feature importance |

**Composite score** = weighted FA (25) + TA (20) + Macro (15) + ML (20) + Forecast (10) + Liquidity (10); weights editable in the sidebar.

### ML ranker
- Universe: VN100 + transport/port stocks; only names trading > 1 bn VND/day are scored.
- Features (point-in-time): momentum/technicals, foreign net buying and ownership change (VNDirect),
  earnings growth & acceleration, E/P, B/P, S/P at the current price, ROE, margins, leverage (Vietcap TTM ratios,
  used 45 days after quarter-end, 90 days for Q4).
- Target: 20-session forward return rank. Retrained every month in the backtest (walk-forward, no look-ahead).
- Retrain: `python train_ranker.py --kronos 24` (≈ 1 h with the Kronos comparison), then commit `models/`.
Ratings: ≥70 STRONG BUY · ≥60 BUY · ≥45 HOLD · ≥35 REDUCE · else SELL.

- **FA**: ROE, P/E, P/B, D/E, net margin, TTM revenue & earnings growth YoY, dividend yield.
- **TA**: price vs MAs, golden/death cross, RSI zone, MACD, 3M momentum, volume confirmation.
- **Macro**: your views on trade, freight, rates, FX, oil, FDI, infrastructure, tariffs (sidebar) × sub-sector sensitivity, plus VN-Index above/below MA200.
- **Forecast**: expected return from Kronos (or GBM fallback), scaled by backtest direction hit-rate.
- **Liquidity**: 20-day average traded value.

Data: Vietcap public APIs (prices, TTM ratios, profile, news) + Google News RSS. For research only — not investment advice.
