"""Headless check: data → indicators → scores → Kronos forecast/backtest for one ticker."""
import sys
import time

import analysis as A
import data as D
import forecast as F

sym = sys.argv[1] if len(sys.argv) > 1 else "GMD"
h, prof, rat = D.history(sym), D.profile(sym), D.ratios_ttm(sym)
print(sym, "bars:", len(h), "last:", h["time"].iloc[-1].date(), h["close"].iloc[-1], "| ratio rows:", len(rat))
d = A.add_indicators(h)
snap = A.ta_snapshot(d)
fa = D.fa_dict(rat, prof, snap["close"])
print("FA:", {k: round(v, 3) if isinstance(v, float) else v for k, v in fa.items()})
print("TA score", A.score_ta(snap)[0], "FA score", A.score_fa(fa)[0], "Liq", A.score_liquidity(snap["value20"])[0])
print("news:", len(D.news_vietcap(sym)), "vietcap,", len(D.news_google(sym)), "google")

t = time.time()
pred, err = F.load_kronos("NeoQuasar/Kronos-small")
print("Kronos loaded:", pred is not None, err or "", f"{time.time() - t:.1f}s")
t = time.time()
fc = F.forecast(h, 20, 400, 3, pred)
print(fc["engine"], "exp_ret", round(fc["exp_ret"], 4), "end", fc["mean"][-1].round(2), f"{time.time() - t:.1f}s")
bt, summ = F.backtest(h, 20, 400, 3, 2, pred)
print(bt[["start", "pred_ret", "actual_ret", "mape", "naive_mape", "direction_ok"]].to_string())
print(summ)
