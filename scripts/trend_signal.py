# uptrend_cond12_only.py
# কন্ডিশন ১ ও ২ ঠিক রেখে শুধু UPTREND সিগন্যাল তৈরি
import pandas as pd
import os, json, numpy as np
from datetime import datetime

# ---------- 1. Config ----------
CFG = "./config.json"
try:
    with open(CFG) as f:
        c = json.load(f)
    CAPITAL = float(c.get("total_capital", 500_000))
    RISK_PCT = float(c.get("risk_percent", 0.01))
except Exception as e:
    print("⚠️ Config fail → defaults")
    CAPITAL, RISK_PCT = 500_000, 0.01

# ---------- 2. Paths ----------
IN_FILE   = "./csv/mongodb.csv"
OUT_FILE  = "./csv/uptrand.csv"
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ---------- 3. Load & prep ----------
if not os.path.exists(IN_FILE):
    raise FileNotFoundError(IN_FILE)

df = pd.read_csv(IN_FILE)
req = ["date","symbol","close","high","low"]
assert all(c in df.columns for c in req), "Required cols missing"

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values(["symbol","date"])

# ---------- 4. Signal engine ----------
signals = []

for sym, g in df.groupby("symbol", sort=False):
    if len(g) < 5:
        continue
    A,B,C,D,E = [g.iloc[-i] for i in range(1,6)]
    buy = sl = tp = None

    # === Condition-1 ===
    if (A["close"] > B["high"] and
        B["low"]  < C["low"]  and
        B["high"] < C["high"] and
        C["high"] < D["high"] and
        C["low"]  < D["low"]):
        buy, sl = A["close"], B["low"]

    # === Condition-2 ===
    elif (A["close"] > B["high"] and
          B["high"] < C["high"] and
          B["low"]  > C["low"]  and
          C["high"] < D["high"] and
          C["low"]  < D["low"]  and
          D["high"] < E["high"] and
          D["low"]  < E["low"]):
        buy, sl = A["close"], C["low"]

    if buy is None or sl is None or sl >= buy:
        continue

    # --- TP scan backward from SL-source row ---
    sl_row = B if sl == B["low"] else C
    try:
        sl_idx = g[g["date"]==sl_row["date"]].index[0]
    except IndexError:
        sl_idx = (abs(g["date"] - sl_row["date"])).idxmin()

    for i in range(sl_idx-1, 1, -1):
        try:
            sb, sa, s = g.iloc[i-2], g.iloc[i-1], g.iloc[i]
        except IndexError:
            break
        if s["high"] > sa["high"] >= sb["high"]:
            tp = s["high"]
            break
    if tp is None or tp <= buy:
        continue

    # --- Position sizing ---
    risk_share = buy - sl
    if risk_share <= 0:
        continue
    pos = max(1, int((CAPITAL * RISK_PCT) / risk_share))
    signals.append({
        "date": A["date"],
        "symbol": sym,
        "buy": round(float(buy),2),
        "sl":  round(float(sl),2),
        "tp":  round(float(tp),2),
        "position_size": pos,
        "exposure_bdt": round(pos * buy, 2),
        "actual_risk_bdt": round(pos * risk_share, 2),
        "diff": round(risk_share, 4),
        "RRR": round((tp - buy) / risk_share, 2)
    })

# ---------- 5. Build DataFrame ----------
if signals:
    res = pd.DataFrame(signals)
    res = res[(res["buy"]>res["sl"]) & (res["tp"]>res["buy"]) & (res["RRR"]>0)]
    res = res.sort_values(["RRR","diff"], ascending=[False,True])
    res.insert(0, "no", range(1, len(res)+1))
    res["date"] = res["date"].dt.strftime("%Y-%m-%d")
else:
    res = pd.DataFrame(columns=["no","date","symbol","buy","sl","tp",
                                "position_size","exposure_bdt","actual_risk_bdt",
                                "diff","RRR"])

# ---------- 6. Save ----------
res.to_csv(OUT_FILE, index=False)
print(f"✅ Uptrend signals saved: {len(res)} rows → {OUT_FILE}")
