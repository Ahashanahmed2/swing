# trand_signal.py  (v3 – downtrend = current only)
import pandas as pd
import os
import json
from datetime import datetime

# --------------------------------------------------
# Load config
# --------------------------------------------------
CONFIG_PATH = "./config.json"
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    TOTAL_CAPITAL = float(config.get("total_capital", 500_000))
    RISK_PERCENT = float(config.get("risk_percent", 0.01))
except Exception as e:
    print(f"⚠️ Config load error: {e}, using defaults")
    TOTAL_CAPITAL = 500_000
    RISK_PERCENT = 0.01

# --------------------------------------------------
# Signal maintenance helper (UPTREND only)
# --------------------------------------------------
def merge_and_track_new_symbols(old_df, new_df, symbol_col="symbol"):
    if old_df is None or old_df.empty:
        return new_df, new_df.copy()

    old_symbols = set(old_df[symbol_col])
    new_symbols = set(new_df[symbol_col])

    common = old_symbols & new_symbols
    preserved = old_df[old_df[symbol_col].isin(common)]
    brand_new = new_df[~new_df[symbol_col].isin(old_symbols)]

    final_df = pd.concat([preserved, brand_new], ignore_index=True)
    return final_df, brand_new

# --------------------------------------------------
# Swing SL logic
# --------------------------------------------------
def find_sl_from_buy(group):
    if len(group) < 5:
        return None
    try:
        A, B, C, D, E = group.iloc[-1], group.iloc[-2], group.iloc[-3], group.iloc[-4], group.iloc[-5]

        if (A["close"] > B["high"] and
            B["low"] < C["low"] and
            B["high"] < C["high"] and
            C["high"] < D["high"] and
            C["low"] < D["low"]):
            return float(B["low"])

        if (A["close"] > B["high"] and
            B["high"] < C["high"] and
            B["low"] > C["low"] and
            C["high"] < D["high"] and
            C["low"] < D["low"] and
            D["high"] < E["high"] and
            D["low"] < E["low"]):
            return float(C["low"])
    except:
        return None
    return None

# --------------------------------------------------
# Swing TP logic
# --------------------------------------------------
def find_tp_from_anchor(group, anchor_date):
    try:
        idxs = group.index[group["date"] == anchor_date]
        if len(idxs) == 0:
            return None
        anchor_idx = idxs[0]

        for i in range(anchor_idx, 2, -1):
            sb, sa, s = group.iloc[i-2], group.iloc[i-1], group.iloc[i]
            if s["high"] > sa["high"] >= sb["high"]:
                return float(s["high"])
    except:
        return None
    return None

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def create_uptrend_downtrend_signals():
    mongodb_csv = "./csv/mongodb.csv"
    trand_base_dir = "./csv/trand/"
    output_dir = "./csv/"
    ai_output_dir = "./output/ai_signal"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ai_output_dir, exist_ok=True)

    uptrend_file = os.path.join(output_dir, "uptrand.csv")
    downtrend_file = os.path.join(output_dir, "downtrand.csv")

    old_up_df = pd.read_csv(uptrend_file) if os.path.exists(uptrend_file) else None

    df = pd.read_csv(mongodb_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])
    groups = df.groupby("symbol", sort=False)

    uptrend_rows = []
    downtrend_rows = []

    for symbol, group in groups:
        if len(group) < 5:
            continue

        latest = group.iloc[-1]
        latest_close = float(latest["close"])
        latest_date = latest["date"]

        symbol_dir = os.path.join(trand_base_dir, symbol)
        high_file = os.path.join(symbol_dir, "high.csv")
        low_file = os.path.join(symbol_dir, "low.csv")

        # ---------------- UPTREND ----------------
        if os.path.exists(high_file):
            high_df = pd.read_csv(high_file)
            if len(high_df) >= 2:
                high_df["date"] = pd.to_datetime(high_df["date"])
                p1, p2 = high_df.iloc[0], high_df.iloc[1]

                if float(p1["price"]) < float(p2["price"]) < latest_close:
                    buy = latest_close
                    sl = find_sl_from_buy(group)
                    if sl is None or sl >= buy:
                        continue

                    tp = find_tp_from_anchor(group, p2["date"])
                    if tp is None or tp <= buy:
                        continue

                    risk = buy - sl
                    pos = max(1, int((TOTAL_CAPITAL * RISK_PERCENT) / risk))
                    rrr = (tp - buy) / risk

                    uptrend_rows.append({
                        "date": latest_date,
                        "symbol": symbol,
                        "buy": round(buy, 2),
                        "sl": round(sl, 2),
                        "tp": round(tp, 2),
                        "position_size": pos,
                        "exposure_bdt": round(pos * buy, 2),
                        "actual_risk_bdt": round(pos * risk, 2),
                        "diff": round(risk, 4),
                        "RRR": round(rrr, 2),
                        "p1": p1["date"],
                        "p2": p2["date"]
                    })

        # ---------------- DOWNTREND (CURRENT ONLY) ----------------
        if os.path.exists(low_file):
            low_df = pd.read_csv(low_file)
            if len(low_df) >= 2:
                low_df["date"] = pd.to_datetime(low_df["date"])
                p1, p2 = low_df.iloc[0], low_df.iloc[1]

                if float(p1["price"]) > float(p2["price"]) > latest_close:
                    downtrend_rows.append({
                        "date": latest_date.strftime("%Y-%m-%d"),
                        "symbol": symbol,
                        "close": round(latest_close, 2),
                        "p1": p1["date"].strftime("%Y-%m-%d"),
                        "p2": p2["date"].strftime("%Y-%m-%d")
                    })

    # ---------------- SAVE UPTREND ----------------
    if uptrend_rows:
        new_up = pd.DataFrame(uptrend_rows)
        new_up["date"] = new_up["date"].dt.strftime("%Y-%m-%d")
        new_up["p1"] = new_up["p1"].dt.strftime("%Y-%m-%d")
        new_up["p2"] = new_up["p2"].dt.strftime("%Y-%m-%d")

        final_up, brand_new = merge_and_track_new_symbols(old_up_df, new_up)
        final_up = final_up.sort_values(["RRR", "diff"], ascending=[False, True])
        final_up.insert(0, "no", range(1, len(final_up) + 1))
        final_up.to_csv(uptrend_file, index=False)

        if not brand_new.empty:
            brand_new.to_csv(os.path.join(ai_output_dir, "uptrand.csv"), index=False)

    # ---------------- SAVE DOWNTREND (OVERWRITE) ----------------
    if downtrend_rows:
        down_df = pd.DataFrame(downtrend_rows)
        down_df.insert(0, "no", range(1, len(down_df) + 1))
        down_df.to_csv(downtrend_file, index=False)
    else:
        pd.DataFrame(columns=["no", "date", "symbol", "close", "p1", "p2"]).to_csv(downtrend_file, index=False)

# --------------------------------------------------
if __name__ == "__main__":
    create_uptrend_downtrend_signals()