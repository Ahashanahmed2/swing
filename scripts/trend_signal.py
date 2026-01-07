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
    
    if new_df.empty:
        return old_df, pd.DataFrame()

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
    try:
        if len(group) < 5:
            return None
        
        # সুরক্ষিতভাবে ইনডেক্স অ্যাক্সেস
        indices = [-1, -2, -3, -4, -5]
        if any(abs(idx) > len(group) for idx in indices):
            return None
            
        A, B, C, D, E = group.iloc[-1], group.iloc[-2], group.iloc[-3], group.iloc[-4], group.iloc[-5]

        # কন্ডিশন ১
        if (A["close"] > B["high"] and
            B["low"] < C["low"] and
            B["high"] < C["high"] and
            C["high"] < D["high"] and
            C["low"] < D["low"]):
            return float(B["low"])

        # কন্ডিশন ২
        if (A["close"] > B["high"] and
            B["high"] < C["high"] and
            B["low"] > C["low"] and
            C["high"] < D["high"] and
            C["low"] < D["low"] and
            D["high"] < E["high"] and
            D["low"] < E["low"]):
            return float(C["low"])
    except Exception as e:
        print(f"Error in find_sl_from_buy: {e}")
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

        # সুরক্ষিত লুপ
        for i in range(min(anchor_idx, len(group)-1), 2, -1):
            if i-2 < 0:
                break
            sb, sa, s = group.iloc[i-2], group.iloc[i-1], group.iloc[i]
            if s["high"] > sa["high"] >= sb["high"]:
                return float(s["high"])
    except Exception as e:
        print(f"Error in find_tp_from_anchor: {e}")
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

    # ডিরেক্টরি তৈরি
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ai_output_dir, exist_ok=True)
    os.makedirs(trand_base_dir, exist_ok=True)

    uptrend_file = os.path.join(output_dir, "uptrand.csv")
    downtrend_file = os.path.join(output_dir, "downtrand.csv")

    # পুরনো ডেটা লোড (সুরক্ষিতভাবে)
    try:
        old_up_df = pd.read_csv(uptrend_file) if os.path.exists(uptrend_file) and os.path.getsize(uptrend_file) > 0 else None
    except Exception as e:
        print(f"⚠️ Error loading old uptrend file: {e}")
        old_up_df = None

# মূল ডেটা লোড
    if not os.path.exists(mongodb_csv):
        print(f"❌ mongodb.csv not found at {mongodb_csv}")
        return
        
    try:
        df = pd.read_csv(mongodb_csv)
        if df.empty:
            print("⚠️ mongodb.csv is empty")
            return
    except Exception as e:
        print(f"❌ Error reading mongodb.csv: {e}")
        return

    # ডেটা প্রিপ্রসেস
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df = df.dropna(subset=['date'])
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
        if os.path.exists(high_file) and os.path.getsize(high_file) > 0:
            try:
                high_df = pd.read_csv(high_file)
                if len(high_df) >= 2:
                    high_df["date"] = pd.to_datetime(high_df["date"], errors='coerce')
                    high_df = high_df.dropna(subset=['date'])
                    
                    if len(high_df) < 2:
                        continue
                        
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
                        if risk <= 0:
                            continue
                            
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
            except Exception as e:
                print(f"Error processing {symbol} uptrend: {e}")
                continue

        # ---------------- DOWNTREND ----------------
        if os.path.exists(low_file) and os.path.getsize(low_file) > 0:
            try:
                low_df = pd.read_csv(low_file)
                if len(low_df) >= 2:
                    low_df["date"] = pd.to_datetime(low_df["date"], errors='coerce')
                    low_df = low_df.dropna(subset=['date'])
                    
                    if len(low_df) < 2:
                        continue
                        
                    p1, p2 = low_df.iloc[0], low_df.iloc[1]

                    if float(p1["price"]) > float(p2["price"]) > latest_close:
                        downtrend_rows.append({
                            "date": latest_date.strftime("%Y-%m-%d"),

"symbol": symbol,
                            "close": round(latest_close, 2),
                            "p1": p1["date"].strftime("%Y-%m-%d") if hasattr(p1["date"], 'strftime') else str(p1["date"]),
                            "p2": p2["date"].strftime("%Y-%m-%d") if hasattr(p2["date"], 'strftime') else str(p2["date"])
                        })
            except Exception as e:
                print(f"Error processing {symbol} downtrend: {e}")
                continue

    # ---------------- SAVE UPTREND ----------------
    try:
        if uptrend_rows:
            new_up = pd.DataFrame(uptrend_rows)
            new_up["date"] = pd.to_datetime(new_up["date"]).dt.strftime("%Y-%m-%d")
            new_up["p1"] = pd.to_datetime(new_up["p1"]).dt.strftime("%Y-%m-%d")
            new_up["p2"] = pd.to_datetime(new_up["p2"]).dt.strftime("%Y-%m-%d")

            final_up, brand_new = merge_and_track_new_symbols(old_up_df, new_up)
            if not final_up.empty:
                final_up = final_up.sort_values(["RRR", "diff"], ascending=[False, True])
                final_up.insert(0, "no", range(1, len(final_up) + 1))
                final_up.to_csv(uptrend_file, index=False)

                if not brand_new.empty:
                    brand_new.to_csv(os.path.join(ai_output_dir, "uptrand.csv"), index=False)
        else:
            # নতুন সিগন্যাল না থাকলে পুরনোটা রাখো বা খালি ফাইল রাখো
            if old_up_df is not None and not old_up_df.empty:
                old_up_df.to_csv(uptrend_file, index=False)
            else:
                pd.DataFrame(columns=["no", "date", "symbol", "buy", "sl", "tp", "position_size", 
                                     "exposure_bdt", "actual_risk_bdt", "diff", "RRR", "p1", "p2"]).to_csv(uptrend_file, index=False)
    except Exception as e:
        print(f"❌ Error saving uptrend: {e}")

    # ---------------- SAVE DOWNTREND ----------------
    try:
        if downtrend_rows:
            down_df = pd.DataFrame(downtrend_rows)
            down_df.insert(0, "no", range(1, len(down_df) + 1))
            down_df.to_csv(downtrend_file, index=False)
        else:
            pd.DataFrame(columns=["no", "date", "symbol", "close", "p1", "p2"]).to_csv(downtrand_file, index=False)
    except Exception as e:
        print(f"❌ Error saving downtrend: {e}")

# --------------------------------------------------
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
    
    if new_df.empty:
        return old_df, pd.DataFrame()

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
    try:
        if len(group) < 5:
            return None
        
        # সুরক্ষিতভাবে ইনডেক্স অ্যাক্সেস
        indices = [-1, -2, -3, -4, -5]
        if any(abs(idx) > len(group) for idx in indices):
            return None
            
        A, B, C, D, E = group.iloc[-1], group.iloc[-2], group.iloc[-3], group.iloc[-4], group.iloc[-5]

        # কন্ডিশন ১
        if (A["close"] > B["high"] and
            B["low"] < C["low"] and
            B["high"] < C["high"] and
            C["high"] < D["high"] and
            C["low"] < D["low"]):
            return float(B["low"])

        # কন্ডিশন ২
        if (A["close"] > B["high"] and
            B["high"] < C["high"] and
            B["low"] > C["low"] and
            C["high"] < D["high"] and
            C["low"] < D["low"] and
            D["high"] < E["high"] and
            D["low"] < E["low"]):
            return float(C["low"])
    except Exception as e:
        print(f"Error in find_sl_from_buy: {e}")
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

        # সুরক্ষিত লুপ
        for i in range(min(anchor_idx, len(group)-1), 2, -1):
            if i-2 < 0:
                break
            sb, sa, s = group.iloc[i-2], group.iloc[i-1], group.iloc[i]
            if s["high"] > sa["high"] >= sb["high"]:
                return float(s["high"])
    except Exception as e:
        print(f"Error in find_tp_from_anchor: {e}")
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

    # ডিরেক্টরি তৈরি
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ai_output_dir, exist_ok=True)
    os.makedirs(trand_base_dir, exist_ok=True)

    uptrend_file = os.path.join(output_dir, "uptrand.csv")
    downtrend_file = os.path.join(output_dir, "downtrand.csv")

    # পুরনো ডেটা লোড (সুরক্ষিতভাবে)
    try:
        old_up_df = pd.read_csv(uptrend_file) if os.path.exists(uptrend_file) and os.path.getsize(uptrend_file) > 0 else None
    except Exception as e:
        print(f"⚠️ Error loading old uptrend file: {e}")
        old_up_df = None

# মূল ডেটা লোড
    if not os.path.exists(mongodb_csv):
        print(f"❌ mongodb.csv not found at {mongodb_csv}")
        return
        
    try:
        df = pd.read_csv(mongodb_csv)
        if df.empty:
            print("⚠️ mongodb.csv is empty")
            return
    except Exception as e:
        print(f"❌ Error reading mongodb.csv: {e}")
        return

    # ডেটা প্রিপ্রসেস
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df = df.dropna(subset=['date'])
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
        if os.path.exists(high_file) and os.path.getsize(high_file) > 0:
            try:
                high_df = pd.read_csv(high_file)
                if len(high_df) >= 2:
                    high_df["date"] = pd.to_datetime(high_df["date"], errors='coerce')
                    high_df = high_df.dropna(subset=['date'])
                    
                    if len(high_df) < 2:
                        continue
                        
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
                        if risk <= 0:
                            continue
                            
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
            except Exception as e:
                print(f"Error processing {symbol} uptrend: {e}")
                continue

        # ---------------- DOWNTREND ----------------
        if os.path.exists(low_file) and os.path.getsize(low_file) > 0:
            try:
                low_df = pd.read_csv(low_file)
                if len(low_df) >= 2:
                    low_df["date"] = pd.to_datetime(low_df["date"], errors='coerce')
                    low_df = low_df.dropna(subset=['date'])
                    
                    if len(low_df) < 2:
                        continue
                        
                    p1, p2 = low_df.iloc[0], low_df.iloc[1]

                    if float(p1["price"]) > float(p2["price"]) > latest_close:
                        downtrend_rows.append({
                            "date": latest_date.strftime("%Y-%m-%d"),

"symbol": symbol,
                            "close": round(latest_close, 2),
                            "p1": p1["date"].strftime("%Y-%m-%d") if hasattr(p1["date"], 'strftime') else str(p1["date"]),
                            "p2": p2["date"].strftime("%Y-%m-%d") if hasattr(p2["date"], 'strftime') else str(p2["date"])
                        })
            except Exception as e:
                print(f"Error processing {symbol} downtrend: {e}")
                continue

    # ---------------- SAVE UPTREND ----------------
    try:
        if uptrend_rows:
            new_up = pd.DataFrame(uptrend_rows)
            new_up["date"] = pd.to_datetime(new_up["date"]).dt.strftime("%Y-%m-%d")
            new_up["p1"] = pd.to_datetime(new_up["p1"]).dt.strftime("%Y-%m-%d")
            new_up["p2"] = pd.to_datetime(new_up["p2"]).dt.strftime("%Y-%m-%d")

            final_up, brand_new = merge_and_track_new_symbols(old_up_df, new_up)
            if not final_up.empty:
                final_up = final_up.sort_values(["RRR", "diff"], ascending=[False, True])
                final_up.insert(0, "no", range(1, len(final_up) + 1))
                final_up.to_csv(uptrend_file, index=False)

                if not brand_new.empty:
                    brand_new.to_csv(os.path.join(ai_output_dir, "uptrand.csv"), index=False)
        else:
            # নতুন সিগন্যাল না থাকলে পুরনোটা রাখো বা খালি ফাইল রাখো
            if old_up_df is not None and not old_up_df.empty:
                old_up_df.to_csv(uptrend_file, index=False)
            else:
                pd.DataFrame(columns=["no", "date", "symbol", "buy", "sl", "tp", "position_size", 
                                     "exposure_bdt", "actual_risk_bdt", "diff", "RRR", "p1", "p2"]).to_csv(uptrend_file, index=False)
    except Exception as e:
        print(f"❌ Error saving uptrend: {e}")

    # ---------------- SAVE DOWNTREND ----------------
    try:
        if downtrend_rows:
            down_df = pd.DataFrame(downtrend_rows)
            down_df.insert(0, "no", range(1, len(down_df) + 1))
            down_df.to_csv(downtrend_file, index=False)
        else:
            pd.DataFrame(columns=["no", "date", "symbol", "close", "p1", "p2"]).to_csv(downtrand_file, index=False)
    except Exception as e:
        print(f"❌ Error saving downtrend: {e}")

# --------------------------------------------------
# --------------------------------------------------
if __name__ == "__main__":
    create_uptrend_downtrend_signals()
    print("✅ Signal generation completed")