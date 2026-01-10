import pandas as pd
import os
import json
import numpy as np

# ---------------------------------------------------------
# 🔧 Load config.json
# ---------------------------------------------------------
CONFIG_PATH = "./config.json"
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    TOTAL_CAPITAL = float(config.get("total_capital", 500000))
    RISK_PERCENT = float(config.get("risk_percent", 0.01))
    print(f"✅ Config: capital={TOTAL_CAPITAL:,.0f} BDT, risk={RISK_PERCENT*100:.1f}% per trade")
except Exception as e:
    print(f"⚠️ Config load failed → using defaults: 5,00,000 BDT, 1% risk")
    TOTAL_CAPITAL = 500000
    RISK_PERCENT = 0.01

# ---------------------------------------------------------
# Paths
#r ---------------------------------------------------------
buy_csv_path = "./csv/uptrand.csv"
mongodb_path = "./csv/mongodb.csv"
output_path = "./output/ai_signal/uptrand.csv"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ---------------------------------------------------------
# Clear old results
# ---------------------------------------------------------
# p1_date এবং p2_date কলাম যোগ করা হয়েছে
full_cols = ["no", "date", "symbol", "buy", "SL", "tp", "p1_date", "p2_date", 
             "position_size", "exposure_bdt", "actual_risk_bdt", "diff", "RRR"]
pd.DataFrame(columns=full_cols).to_csv(output_path, index=False)

# ---------------------------------------------------------
# Load and validate files
# ---------------------------------------------------------
if not os.path.exists(buy_csv_path):
    print("❌ buy.csv not found!")
    exit()

if not os.path.exists(mongodb_path):
    print("❌ mongodb.csv not found!")
    exit()

buy_df = pd.read_csv(buy_csv_path)
mongo_df = pd.read_csv(mongodb_path)

# Check required columns
required_buy_cols = ["date", "symbol", "close", "p1_date", "p2_date"]
for col in required_buy_cols:
    if col not in buy_df.columns:
        raise Exception(f"Column '{col}' missing in buy.csv")

required_mongo_cols = ["date", "symbol", "close", "high", "low"]
for col in required_mongo_cols:
    if col not in mongo_df.columns:
        raise Exception(f"Column '{col}' missing in mongodb.csv")

# Preprocess dates
buy_df["date"] = pd.to_datetime(buy_df["date"], errors="coerce")
buy_df["p1_date"] = pd.to_datetime(buy_df["p1_date"], errors="coerce")
buy_df["p2_date"] = pd.to_datetime(buy_df["p2_date"], errors="coerce")
buy_df = buy_df.dropna(subset=["date", "symbol", "close"])

mongo_df["date"] = pd.to_datetime(mongo_df["date"], errors="coerce")
mongo_df = mongo_df.dropna(subset=["date", "symbol"])
mongo_df = mongo_df.sort_values(["symbol", "date"]).reset_index(drop=True)

# Group mongodb data by symbol
mongo_groups = mongo_df.groupby("symbol", sort=False)

# ---------------------------------------------------------
# 🔴 IMPORTANT: Pattern Detection Logic (Same as swing_buy.py)
# ---------------------------------------------------------
def detect_patterns(symbol_data):
    """মূল swing_buy স্ক্রিপ্টের মতো ৫-বার প্যাটার্ন ডিটেকশন"""
    if len(symbol_data) < 5:
        return None, None, None, None, None

    # শুধুমাত্র শেষের ৫টা বার চেক করছি (আপনার মূল লজিক অনুযায়ী)
    A, B, C, D, E = [symbol_data.iloc[-i] for i in range(1, 6)]

    buy = SL = SL_source_row = p1_date = p2_date = None

    # 🔥 লজিক 1: SL = B["low"]
    if (A["close"] > B["high"] and
        B["low"] < C["low"] and
        B["high"] < C["high"] and
        C["high"] < D["high"] and
        C["low"] < D["low"]):
        buy, SL = A["close"], B["low"]
        SL_source_row = B
        p1_date = B["date"]  # p1_date = B এর তারিখ

    # 🔥 লজিক 2: SL = C["low"]
    elif (A["close"] > B["high"] and
          B["high"] < C["high"] and
          B["low"] > C["low"] and
          C["high"] < D["high"] and
          C["low"] < D["low"] and
          D["high"] < E["high"] and
          D["low"] < E["low"]):
        buy, SL = A["close"], C["low"]
        SL_source_row = C
        p1_date = C["date"]  # p1_date = C এর তারিখ

    # যদি কোনো প্যাটার্ন পাওয়া যায়, TP বের করুন
    if buy is not None and SL is not None and SL_source_row is not None:
        # TP বের করার লজিক (আপনার স্ক্রিপ্টের মতো)
        tp = None
        try:
            sl_idx = symbol_data[symbol_data["date"] == SL_source_row["date"]].index[0]
        except IndexError:
            sl_idx = (abs(symbol_data["date"] - SL_source_row["date"])).idxmin()

        # Backward scanning for TP
        tp_date = None
        for i in range(sl_idx - 1, 1, -1):
            try:
                sb = symbol_data.iloc[i - 2]
                sa = symbol_data.iloc[i - 1]
                s = symbol_data.iloc[i]
            except IndexError:
                break

            if not (sb["date"] < sa["date"] < s["date"] < SL_source_row["date"]):
                continue

            if (s["high"] > sa["high"]) and (sa["high"] >= sb["high"]):
                tp = s["high"]
                tp_date = s["date"]  # p2_date = TP এর তারিখ
                break

        return buy, SL, tp, p1_date, tp_date

    return None, None, None, None, None

# ---------------------------------------------------------
# Main processing
# ---------------------------------------------------------
results = []

# প্রতিটি buy.csv এর row এর জন্য
for idx, buy_row in buy_df.iterrows():
    symbol = buy_row["symbol"]
    buy_date = buy_row["date"]

    # mongodb.csv থেকে ঐ সিম্বলের সম্পূর্ণ ডেটা নিন
    if symbol not in mongo_groups.groups:
        continue

    symbol_data = mongo_groups.get_group(symbol).sort_values("date").reset_index(drop=True)

    # 🔴 OPTION 1: buy.csv এর close কে buy ধরে এবং p1_date, p2_date থেকে SL, TP
    buy_price = buy_row["close"]
    buy_p1_date = buy_row["p1_date"]  # buy.csv থেকে p1_date
    buy_p2_date = buy_row["p2_date"]  # buy.csv থেকে p2_date

    # p1_date থেকে SL বের করা
    SL = None
    SL_source_row = None
    p1_date = buy_row["p1_date"]

    p1_row = symbol_data[symbol_data["date"] == p1_date]
    if len(p1_row) == 0:
        date_diffs = abs(symbol_data["date"] - p1_date)
        if len(date_diffs) > 0:
            p1_idx = date_diffs.idxmin()
            p1_row = symbol_data.iloc[[p1_idx]]

    if len(p1_row) > 0:
        SL = p1_row.iloc[0]["low"]
        SL_source_row = p1_row.iloc[0]
        p1_date_value = p1_row.iloc[0]["date"]  # প্রকৃত p1_date
    else:
        p1_date_value = p1_date

    # 🔴 OPTION 2: প্যাটার্ন ডিটেকশন দিয়ে buy, SL, TP বের করুন
    pattern_buy, pattern_SL, pattern_tp, pattern_p1_date, pattern_p2_date = detect_patterns(symbol_data)

    # 🔴 এখন decision নিন কোনটি ব্যবহার করবেন:
    # Option A: শুধু buy.csv এর ডেটা ব্যবহার করুন
    use_buy_price = buy_price
    use_SL = SL
    use_p1_date = p1_date_value  # p1_date সংরক্ষণ
    use_p2_date = buy_p2_date    # p2_date সংরক্ষণ
    pattern_used = False

    # Option B: প্যাটার্ন ডিটেকশন ব্যবহার করুন (যদি পাওয়া যায়)
    if pattern_buy is not None and pattern_SL is not None:
        use_buy_price = pattern_buy
        use_SL = pattern_SL
        pattern_used = True
        # প্যাটার্ন থেকে p1_date এবং p2_date আপডেট করুন
        if pattern_p1_date is not None:
            use_p1_date = pattern_p1_date
        if pattern_p2_date is not None:
            use_p2_date = pattern_p2_date

    if use_SL is None:
        continue

    # TP বের করা
    tp = None
    final_p2_date = use_p2_date  # p2_date সংরক্ষণ

    # যদি প্যাটার্ন থেকে TP পাওয়া যায়, সেটা ব্যবহার করুন
    if pattern_tp is not None and pattern_used:
        tp = pattern_tp
        if pattern_p2_date is not None:
            final_p2_date = pattern_p2_date
    else:
        # নাহলে p2_date থেকে TP বের করুন
        p2_date = buy_row["p2_date"]
        p2_row = symbol_data[symbol_data["date"] == p2_date]

        if len(p2_row) == 0:
            date_diffs = abs(symbol_data["date"] - p2_date)
            if len(date_diffs) > 0:
                p2_idx = date_diffs.idxmin()
                p2_row = symbol_data.iloc[[p2_idx]]

        if len(p2_row) > 0:
            p2_idx = p2_row.index[0]

            # Backward scanning for TP (same as your script)
            tp_date_found = None
            for i in range(p2_idx - 1, 1, -1):
                try:
                    sb = symbol_data.iloc[i - 2]
                    sa = symbol_data.iloc[i - 1]
                    s = symbol_data.iloc[i]
                except IndexError:
                    break

                if not (sb["date"] < sa["date"] < s["date"] < p2_row.iloc[0]["date"]):
                    continue

                if (s["high"] > sa["high"]) and (sa["high"] >= sb["high"]):
                    tp = s["high"]
                    tp_date_found = s["date"]
                    break

            # যদি TP পাওয়া যায়, p2_date আপডেট করুন
            if tp_date_found is not None:
                final_p2_date = tp_date_found
            # যদি TP না পাওয়া যায়, p2_date এর high নিন
            elif tp is None and len(p2_row) > 0:
                tp = p2_row.iloc[0]["high"]
                final_p2_date = p2_row.iloc[0]["date"]

    if tp is None:
        continue

    # ✅ DSE-COMPLIANT POSITION SIZING
    risk_per_trade = TOTAL_CAPITAL * RISK_PERCENT
    risk_per_share = use_buy_price - use_SL

    if risk_per_share <= 0:
        continue

    position_size = int(risk_per_trade / risk_per_share)
    position_size = max(1, position_size)

    exposure_bdt = position_size * use_buy_price
    actual_risk_bdt = position_size * risk_per_share

    # Append result with p1_date and p2_date
    results.append({
        "date": buy_date,
        "symbol": symbol,
        "buy": use_buy_price,
        "SL": use_SL,
        "tp": tp,
        "p1_date": use_p1_date,
        "p2_date": final_p2_date,
        "position_size": position_size,
        "exposure_bdt": round(exposure_bdt, 2),
        "actual_risk_bdt": round(actual_risk_bdt, 2)
    })

# ---------------------------------------------------------
# Create final DataFrame
# ---------------------------------------------------------
if results:
    result_df = pd.DataFrame(results)

    # Numeric conversion
    numeric_cols = ["buy", "SL", "tp", "exposure_bdt", "actual_risk_bdt"]
    for col in numeric_cols:
        result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

    # Date conversion for p1_date and p2_date
    result_df["p1_date"] = pd.to_datetime(result_df["p1_date"], errors="coerce")
    result_df["p2_date"] = pd.to_datetime(result_df["p2_date"], errors="coerce")

    result_df["position_size"] = result_df["position_size"].astype(int)

    # Compute diff & RRR
    result_df["diff"] = (result_df["buy"] - result_df["SL"]).round(4)
    result_df["RRR"] = ((result_df["tp"] - result_df["buy"]) / 
                       (result_df["buy"] - result_df["SL"])).round(2)

    # ✅ Filter valid signals
    result_df = result_df[
        (result_df["buy"] > result_df["SL"]) &
        (result_df["tp"] > result_df["buy"]) &
        (result_df["RRR"] > 0)
    ].reset_index(drop=True)

    if len(result_df) > 0:
        # Sort by RRR and diff
        result_df = result_df.sort_values(["RRR", "diff"], 
                                         ascending=[False, True]).reset_index(drop=True)
        result_df.insert(0, "no", range(1, len(result_df) + 1))

        # Format dates for output
        result_df["date"] = pd.to_datetime(result_df["date"]).dt.strftime("%Y-%m-%d")
        result_df["p1_date"] = pd.to_datetime(result_df["p1_date"]).dt.strftime("%Y-%m-%d")
        result_df["p2_date"] = pd.to_datetime(result_df["p2_date"]).dt.strftime("%Y-%m-%d")

        # Final column order
        result_df = result_df[full_cols]
    else:
        result_df = pd.DataFrame(columns=full_cols)
else:
    result_df = pd.DataFrame(columns=full_cols)

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------
result_df.to_csv(output_path, index=False)

print(f"✅ ai_signal/buy.csv updated with {len(result_df)} signals:")
if len(result_df) > 0:
    print(f"   📈 Top RRR: {result_df['RRR'].max():.2f} | Avg RRR: {result_df['RRR'].mean():.2f}")
    print(f"   📉 Min diff: {result_df['diff'].min():.4f}")
    print(f"   💰 Avg position: {result_df['position_size'].mean():.0f} shares")
    print(f"   🎯 Avg actual risk: {result_df['actual_risk_bdt'].mean():,.0f} BDT")
    print(f"   📅 p1_date range: {result_df['p1_date'].min()} to {result_df['p1_date'].max()}")
    print(f"   📅 p2_date range: {result_df['p2_date'].min()} to {result_df['p2_date'].max()}")
else:
    print("   ⚠️ No valid signals found")