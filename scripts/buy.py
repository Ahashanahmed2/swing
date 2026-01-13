import pandas as pd
import os
import json
import numpy as np
from datetime import datetime

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
# ---------------------------------------------------------
buy_csv_path = "./csv/uptrand.csv"
mongodb_path = "./csv/mongodb.csv"
buy_path = "./csv/buy.csv"
output_buy ="./output/ai_signal/buy.csv"
# ---------------------------------------------------------
# Clear old results
# ---------------------------------------------------------
full_cols = ["no", "date", "symbol", "buy", "SL", "tp", "p1_date", "p2_date", 
             "position_size", "exposure_bdt", "actual_risk_bdt", "diff", "RRR"]
pd.DataFrame(columns=full_cols).to_csv(buy_path, index=False)

# ---------------------------------------------------------
# Load and validate files
# ---------------------------------------------------------
if not os.path.exists(buy_csv_path):
    print("❌ uptrand.csv not found!")
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
        raise Exception(f"Column '{col}' missing in uptrand.csv")

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
# 🔴 PATTERN-BASED SL DETECTION (swing_buy.py থেকে)
# ---------------------------------------------------------
def detect_pattern_and_sl(symbol_data):
    """প্যাটার্ন ডিটেক্ট করে buy price, SL এবং SL_source_row রিটার্ন করে"""
    if len(symbol_data) < 5:
        return None, None, None
    
    # Get last 5 bars
    A = symbol_data.iloc[-1]  # Latest bar
    B = symbol_data.iloc[-2]
    C = symbol_data.iloc[-3]
    D = symbol_data.iloc[-4]
    E = symbol_data.iloc[-5]  # Oldest of last 5
    
    buy = SL = SL_source_row = None
    
    # Logic 1: SL = B["low"]
    if (A["close"] > B["high"] and
        B["low"] < C["low"] and
        B["high"] < C["high"] and
        C["high"] < D["high"] and
        C["low"] < D["low"]):
        buy, SL = A["close"], B["low"]
        SL_source_row = B
        print(f"    ✅ Pattern 1: SL = B.low ({B['low']})")
    
    # Logic 2: SL = C["low"]
    elif (A["close"] > B["high"] and
          B["high"] < C["high"] and
          B["low"] > C["low"] and
          C["high"] < D["high"] and
          C["low"] < D["low"] and
          D["high"] < E["high"] and
          D["low"] < E["low"]):
        buy, SL = A["close"], C["low"]
        SL_source_row = C
        print(f"    ✅ Pattern 2: SL = C.low ({C['low']})")
    
    return buy, SL, SL_source_row

# ---------------------------------------------------------
# 🔴 TP DETECTION FROM p2_date (backward scanning)
# ---------------------------------------------------------
def get_tp_from_p2(symbol_data, p2_date, SL_date):
    """p2_date থেকে backward scanning করে TP বের করা"""
    if pd.isna(p2_date):
        return None, None
    
    print(f"    🔍 Looking for p2_date: {p2_date}")
    
    # Find p2_date row
    p2_rows = symbol_data[symbol_data["date"] == p2_date]
    
    if len(p2_rows) == 0:
        # Find closest date to p2_date
        date_diffs = abs(symbol_data["date"] - p2_date)
        if len(date_diffs) > 0:
            min_idx = date_diffs.idxmin()
            min_days = date_diffs.min().days
            
            if min_days <= 7:
                p2_rows = symbol_data.iloc[[min_idx]]
                print(f"    📅 Using closest p2_date ({min_days} days diff): {p2_rows.iloc[0]['date']}")
            else:
                print(f"    ❌ No close match for p2_date (closest: {min_days} days)")
                return None, None
    
    if len(p2_rows) == 0:
        return None, None
    
    p2_idx = p2_rows.index[0]
    actual_p2_date = p2_rows.iloc[0]["date"]
    
    print(f"    📍 p2_date found at index {p2_idx}: {actual_p2_date}")
    
    # Find SL_date index for reference
    SL_idx = None
    if SL_date is not None:
        SL_rows = symbol_data[symbol_data["date"] == SL_date]
        if len(SL_rows) > 0:
            SL_idx = SL_rows.index[0]
    
    # 🔴 BACKWARD SCANNING from p2_idx
    print(f"    🔄 Starting BACKWARD scanning from index {p2_idx}")
    
    tp = None
    tp_date = None
    
    for i in range(p2_idx, 1, -1):  # p2_idx থেকে 2 পর্যন্ত backward
        try:
            s = symbol_data.iloc[i]      # Current bar
            sa = symbol_data.iloc[i - 1] # 1 bar আগে
            sb = symbol_data.iloc[i - 2] # 2 bar আগে
            
            # তারিখের ক্রম চেক (optional)
            if SL_idx is not None and not (sb["date"] < sa["date"] < s["date"] <= actual_p2_date):
                continue
            
            # ✅ TP প্যাটার্ন কন্ডিশন: s["high"] > sa["high"] and sa["high"] >= sb["high"]
            if (s["high"] > sa["high"]) and (sa["high"] >= sb["high"]):
                tp = s["high"]
                tp_date = s["date"]
                print(f"    ✅ TP found at index {i}: {tp} on {tp_date}")
                print(f"       Pattern: {sb['date']}({sb['high']}) → {sa['date']}({sa['high']}) → {s['date']}({s['high']})")
                return tp, tp_date
                
        except IndexError:
            break
    
    # যদি প্যাটার্ন না মেলে, p2_date এর high কে TP ধরুন
    if tp is None:
        tp = p2_rows.iloc[0]["high"]
        tp_date = actual_p2_date
        print(f"    ℹ️ No pattern found, using p2_date high as TP: {tp}")
    
    return tp, tp_date

# ---------------------------------------------------------
# Main processing
# ---------------------------------------------------------
results = []
processed = 0
skipped = 0

print(f"\n{'='*60}")
print("🚀 PROCESSING SIGNALS")
print(f"{'='*60}")

for idx, buy_row in buy_df.iterrows():
    symbol = buy_row["symbol"]
    buy_date = buy_row["date"]
    p1_date = buy_row["p1_date"]
    p2_date = buy_row["p2_date"]
    
    print(f"\n[{idx+1}/{len(buy_df)}] Processing {symbol}")
    print(f"  📅 Date: {buy_date}, p1_date: {p1_date}, p2_date: {p2_date}")
    
    # Basic validation
    if pd.isna(p1_date) or pd.isna(p2_date):
        print(f"  ❌ Missing dates - SKIPPING")
        skipped += 1
        continue
    
    # Get symbol data from mongodb
    if symbol not in mongo_groups.groups:
        print(f"  ❌ Symbol '{symbol}' not found in mongodb - SKIPPING")
        skipped += 1
        continue
    
    symbol_data = mongo_groups.get_group(symbol).sort_values("date").reset_index(drop=True)
    print(f"  📊 Found {len(symbol_data)} rows in mongodb")
    
    # 🔴 STEP 1: PATTERN-BASED SL DETECTION
    print(f"  🔍 Detecting pattern for SL...")
    
    # Filter data up to buy_date
    data_upto_buy = symbol_data[symbol_data["date"] <= buy_date]
    
    if len(data_upto_buy) < 5:
        print(f"  ❌ Not enough data for pattern detection ({len(data_upto_buy)} < 5) - SKIPPING")
        skipped += 1
        continue
    
    # Get buy price, SL and SL source row from pattern
    buy_price, SL, SL_source_row = detect_pattern_and_sl(data_upto_buy)
    
    if buy_price is None or SL is None:
        print(f"  ❌ No valid pattern found - SKIPPING")
        skipped += 1
        continue
    
    print(f"  ✅ Pattern-based: Buy={buy_price:.2f}, SL={SL:.2f}")
    
    # Get actual SL date from SL_source_row
    SL_date = SL_source_row["date"] if SL_source_row is not None else None
    
    # 🔴 STEP 2: TP FROM p2_date (BACKWARD SCANNING)
    tp, actual_p2_date = get_tp_from_p2(symbol_data, p2_date, SL_date)
    
    if tp is None:
        print(f"  ❌ Could not determine TP - SKIPPING")
        skipped += 1
        continue
    
    print(f"  📈 TP from p2: {tp:.2f}")
    
    # 🔴 STEP 3: VALIDATION
    if buy_price <= SL:
        print(f"  ❌ Invalid: Buy ({buy_price:.2f}) <= SL ({SL:.2f}) - SKIPPING")
        skipped += 1
        continue
    
    if tp <= buy_price:
        print(f"  ❌ Invalid: TP ({tp:.2f}) <= Buy ({buy_price:.2f}) - SKIPPING")
        skipped += 1
        continue
    
    # 🔴 STEP 4: POSITION SIZING
    risk_per_trade = TOTAL_CAPITAL * RISK_PERCENT
    risk_per_share = buy_price - SL
    
    if risk_per_share <= 0:
        print(f"  ❌ Risk per share <= 0 - SKIPPING")
        skipped += 1
        continue
    
    position_size = int(risk_per_trade / risk_per_share)
    position_size = max(1, position_size)
    
    exposure_bdt = position_size * buy_price
    actual_risk_bdt = position_size * risk_per_share
    
    # 🔴 STEP 5: RRR CALCULATION
    RRR = (tp - buy_price) / risk_per_share
    
    print(f"  📊 Risk/Share: {risk_per_share:.2f}, RRR: {RRR:.2f}")
    print(f"  💰 Position: {position_size:,} shares, Risk: BDT {actual_risk_bdt:,.0f}")
    
    if RRR <= 0:
        print(f"  ❌ RRR <= 0 - SKIPPING")
        skipped += 1
        continue
    
    # 🔴 STEP 6: STORE RESULTS
    results.append({
        "date": buy_date,
        "symbol": symbol,
        "buy": round(buy_price, 2),
        "SL": round(SL, 2),
        "tp": round(tp, 2),
        "p1_date": SL_date,  # p1_date = SL এর তারিখ
        "p2_date": actual_p2_date,  # p2_date = TP এর তারিখ
        "position_size": position_size,
        "exposure_bdt": round(exposure_bdt, 2),
        "actual_risk_bdt": round(actual_risk_bdt, 2),
        "diff": round(risk_per_share, 4),
        "RRR": round(RRR, 2)
    })
    
    processed += 1
    print(f"  ✅ SUCCESS - Added to results")

print(f"\n{'='*60}")
print("🎯 PROCESSING COMPLETE")
print(f"{'='*60}")
print(f"✅ Successfully processed: {processed} signals")
print(f"❌ Skipped: {skipped} symbols")

if processed + skipped > 0:
    print(f"📈 Success rate: {processed/(processed+skipped)*100:.1f}%")

# ---------------------------------------------------------
# Create final DataFrame
# ---------------------------------------------------------
if results:
    result_df = pd.DataFrame(results)
    
    # Convert to numeric
    numeric_cols = ["buy", "SL", "tp", "exposure_bdt", "actual_risk_bdt"]
    for col in numeric_cols:
        result_df[col] = pd.to_numeric(result_df[col], errors="coerce")
    
    result_df["position_size"] = result_df["position_size"].astype(int)
    
    # Convert dates
    result_df["p1_date"] = pd.to_datetime(result_df["p1_date"], errors="coerce")
    result_df["p2_date"] = pd.to_datetime(result_df["p2_date"], errors="coerce")
    
    # Already calculated diff & RRR
    
    # Filter valid signals
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
result_df.to_csv(buy_path, index=False)
result_df.to_csv(output_buy, index=False)
print(f"\n{'='*60}")
print(f"💾 SAVED TO: {buy_path}")
print(f"{'='*60}")

if len(result_df) > 0:
    print(f"📊 FINAL RESULTS: {len(result_df)} valid signals")
    print(f"   📈 Top RRR: {result_df['RRR'].max():.2f} | Avg RRR: {result_df['RRR'].mean():.2f}")
    print(f"   📉 Min diff: {result_df['diff'].min():.4f}")
    print(f"   💰 Avg position: {result_df['position_size'].mean():.0f} shares")
    print(f"   🎯 Avg actual risk: {result_df['actual_risk_bdt'].mean():,.0f} BDT")
    print(f"   📅 p1_date range: {result_df['p1_date'].min()} to {result_df['p1_date'].max()}")
    print(f"   📅 p2_date range: {result_df['p2_date'].min()} to {result_df['p2_date'].max()}")
    
    # Show top 5 signals
    print(f"\n🏆 TOP 5 SIGNALS:")
    print(result_df[["symbol", "buy", "SL", "tp", "RRR", "position_size"]].head().to_string(index=False))
else:
    print("   ⚠️ No valid signals found")