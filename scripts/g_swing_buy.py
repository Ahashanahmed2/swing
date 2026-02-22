import pandas as pd
import os
import json
import numpy as np

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
input_path = "./csv/mongodb.csv"
output_path = "./csv/swing_buy.csv"


os.makedirs(os.path.dirname(output_path), exist_ok=True)


# ---------------------------------------------------------
# Clear old results
# ---------------------------------------------------------
full_cols = ["no", "date", "symbol", "buy", "SL"]
empty_df = pd.DataFrame(columns=full_cols)
empty_df.to_csv(output_path, index=False)


# ---------------------------------------------------------
# Load & validate
# ---------------------------------------------------------
if not os.path.exists(input_path):
    print("❌ mongodb.csv not found!")
    exit()

df = pd.read_csv(input_path)

required_cols = ["date", "symbol", "close", "high", "low"]
for col in required_cols:
    if col not in df.columns:
        raise Exception(f"Column '{col}' missing in mongodb.csv")

df = df.dropna(subset=["date"])
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

# Group by symbol for efficiency
mongo_groups = df.groupby("symbol", sort=False)

results = []

# ---------------------------------------------------------
# Per-symbol latest pattern check (5-bar logic)
# ---------------------------------------------------------
for symbol, group in mongo_groups:
    group = group.sort_values("date").reset_index(drop=True)
    if len(group) < 5:
        continue

    # Get last 5 bars: A (latest), B, C, D, E (oldest)
    A, B, C, D, E = [group.iloc[-i] for i in range(1, 6)]

    buy = SL = None

    # Logic 1: SL = B["low"]
    if (A["close"] > B["high"] and
        B["low"] < C["low"] and
        B["high"] < C["high"] and
        C["high"] < D["high"] and
        C["low"] < D["low"]):
        buy, SL = A["close"], B["low"]

    # Logic 2: SL = C["low"]
    elif (A["close"] > B["high"] and
          B["high"] < C["high"] and
          B["low"] > C["low"] and
          C["high"] < D["high"] and
          C["low"] < D["low"] and
          D["high"] < E["high"] and
          D["low"] < E["low"]):
        buy, SL = A["close"], C["low"]

    if buy is None or SL is None:
        continue

    # ✅ Filter valid signals (buy > SL)
    if buy <= SL:
        continue

    # ✅ শুধু বেসিক তথ্য সংরক্ষণ (tp, position size ইত্যাদি বাদ)
    results.append({
        "date": A["date"],
        "symbol": symbol,
        "buy": buy,
        "SL": SL
    })

# ---------------------------------------------------------
# Create DataFrame, filter, sort
# ---------------------------------------------------------
if results:
    result_df = pd.DataFrame(results)
    result_df["buy"] = pd.to_numeric(result_df["buy"], errors="coerce")
    result_df["SL"] = pd.to_numeric(result_df["SL"], errors="coerce")

    # Filter valid signals
    result_df = result_df[result_df["buy"] > result_df["SL"]].reset_index(drop=True)

    if len(result_df) > 0:
        # Sort by date (newest first)
        result_df = result_df.sort_values(["date"], ascending=[False]).reset_index(drop=True)
        result_df.insert(0, "no", range(1, len(result_df) + 1))
        result_df["date"] = result_df["date"].dt.strftime("%Y-%m-%d")

        # ✅ Final column order (শুধু বেসিক তথ্য)
        result_df = result_df[[
            "no", "date", "symbol", "buy", "SL"
        ]]
    else:
        result_df = pd.DataFrame(columns=[
            "no", "date", "symbol", "buy", "SL"
        ])
else:
    result_df = pd.DataFrame(columns=[
        "no", "date", "symbol", "buy", "SL"
    ])

# ---------------------------------------------------------
# Load existing data and merge (remove duplicates)
# ---------------------------------------------------------
existing_df = pd.DataFrame()
if os.path.exists(output_path):
    existing_df = pd.read_csv(output_path)
    print(f"📂 Existing file loaded with {len(existing_df)} signals")

if not existing_df.empty and not result_df.empty:
    # Check if existing file has the correct format
    required_cols = ['symbol', 'date', 'buy', 'SL']
    if not all(col in existing_df.columns for col in required_cols):
        # If old format, use new data
        print("🔄 Existing file has old format - creating new file with updated format")
        final_df = result_df.copy()
    else:
        # Get existing symbols
        existing_symbols = set(existing_df['symbol'].unique())
        
        # Filter out symbols that already exist
        new_symbols_df = result_df[~result_df['symbol'].isin(existing_symbols)]
        
        if not new_symbols_df.empty:
            # Adjust 'no' column for new symbols
            new_symbols_df = new_symbols_df.reset_index(drop=True)
            new_symbols_df.insert(0, "no", range(1, len(new_symbols_df) + 1))
            
            # Combine existing and new data
            final_df = pd.concat([existing_df, new_symbols_df], ignore_index=True)
            print(f"➕ Added {len(new_symbols_df)} new symbols to existing {len(existing_df)} signals")
        else:
            final_df = existing_df
            print("⏭️ No new symbols to add")
        
elif not result_df.empty:
    final_df = result_df
    print(f"🆕 Created new file with {len(result_df)} signals")
else:
    final_df = existing_df if not existing_df.empty else pd.DataFrame()
    print("⚠️ No signals found")

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
if not final_df.empty:
    # Ensure correct column order
    column_order = ["no", "date", "symbol", "buy", "SL"]
    
    # Make sure all columns exist
    for col in column_order:
        if col not in final_df.columns:
            if col == "no":
                final_df.insert(0, "no", range(1, len(final_df) + 1))
            else:
                final_df[col] = None
    
    final_df = final_df[column_order]
    final_df.to_csv(output_path, index=False)
    
    print(f"✅ swing_buy.csv saved with {len(final_df)} signals")
    if len(final_df) > 0:
        print(f"   📈 Latest signal: {final_df.iloc[0]['symbol']} - Buy: {final_df.iloc[0]['buy']:.2f}, SL: {final_df.iloc[0]['SL']:.2f}")
    
    # Show newly added symbols if any
    if not existing_df.empty and not result_df.empty and 'new_symbols_df' in locals() and not new_symbols_df.empty:
        new_added = set(new_symbols_df['symbol'].unique())
        if new_added:
            print(f"   🆕 New symbols added: {', '.join(sorted(new_added))}")
else:
    # Save empty DataFrame with correct columns
    empty_df = pd.DataFrame(columns=["no", "date", "symbol", "buy", "SL"])
    empty_df.to_csv(output_path, index=False)
    print("⚠️ No signals found - empty file saved")