import pandas as pd
import os
import numpy as np

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
short_path = "./csv/short_buy.csv"
gape_path = "./csv/gape_buy.csv"
rsi_path = "./csv/rsi_30_buy.csv"

mongodb_path = "./csv/mongodb.csv"
trade_stock_path = "./csv/trade_stock.csv"
profit_loss_path = "./output/ai_signal/profit-loss.csv"

os.makedirs("./output/ai_signal", exist_ok=True)

# ---------------------------------------------------------
# Helper: Load BUY data
# ---------------------------------------------------------
def load_file(path, reference, buy_col):
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        return pd.DataFrame(columns=["date", "symbol", "buy", "SL", "Reference"])

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")
        return pd.DataFrame(columns=["date", "symbol", "buy", "SL", "Reference"])

    required_cols = {"symbol", "date", buy_col, "SL"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"⚠️ Missing columns in {path}: {missing}")
        return pd.DataFrame(columns=["date", "symbol", "buy", "SL", "Reference"])

    # Clean and convert
    df = df.dropna(subset=["symbol", "date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    return pd.DataFrame({
        "date": df["date"].dt.date,  # ✅ only date part for signal
        "symbol": df["symbol"].astype(str).str.strip().str.upper(),
        "buy": pd.to_numeric(df[buy_col], errors="coerce"),
        "SL": pd.to_numeric(df["SL"], errors="coerce"),
        "Reference": reference
    }).dropna(subset=["buy", "SL"])


# ---------------------------------------------------------
# Create trade_stock.csv
# ---------------------------------------------------------
short_df = load_file(short_path, "short", "last_row_close")
gape_df  = load_file(gape_path,  "gape",  "last_row_close")
rsi_df   = load_file(rsi_path,   "rsi",   "close")

trade_df = pd.concat([short_df, gape_df, rsi_df], ignore_index=True)
print(f"✅ Loaded {len(trade_df)} trade signals.")

if trade_df.empty:
    print("🛑 No valid trade signals. Exiting.")
    exit()

# Add unique ID to track individual trades
trade_df = trade_df.reset_index(drop=True)
trade_df["trade_id"] = trade_df.index

# Save initial trade_stock
trade_df_with_no = trade_df.copy()
trade_df_with_no.insert(0, "no", range(1, len(trade_df_with_no) + 1))
trade_df_with_no.drop(columns=["trade_id"], inplace=True)
trade_df_with_no.to_csv(trade_stock_path, index=False)


# ---------------------------------------------------------
# Load mongodb database
# ---------------------------------------------------------
if not os.path.exists(mongodb_path):
    raise FileNotFoundError(f"❌ mongodb.csv not found at {mongodb_path}")

try:
    mongodb = pd.read_csv(mongodb_path)
    if mongodb.empty:
        raise ValueError("mongodb.csv is empty!")
except Exception as e:
    raise Exception(f"❌ Failed to load mongodb.csv: {e}")

# Ensure required columns
if not {"symbol", "date", "close"}.issubset(mongodb.columns):
    raise ValueError(f"mongodb.csv missing required columns: symbol, date, close")

# Clean mongodb
mongodb["symbol"] = mongodb["symbol"].astype(str).str.strip().str.upper()
mongodb["date"] = pd.to_datetime(mongodb["date"], errors="coerce")
mongodb = mongodb.dropna(subset=["symbol", "date", "close"])
mongodb["close"] = pd.to_numeric(mongodb["close"], errors="coerce")
mongodb = mongodb.dropna(subset=["close"])

# Sort for reliable indexing
mongodb = mongodb.sort_values(["symbol", "date"]).reset_index(drop=True)
print(f"✅ Loaded {len(mongodb)} rows from mongodb.csv")


# ---------------------------------------------------------
# Profit–Loss Calculator
# ---------------------------------------------------------
results = []
remove_trade_ids = []  # ✅ Track by trade_id, not symbol

for _, row in trade_df.iterrows():
    symbol = row["symbol"]
    buy_date = row["date"]        # already a datetime.date
    buy = float(row["buy"])
    SL_percent = float(row["SL"])
    trade_id = row["trade_id"]
    
    # Default fallbacks
    if SL_percent <= 0 or SL_percent > 50:
        SL_percent = 5.0  # safe default

    SL_value = buy * (1 - SL_percent / 100)
    profit_target = buy * 1.10  # 10% target

    # Filter data for symbol (preserve full datetime for sequencing)
    df_sym = mongodb[mongodb["symbol"] == symbol].copy()
    if df_sym.empty:
        print(f"⚠️ No price data for {symbol}")
        continue

    # Match on DATE only (ignore time)
    df_sym["date_only"] = df_sym["date"].dt.date
    buy_rows = df_sym[df_sym["date_only"] == buy_date]

    if buy_rows.empty:
        print(f"⚠️ No buy date match for {symbol} on {buy_date}")
        continue

    # Use the FIRST matching row (e.g., earliest timestamp on that day)
    buy_idx = buy_rows.index[0]
    future_rows = df_sym.loc[df_sym.index > buy_idx].sort_values("date")  # ensure order

    if future_rows.empty:
        print(f"⚠️ No future data for {symbol} after {buy_date}")
        continue

    hit = False
    for i, r in future_rows.iterrows():
        diff = i - buy_idx  # number of bars after buy
        close = r["close"]
        cur_date = r["date"].date()

        # 🔴 STOP-LOSS: close < SL_value
        if close < SL_value:
            loss_percent = ((buy - close) / buy) * 100
            results.append([
                None, symbol, buy_date, buy,
                cur_date, close,
                round(loss_percent, 2), np.nan, int(diff)
            ])
            remove_trade_ids.append(trade_id)
            hit = True
            break

        # 🟢 PROFIT: close >= 10% target
        if close >= profit_target:
            profit_percent = ((close - buy) / buy) * 100
            results.append([
                None, symbol, buy_date, buy,
                cur_date, close,
                np.nan, round(profit_percent, 2), int(diff)
            ])
            remove_trade_ids.append(trade_id)
            hit = True
            break

    if not hit:
        print(f"⏳ {symbol} ({buy_date}): No exit triggered (SL={SL_value:.2f}, Target={profit_target:.2f})")


# ---------------------------------------------------------
# Save profit-loss.csv
# ---------------------------------------------------------
if results:
    out = pd.DataFrame(results, columns=[
        "no", "symbol", "date", "buy",
        "sell_date", "sell",
        "loss", "profit", "diff"
    ])
    out["no"] = range(1, len(out) + 1)
    # Ensure numeric types & clean NaN
    out["loss"] = pd.to_numeric(out["loss"], errors="coerce")
    out["profit"] = pd.to_numeric(out["profit"], errors="coerce")
    out.to_csv(profit_loss_path, index=False)
    print(f"✅ Saved {len(out)} exit records to {profit_loss_path}")
else:
    print("⚠️ No exits triggered. profit-loss.csv not generated.")


# ---------------------------------------------------------
# Update trade_stock.csv (remove only HIT trades)
# ---------------------------------------------------------
clean_trade = trade_df[~trade_df["trade_id"].isin(remove_trade_ids)].copy()
print(f"♻️ Removed {len(remove_trade_ids)} exited trades. {len(clean_trade)} remain.")

clean_trade = clean_trade.drop(columns=["trade_id"])
if not clean_trade.empty:
    clean_trade.insert(0, "no", range(1, len(clean_trade) + 1))
clean_trade.to_csv(trade_stock_path, index=False)

print("🎉 Profit–Loss calculation completed & trade_stock.csv updated successfully!")