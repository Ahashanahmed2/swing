import pandas as pd
import os
import numpy as np

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
short_path = "./csv/short_buy.csv"
gape_path = "./csv/gape_buy.csv"
rsi_path = "./csv/rsi_30_buy.csv"
swing_path = "./csv/swing_buy.csv"

mongodb_path = "./csv/mongodb.csv"
trade_stock_path = "./csv/trade_stock.csv"
profit_loss_path = "./output/ai_signal/profit-loss.csv"

os.makedirs("./output/ai_signal", exist_ok=True)


# ---------------------------------------------------------
# Helper: Load BUY data (now with tp & RRR)
# ---------------------------------------------------------
def load_file(path, reference, buy_col, tp_col="tp", rrr_col="RRR"):
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        return pd.DataFrame(columns=["date", "symbol", "buy", "SL", "tp", "RRR", "Reference"])

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")
        return pd.DataFrame(columns=["date", "symbol", "buy", "SL", "tp", "RRR", "Reference"])

    cols_needed = {"symbol", "date", buy_col, "SL"}
    if tp_col in df.columns:
        cols_needed.add(tp_col)
    if rrr_col in df.columns:
        cols_needed.add(rrr_col)

    missing = cols_needed - set(df.columns)
    if missing:
        print(f"⚠️ Missing columns in {path}: {missing}")

    df = df.dropna(subset=["symbol", "date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Build result
    result = {
        "date": df["date"].dt.date,
        "symbol": df["symbol"].astype(str).str.strip().str.upper(),
        "buy": pd.to_numeric(df[buy_col], errors="coerce"),
        "SL": pd.to_numeric(df["SL"], errors="coerce"),
        "Reference": reference
    }

    # Add tp & RRR if available
    if tp_col in df.columns:
        result["tp"] = pd.to_numeric(df[tp_col], errors="coerce")
    else:
        result["tp"] = np.nan

    if rrr_col in df.columns:
        result["RRR"] = pd.to_numeric(df[rrr_col], errors="coerce")
    else:
        result["RRR"] = np.nan

    out_df = pd.DataFrame(result).dropna(subset=["buy", "SL"])

    # Fill missing tp from RRR (if RRR known but tp missing): tp = buy + RRR * (buy - SL)
    mask = out_df["tp"].isna() & out_df["RRR"].notna()
    if mask.any():
        out_df.loc[mask, "tp"] = out_df.loc[mask, "buy"] + out_df.loc[mask, "RRR"] * (out_df.loc[mask, "buy"] - out_df.loc[mask, "SL"])

    return out_df


# ---------------------------------------------------------
# ✅ DSEX-Optimized k Selector (internal only)
# ---------------------------------------------------------
def get_dsex_k(market_cap_million, atr_pct):
    market_cap_cr = market_cap_million / 10 if pd.notna(market_cap_million) else 0
    if market_cap_cr >= 5000:      # Large
        return 1.2 if atr_pct < 3.0 else (1.4 if atr_pct <= 5.0 else 1.6)
    elif market_cap_cr >= 500:     # Mid
        return 1.5
    else:                          # Small
        return 1.6 if atr_pct < 4.0 else 1.8


# ---------------------------------------------------------
# Load signals (now with tp & RRR)
# ---------------------------------------------------------
short_df = load_file(short_path, "short", "buy")          # short_buy.csv uses 'buy'
gape_df = load_file(gape_path, "gape", "buy")             # gape_buy.csv uses 'buy'
rsi_df = load_file(rsi_path, "rsi", "buy")                # rsi_30_buy.csv now uses 'buy'
swing_df = load_file(swing_path, "swing", "buy")          # swing_buy.csv uses 'buy'

trade_df = pd.concat([short_df, gape_df, rsi_df, swing_df], ignore_index=True)
print(f"✅ Loaded {len(trade_df)} trade signals (with tp & RRR where available).")

if trade_df.empty:
    print("🛑 No valid trade signals. Exiting.")
    exit()

trade_df = trade_df.reset_index(drop=True)
trade_df["trade_id"] = trade_df.index

# Save trade_stock.csv (with tp & RRR)
trade_df_with_no = trade_df.copy()
trade_df_with_no.insert(0, "no", range(1, len(trade_df_with_no) + 1))
cols_order = ["no", "date", "symbol", "buy", "SL", "tp", "RRR", "Reference"]
trade_df_with_no = trade_df_with_no.reindex(columns=cols_order)
trade_df_with_no.to_csv(trade_stock_path, index=False)


# ---------------------------------------------------------
# Load mongodb.csv
# ---------------------------------------------------------
if not os.path.exists(mongodb_path):
    raise FileNotFoundError(f"❌ mongodb.csv not found at {mongodb_path}")

try:
    mongodb = pd.read_csv(mongodb_path)
    if mongodb.empty:
        raise ValueError("mongodb.csv is empty!")
except Exception as e:
    raise Exception(f"❌ Failed to load mongodb.csv: {e}")

essential_cols = {"symbol", "date", "close", "atr", "marketCap"}
missing = essential_cols - set(mongodb.columns)
if missing:
    raise ValueError(f"mongodb.csv missing: {missing}")

mongodb["symbol"] = mongodb["symbol"].astype(str).str.strip().str.upper()
mongodb["date"] = pd.to_datetime(mongodb["date"], errors="coerce")
for col in ["close", "atr", "marketCap"]:
    mongodb[col] = pd.to_numeric(mongodb[col], errors="coerce")
mongodb = mongodb.dropna(subset=["symbol", "date", "close", "atr"])
mongodb = mongodb.sort_values(["symbol", "date"]).reset_index(drop=True)
print(f"✅ Loaded {len(mongodb)} rows (marketCap in million BDT)")


# ---------------------------------------------------------
# Profit–Loss Calculator (using tp, not 10%!)
# ---------------------------------------------------------
results = []
remove_trade_ids = []

for _, row in trade_df.iterrows():
    symbol = row["symbol"]
    buy_date = row["date"]
    buy = float(row["buy"])
    SL_manual = float(row["SL"])
    tp_val = float(row["tp"]) if pd.notna(row["tp"]) else buy * 1.10  # fallback to 10%
    rrr_val = float(row["RRR"]) if pd.notna(row["RRR"]) else np.nan
    trade_id = row["trade_id"]
    ref = row["Reference"]

    # Compute SL_value (from manual SL, not %)
    SL_value = SL_manual  # ✅ SL is absolute price, not %
    buy_sl_diff = buy - SL_value
    sl_pct = ((buy - SL_value) / buy) * 100 if buy > 0 else np.nan

    # Get ATR & marketCap
    df_sym = mongodb[mongodb["symbol"] == symbol].copy()
    if df_sym.empty:
        continue

    df_sym["date_only"] = df_sym["date"].dt.date
    buy_rows = df_sym[df_sym["date_only"] == buy_date]
    if buy_rows.empty:
        continue

    buy_row = buy_rows.iloc[0]
    atr = buy_row["atr"]
    market_cap_million = buy_row.get("marketCap", np.nan)

    # ATR%
    atr_pct = (atr / buy) * 100 if atr > 0 else 3.0
    k = get_dsex_k(market_cap_million, atr_pct)
    atr_sl_pct = (k * atr / buy) * 100

    # Future data
    buy_idx = buy_rows.index[0]
    future_rows = df_sym.loc[df_sym.index > buy_idx].sort_values("date")
    if future_rows.empty:
        continue

    hit = False
    for _, r in future_rows.iterrows():
        close = r["close"]
        cur_date = r["date"].date()
        diff_days = (cur_date - buy_date).days

        # SL hit
        if close < SL_value:
            loss_pct = ((buy - close) / buy) * 100
            results.append([
                None, symbol, buy_date, buy, SL_value,
                cur_date, close,
                round(loss_pct, 2), np.nan,
                diff_days, ref,
                round(buy_sl_diff, 4), round(sl_pct, 2),
                round(atr_sl_pct, 2),
                round(tp_val, 4),  # ✅ tp
                round(rrr_val, 2) if pd.notna(rrr_val) else np.nan  # ✅ RRR
            ])
            remove_trade_ids.append(trade_id)
            hit = True
            break

        # Profit hit (✅ now using tp!)
        if close >= tp_val:
            profit_pct = ((close - buy) / buy) * 100
            results.append([
                None, symbol, buy_date, buy, SL_value,
                cur_date, close,
                np.nan, round(profit_pct, 2),
                diff_days, ref,
                round(buy_sl_diff, 4), round(sl_pct, 2),
                round(atr_sl_pct, 2),
                round(tp_val, 4),  # ✅ tp
                round(rrr_val, 2) if pd.notna(rrr_val) else np.nan  # ✅ RRR
            ])
            remove_trade_ids.append(trade_id)
            hit = True
            break


# ---------------------------------------------------------
# ✅ Save with tp & RRR (16 columns total)
# ---------------------------------------------------------
if results:
    out = pd.DataFrame(results, columns=[
        "no", "symbol", "buy_date", "buy", "SL_value",
        "sell_date", "sell",
        "loss_pct", "profit_pct",
        "days_held", "Reference", "buy_sl_diff", "sl_pct", "atr_sl_pct",
        "tp", "RRR"  # ✅ 2 new columns
    ])
    out["no"] = range(1, len(out) + 1)
    out = out.sort_values("buy_sl_diff", ascending=True).reset_index(drop=True)
    out["no"] = range(1, len(out) + 1)
    out.to_csv(profit_loss_path, index=False)
    print(f"✅ Saved {len(out)} records to {profit_loss_path} (16 columns: +tp, +RRR)")
else:
    print("⚠️ No exits triggered.")


# Update trade_stock.csv (remove exited trades)
clean_trade = trade_df[~trade_df["trade_id"].isin(remove_trade_ids)].copy()
clean_trade = clean_trade.drop(columns=["trade_id"])
if not clean_trade.empty:
    clean_trade.insert(0, "no", range(1, len(clean_trade) + 1))
cols_order = ["no", "date", "symbol", "buy", "SL", "tp", "RRR", "Reference"]
clean_trade = clean_trade.reindex(columns=cols_order)
clean_trade.to_csv(trade_stock_path, index=False)

print("\n🎉 Done — now using real 'tp' & 'RRR' from signal files!")