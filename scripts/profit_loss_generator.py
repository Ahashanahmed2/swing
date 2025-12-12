import pandas as pd
import os
import numpy as np
import json

# ---------------------------------------------------------
# 🔧 Load config.json
# ---------------------------------------------------------
CONFIG_PATH = "./config.json"
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    TOTAL_CAPITAL = float(config.get("total_capital", 500000))
    RISK_PERCENT = float(config.get("risk_percent", 0.01))
    print(f"✅ Config loaded: capital={TOTAL_CAPITAL:,.0f} BDT, risk={RISK_PERCENT*100:.1f}%")
except FileNotFoundError:
    print(f"⚠️ {CONFIG_PATH} not found → using defaults: 5,00,000 BDT, 1% risk")
    TOTAL_CAPITAL = 500000
    RISK_PERCENT = 0.01
except Exception as e:
    print(f"❌ Error loading config: {e} → using defaults")
    TOTAL_CAPITAL = 500000
    RISK_PERCENT = 0.01

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
metrics_path = "./output/ai_signal/performance_metrics.csv"
strategy_metrics_path = "./output/ai_signal/strategy_metrics.csv"
symbol_ref_metrics_path = "./output/ai_signal/symbol_reference_metrics.csv"

os.makedirs(os.path.dirname(profit_loss_path), exist_ok=True)


# ---------------------------------------------------------
# Helper: Load BUY data (with tp & RRR)
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

    # Ensure consistent column names
    if "Price" in df.columns and "SL" not in df.columns:
        df = df.rename(columns={"Price": "SL"})
    if "last_row_close" in df.columns and "buy" not in df.columns:
        df = df.rename(columns={"last_row_close": "buy"})

    df = df.dropna(subset=["symbol", "date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    result = {
        "date": df["date"].dt.date,
        "symbol": df["symbol"].astype(str).str.strip().str.upper(),
        "buy": pd.to_numeric(df[buy_col], errors="coerce"),
        "SL": pd.to_numeric(df["SL"], errors="coerce"),
        "Reference": reference
    }

    result["tp"] = pd.to_numeric(df[tp_col], errors="coerce") if tp_col in df.columns else np.nan
    result["RRR"] = pd.to_numeric(df[rrr_col], errors="coerce") if rrr_col in df.columns else np.nan

    out_df = pd.DataFrame(result).dropna(subset=["buy", "SL"])

    # Fill missing tp from RRR
    mask = out_df["tp"].isna() & out_df["RRR"].notna()
    if mask.any():
        out_df.loc[mask, "tp"] = out_df.loc[mask, "buy"] + out_df.loc[mask, "RRR"] * (out_df.loc[mask, "buy"] - out_df.loc[mask, "SL"])
    return out_df


# ---------------------------------------------------------
# ✅ DSEX-Optimized k Selector
# ---------------------------------------------------------
def get_dsex_k(market_cap_million, atr_pct):
    market_cap_cr = market_cap_million / 10 if pd.notna(market_cap_million) else 0
    if market_cap_cr >= 5000:
        return 1.2 if atr_pct < 3.0 else (1.4 if atr_pct <= 5.0 else 1.6)
    elif market_cap_cr >= 500:
        return 1.5
    else:
        return 1.6 if atr_pct < 4.0 else 1.8


# ---------------------------------------------------------
# Load & size signals
# ---------------------------------------------------------
short_df = load_file(short_path, "short", "buy")
gape_df = load_file(gape_path, "gape", "buy")
rsi_df = load_file(rsi_path, "rsi", "buy")
swing_df = load_file(swing_path, "swing", "buy")

trade_df = pd.concat([short_df, gape_df, rsi_df, swing_df], ignore_index=True)
print(f"✅ Loaded {len(trade_df)} trade signals.")

if trade_df.empty:
    print("🛑 No valid trade signals. Exiting.")
    exit()

trade_df = trade_df.reset_index(drop=True)
trade_df["trade_id"] = trade_df.index

# ✅ Position Sizing (auto from config)
trade_df["risk_per_trade"] = TOTAL_CAPITAL * RISK_PERCENT
trade_df["position_size"] = (trade_df["risk_per_trade"] / (trade_df["buy"] - trade_df["SL"])).astype(float)
trade_df["position_size"] = trade_df["position_size"].clip(lower=100).fillna(100)
trade_df["position_size"] = (trade_df["position_size"] // 100 * 100).astype(int)
trade_df["position_size"] = trade_df["position_size"].clip(upper=10000)
trade_df["exposure_bdt"] = trade_df["position_size"] * trade_df["buy"]
trade_df["actual_risk_bdt"] = trade_df["position_size"] * (trade_df["buy"] - trade_df["SL"])


# ---------------------------------------------------------
# Load mongodb
# ---------------------------------------------------------
if not os.path.exists(mongodb_path):
    raise FileNotFoundError(f"❌ mongodb.csv not found at {mongodb_path}")

try:
    mongodb = pd.read_csv(mongodb_path)
    assert not mongodb.empty, "mongodb.csv is empty!"
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
# Profit–Loss Calculator
# ---------------------------------------------------------
results = []
remove_trade_ids = []

for _, row in trade_df.iterrows():
    symbol, buy_date, buy = row["symbol"], row["date"], float(row["buy"])
    SL_value = float(row["SL"])
    tp_val = float(row["tp"]) if pd.notna(row["tp"]) else buy * 1.10
    rrr_val = float(row["RRR"]) if pd.notna(row["RRR"]) else np.nan
    trade_id, ref = row["trade_id"], row["Reference"]
    pos_size = int(row["position_size"])
    exp_bdt = float(row["exposure_bdt"])

    buy_sl_diff = buy - SL_value
    sl_pct = ((buy - SL_value) / buy) * 100 if buy > 0 else np.nan

    df_sym = mongodb[mongodb["symbol"] == symbol]
    if df_sym.empty:
        continue

    df_sym = df_sym.copy()
    df_sym["date_only"] = df_sym["date"].dt.date
    buy_rows = df_sym[df_sym["date_only"] == buy_date]
    if buy_rows.empty:
        continue

    buy_row = buy_rows.iloc[0]
    atr = buy_row["atr"]
    market_cap_million = buy_row.get("marketCap", np.nan)

    atr_pct = (atr / buy) * 100 if atr > 0 else 3.0
    k = get_dsex_k(market_cap_million, atr_pct)
    atr_sl_pct = (k * atr / buy) * 100

    buy_idx = buy_rows.index[0]
    future_rows = df_sym.loc[df_sym.index > buy_idx].sort_values("date")
    if future_rows.empty:
        continue

    for _, r in future_rows.iterrows():
        close = r["close"]
        cur_date = r["date"].date()
        diff_days = (cur_date - buy_date).days

        # SL hit
        if close < SL_value:
            loss_pct = ((buy - close) / buy) * 100
            results.append([
                None, symbol, buy_date, buy, SL_value,
                cur_date, close, round(loss_pct, 2), np.nan,
                diff_days, ref, round(buy_sl_diff, 4),
                round(sl_pct, 2), round(atr_sl_pct, 2),
                round(tp_val, 4), round(rrr_val, 2) if pd.notna(rrr_val) else np.nan,
                pos_size, round(exp_bdt, 0)
            ])
            remove_trade_ids.append(trade_id)
            break

        # TP hit
        if close >= tp_val:
            profit_pct = ((close - buy) / buy) * 100
            results.append([
                None, symbol, buy_date, buy, SL_value,
                cur_date, close, np.nan, round(profit_pct, 2),
                diff_days, ref, round(buy_sl_diff, 4),
                round(sl_pct, 2), round(atr_sl_pct, 2),
                round(tp_val, 4), round(rrr_val, 2) if pd.notna(rrr_val) else np.nan,
                pos_size, round(exp_bdt, 0)
            ])
            remove_trade_ids.append(trade_id)
            break


# ---------------------------------------------------------
# ✅ Save profit-loss.csv
# ---------------------------------------------------------
if results:
    out = pd.DataFrame(results, columns=[
        "no", "symbol", "buy_date", "buy", "SL_value",
        "sell_date", "sell", "loss_pct", "profit_pct", "days_held",
        "Reference", "buy_sl_diff", "sl_pct", "atr_sl_pct", "tp", "RRR",
        "position_size", "exposure_bdt"
    ])
    out["no"] = range(1, len(out) + 1)
    out = out.sort_values("buy_sl_diff", ascending=True).reset_index(drop=True)
    out["no"] = range(1, len(out) + 1)
    out.to_csv(profit_loss_path, index=False)
    print(f"✅ Saved {len(out)} records to {profit_loss_path}")
else:
    print("⚠️ No exits triggered.")


# ---------------------------------------------------------
# ✅ GLOBAL PERFORMANCE METRICS
# ---------------------------------------------------------
global_metrics = {}
if results:
    profits = [r[8] for r in results if pd.notna(r[8])]
    losses = [abs(r[7]) for r in results if pd.notna(r[7])]
    wins, losses_cnt = len(profits), len(losses)
    total = wins + losses_cnt

    if total > 0:
        win_rate = wins / total
        avg_win = np.mean(profits) if wins else 0
        avg_loss = np.mean(losses) if losses_cnt else 0
        profit_factor = (sum(profits) / sum(losses)) if losses_cnt and sum(losses) else np.inf
        expectancy_pct = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # In BDT
        profit_bdt = [r[17] * (r[8] / 100) for r in results if pd.notna(r[8])]
        loss_bdt = [r[17] * (r[7] / 100) for r in results if pd.notna(r[7])]
        avg_win_bdt = np.mean(profit_bdt) if profit_bdt else 0
        avg_loss_bdt = np.mean(loss_bdt) if loss_bdt else 0
        expectancy_bdt = (win_rate * avg_win_bdt) - ((1 - win_rate) * avg_loss_bdt)

        global_metrics = {
            "Total Trades": total,
            "Wins": wins,
            "Losses": losses_cnt,
            "Win Rate (%)": round(win_rate * 100, 2),
            "Avg Win (%)": round(avg_win, 2),
            "Avg Loss (%)": round(avg_loss, 2),
            "Profit Factor": round(profit_factor, 2) if np.isfinite(profit_factor) else "∞",
            "Expectancy (%)": round(expectancy_pct, 2),
            "Avg Win (BDT)": round(avg_win_bdt, 0),
            "Avg Loss (BDT)": round(avg_loss_bdt, 0),
            "Expectancy (BDT)": round(expectancy_bdt, 2)
        }

        metrics_df = pd.DataFrame(list(global_metrics.items()), columns=["Metric", "Value"])
        metrics_df.to_csv(metrics_path, index=False)
        print(f"✅ Global metrics saved to {metrics_path}")


# ---------------------------------------------------------
# ✅ STRATEGY-BASED METRICS (by Reference)
# ---------------------------------------------------------
strategy_metrics = []
if results:
    for ref in ["swing", "gape", "rsi", "short"]:
        ref_results = [r for r in results if r[10] == ref]  # r[10] = Reference
        if not ref_results:
            continue

        profits = [r[8] for r in ref_results if pd.notna(r[8])]
        losses = [abs(r[7]) for r in ref_results if pd.notna(r[7])]
        wins, losses_cnt = len(profits), len(losses)
        total = wins + losses_cnt

        if total == 0:
            continue

        win_rate = wins / total
        avg_win = np.mean(profits) if wins else 0
        avg_loss = np.mean(losses) if losses_cnt else 0
        profit_factor = (sum(profits) / sum(losses)) if losses_cnt and sum(losses) else np.inf
        expectancy_pct = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # In BDT
        profit_bdt = [r[17] * (r[8] / 100) for r in ref_results if pd.notna(r[8])]
        loss_bdt = [r[17] * (r[7] / 100) for r in ref_results if pd.notna(r[7])]
        avg_win_bdt = np.mean(profit_bdt) if profit_bdt else 0
        avg_loss_bdt = np.mean(loss_bdt) if loss_bdt else 0
        expectancy_bdt = (win_rate * avg_win_bdt) - ((1 - win_rate) * avg_loss_bdt)

        strategy_metrics.append({
            "Reference": ref.upper(),
            "Trades": total,
            "Win%": round(win_rate * 100, 1),
            "Avg Win (BDT)": round(avg_win_bdt, 0),
            "Avg Loss (BDT)": round(avg_loss_bdt, 0),
            "Profit Factor": round(profit_factor, 2) if np.isfinite(profit_factor) else "∞",
            "Expectancy (BDT)": round(expectancy_bdt, 2)
        })

    if strategy_metrics:
        strategy_df = pd.DataFrame(strategy_metrics)
        strategy_df.to_csv(strategy_metrics_path, index=False)
        print(f"✅ Strategy-wise metrics saved to {strategy_metrics_path}")


# ---------------------------------------------------------
# ✅ SYMBOL × REFERENCE METRICS (2D Win%)
# ---------------------------------------------------------
symbol_ref_metrics = []
if results:
    symbols = sorted(set(r[1] for r in results))  # r[1] = symbol
    refs = ["swing", "gape", "rsi", "short"]

    for sym in symbols:
        for ref in refs:
            filtered = [r for r in results if r[1] == sym and r[10] == ref]
            if not filtered:
                continue

            profits = [r[8] for r in filtered if pd.notna(r[8])]
            losses = [abs(r[7]) for r in filtered if pd.notna(r[7])]
            wins, losses_cnt = len(profits), len(losses)
            total = wins + losses_cnt

            if total == 0:
                continue

            win_rate = wins / total
            avg_win = np.mean(profits) if wins else 0
            avg_loss = np.mean(losses) if losses_cnt else 0
            profit_factor = (sum(profits) / sum(losses)) if losses_cnt and sum(losses) else np.inf

            # In BDT
            profit_bdt = [r[17] * (r[8] / 100) for r in filtered if pd.notna(r[8])]
            loss_bdt = [r[17] * (r[7] / 100) for r in filtered if pd.notna(r[7])]
            avg_win_bdt = np.mean(profit_bdt) if profit_bdt else 0
            avg_loss_bdt = np.mean(loss_bdt) if loss_bdt else 0
            expectancy_bdt = (win_rate * avg_win_bdt) - ((1 - win_rate) * avg_loss_bdt)

            symbol_ref_metrics.append({
                "Symbol": sym,
                "Reference": ref.upper(),
                "Trades": total,
                "Win%": round(win_rate * 100, 1),
                "Avg Win (BDT)": round(avg_win_bdt, 0),
                "Avg Loss (BDT)": round(avg_loss_bdt, 0),
                "Profit Factor": round(profit_factor, 2) if np.isfinite(profit_factor) else "∞",
                "Expectancy (BDT)": round(expectancy_bdt, 2)
            })

    if symbol_ref_metrics:
        sym_ref_df = pd.DataFrame(symbol_ref_metrics)
        sym_ref_df.to_csv(symbol_ref_metrics_path, index=False)
        print(f"✅ Symbol × Strategy metrics saved to {symbol_ref_metrics_path}")

        # Top 10 combos by Expectancy
        top10 = sorted(symbol_ref_metrics, key=lambda x: x["Expectancy (BDT)"], reverse=True)[:10]
        if top10:
            print("\n" + "="*85)
            print("🔍 TOP SYMBOL × STRATEGY COMBINATIONS (by Expectancy)")
            print("="*85)
            print(f"{'Symbol':<10} {'Ref':<8} {'Trades':<8} {'Win%':<8} {'Avg Win':<10} {'Avg Loss':<10} {'Exp (BDT)':<10}")
            print("-"*85)
            for r in top10:
                print(
                    f"{r['Symbol']:<10} "
                    f"{r['Reference']:<8} "
                    f"{r['Trades']:<8} "
                    f"{r['Win%']:<8.1f} "
                    f"{r['Avg Win (BDT)']:>+10,.0f} "
                    f"{r['Avg Loss (BDT)']:>+10,.0f} "
                    f"{r['Expectancy (BDT)']:+10.2f}"
                )
            print("="*85)


# ---------------------------------------------------------
# ✅ PRINT GLOBAL + STRATEGY METRICS
# ---------------------------------------------------------
if global_metrics:
    print("\n" + "="*50)
    print("📊 GLOBAL PERFORMANCE")
    print("="*50)
    print(f"✅ Win Rate       : {global_metrics['Win Rate (%)']:.1f}% ({global_metrics['Wins']}/{global_metrics['Total Trades']})")
    print(f"📈 Avg Win        : +{global_metrics['Avg Win (%)']:.2f}%  ({global_metrics['Avg Win (BDT)']:,.0f} BDT)")
    print(f"📉 Avg Loss       : -{global_metrics['Avg Loss (%)']:.2f}%  ({global_metrics['Avg Loss (BDT)']:,.0f} BDT)")
    print(f"💰 Profit Factor  : {global_metrics['Profit Factor']}")
    print(f"🎯 Expectancy     : {global_metrics['Expectancy (%)']:+.2f}%  ({global_metrics['Expectancy (BDT)']:+.2f} BDT)")
    print("="*50)

if strategy_metrics:
    print("\n" + "="*70)
    print("🔍 STRATEGY PERFORMANCE (by Reference)")
    print("="*70)
    print(f"{'Ref':<8} {'Trades':<8} {'Win%':<8} {'Avg Win':<10} {'Avg Loss':<10} {'PF':<6} {'Exp (BDT)':<10}")
    print("-"*70)
    for r in strategy_metrics:
        print(
            f"{r['Reference']:<8} "
            f"{r['Trades']:<8} "
            f"{r['Win%']:<8.1f} "
            f"{r['Avg Win (BDT)']:>+10,.0f} "
            f"{r['Avg Loss (BDT)']:>+10,.0f} "
            f"{r['Profit Factor']:<6} "
            f"{r['Expectancy (BDT)']:+10.2f}"
        )
    print("="*70)

    best = max(strategy_metrics, key=lambda x: x["Expectancy (BDT)"])
    worst = min(strategy_metrics, key=lambda x: x["Expectancy (BDT)"])
    print(f"🌟 Best:  {best['Reference']} → {best['Expectancy (BDT)']:+.2f} BDT/trade")
    print(f"⚠️  Worst: {worst['Reference']} → {worst['Expectancy (BDT)']:+.2f} BDT/trade")

    if worst["Expectancy (BDT)"] < 0:
        print(f"\n❗ Action: Consider pausing {worst['Reference']} until optimized.")


# ---------------------------------------------------------
# ✅ SMART MERGE: trade_stock.csv
# ---------------------------------------------------------
def merge_open_trades(old_path, new_df, exited_ids):
    old_df = pd.DataFrame()
    if os.path.exists(old_path):
        try:
            old_df = pd.read_csv(old_path)
            for col in ["symbol", "date", "Reference", "buy", "SL"]:
                assert col in old_df.columns, f"Missing in old: {col}"
            old_df["symbol"] = old_df["symbol"].str.upper().str.strip()
            old_df["date"] = pd.to_datetime(old_df["date"]).dt.date
            if "no" in old_df.columns:
                old_df = old_df.drop(columns=["no"])
        except Exception as e:
            print(f"⚠️ Error loading old trades: {e}")

    exited_keys = {(r[1], r[2], r[10]) for r in results} if results else set()

    if not old_df.empty:
        old_df["key"] = list(zip(old_df["symbol"], old_df["date"], old_df["Reference"]))
        old_open = old_df[~old_df["key"].isin(exited_keys)].drop(columns=["key"])
    else:
        old_open = old_df.copy()

    new_clean = new_df.copy()
    new_clean["symbol"] = new_clean["symbol"].str.upper().str.strip()
    new_clean["date"] = pd.to_datetime(new_clean["date"]).dt.date
    cols_to_drop = ["trade_id", "no", "risk_per_trade"]
    new_clean = new_clean.drop(columns=[c for c in cols_to_drop if c in new_clean.columns])

    combined = pd.concat([old_open, new_clean], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["symbol", "date", "Reference"], keep="last"
    ).reset_index(drop=True)

    combined["DIFF"] = combined["buy"] - combined["SL"]
    combined = combined.sort_values("DIFF", ascending=True).reset_index(drop=True)
    combined.insert(0, "no", range(1, len(combined) + 1))

    col_order = [
        "no", "date", "symbol", "buy", "SL", "tp", "RRR",
        "position_size", "exposure_bdt", "actual_risk_bdt", "Reference"
    ]
    return combined.reindex(columns=col_order)


final_trades = merge_open_trades(trade_stock_path, trade_df, remove_trade_ids)
final_trades.to_csv(trade_stock_path, index=False)
print(f"\n✅ Updated trade_stock.csv: {len(final_trades)} open signals.")

print("\n🎉 SYSTEM READY — Global + Strategy + Symbol×Strategy Win% & Expectancy!")