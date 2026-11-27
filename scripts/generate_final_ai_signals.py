#!/usr/bin/env python3
"""
Daily signals from the tested model only. No votes from untested models.
The explanation is built from the same numbers the model saw; it never changes a signal.
"""
import argparse
import os

import joblib
import numpy as np

from train_signal import add_features, load_prices


def explain(r):
    trend = "উপরে" if r.dist_sma50 > 0 else "নিচে"
    rsi = "বেশি কেনা অঞ্চলে" if r.rsi_14 > 70 else "বেশি বিক্রি অঞ্চলে" if r.rsi_14 < 30 else "মাঝামাঝি"
    return (f"৫ দিনে {r.ret_5:+.1%}, ২০ দিনে {r.ret_20:+.1%}; দাম ৫০ দিনের গড়ের {trend} "
            f"({r.dist_sma50:+.1%}); RSI {r.rsi_14:.0f} ({rsi}); ভলিউম গড়ের {r.vol_ratio:.1f} গুণ; "
            f"বাজার ৫ দিনে {r.mkt_ret_5:+.1%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./csv/mongodb.csv")
    p.add_argument("--model", default="./csv/signal_model/model.joblib")
    p.add_argument("--out", default="./output/ai_signal/FINAL_AI_SIGNALS.csv")
    args = p.parse_args()

    saved = joblib.load(args.model)
    s = saved["settings"]
    df = add_features(load_prices(args.data))
    last = df["date"].max()
    # only symbols that traded on the latest day and pass the liquidity filter
    d = df[(df["date"] == last) & (df["avg_value_20"] >= s["min_value"])]
    d = d.dropna(subset=["ret_20", "dist_sma50"]).copy()
    if d.empty:
        raise SystemExit("No symbol has enough fresh data")

    d["prob"] = saved["model"].predict_proba(d[saved["features"]])[:, 1]
    d = d.sort_values("prob", ascending=False)
    buy = (d["prob"] >= saved["threshold"]) & (np.arange(len(d)) < s["top_k"])
    d["final_signal"] = np.where(buy, "BUY", "NO TRADE")
    d["status"] = "APPROVED" if saved["approved"] else "PAPER ONLY"
    d["final_combined_score"] = (d["prob"] * 100).round(1)
    d["current_price"] = d["close"]
    d["max_entry_price"] = (d["close"] * (1 + s["max_gap"])).round(2)
    d["exit_after_days"] = s["horizon"]
    d["explanation"] = [explain(r) for r in d.itertuples()]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["symbol", "date", "current_price", "high", "final_signal", "final_combined_score",
            "status", "max_entry_price", "exit_after_days", "explanation"]
    d[cols].to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"{last:%Y-%m-%d}: {int(buy.sum())} BUY of {len(d)} symbols | "
          f"{d['status'].iloc[0]} -> {args.out}")


if __name__ == "__main__":
    main()
