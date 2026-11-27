#!/usr/bin/env python3
"""
Leakage-free trade signal training + backtest (one model for all symbols).

Rule: signal uses data up to the close of day t. Entry = next row's open.
Exit = close of the `horizon`-th row after t. Labels are net of costs.

Usage:
  python scripts/train_signal.py                 # train + walk-forward check
  python scripts/train_signal.py --final-test    # run ONCE; decides APPROVED or not
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from ensemble import SeedEnsemble

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
SEEDS = (42, 7, 123, 2024, 99)
FEATURES = ["ret_1", "ret_5", "ret_10", "ret_20", "vol_10", "vol_20", "range_pct",
            "gap_pct", "vol_ratio", "dist_sma20", "dist_sma50", "rsi_14", "pos_20",
            "mkt_ret_5", "rel_ret_5"]


def load_prices(path):
    df = pd.read_csv(path)
    need = ["symbol", "date", "open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")
    df = df[need].copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    for c in need[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df = df[(df[["open", "high", "low", "close", "volume"]] > 0).all(axis=1)]
    df = df.drop_duplicates(["symbol", "date"], keep="last")
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def add_features(df):
    """Every feature at row t uses rows <= t only."""
    df = df.copy()
    g = df.groupby("symbol", sort=False)

    def roll(col, n, fn):
        return df.groupby("symbol", sort=False)[col].transform(
            lambda x: getattr(x.rolling(n), fn)())

    prev = g["close"].shift(1)
    for n in (1, 5, 10, 20):
        df[f"ret_{n}"] = df["close"] / g["close"].shift(n) - 1
    df["vol_10"] = roll("ret_1", 10, "std")
    df["vol_20"] = roll("ret_1", 20, "std")
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["gap_pct"] = df["open"] / prev - 1
    df["vol_ratio"] = df["volume"] / roll("volume", 20, "mean")
    df["dist_sma20"] = df["close"] / roll("close", 20, "mean") - 1
    df["dist_sma50"] = df["close"] / roll("close", 50, "mean") - 1
    delta = df["close"] - prev
    df["_up"], df["_dn"] = delta.clip(lower=0), (-delta).clip(lower=0)
    up, dn = roll("_up", 14, "mean"), roll("_dn", 14, "mean")
    df["rsi_14"] = 100 * up / (up + dn).replace(0, np.nan)
    hi, lo = roll("high", 20, "max"), roll("low", 20, "min")
    df["pos_20"] = (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    df["_value"] = df["close"] * df["volume"]
    df["avg_value_20"] = roll("_value", 20, "mean")  # liquidity, used as a filter
    df["mkt_ret_5"] = df.groupby("date")["ret_5"].transform("mean")  # same-day only
    df["rel_ret_5"] = df["ret_5"] - df["mkt_ret_5"]
    return df.drop(columns=["_up", "_dn", "_value"])


def add_labels(df, horizon, cost, min_profit, max_gap):
    df = df.copy()
    g = df.groupby("symbol", sort=False)
    entry = g["open"].shift(-1)
    exit_ = g["close"].shift(-horizon)
    df["exit_date"] = g["date"].shift(-horizon)
    # cost = round-trip (commission + tax + slippage)
    df["net_ret"] = exit_ / entry - 1 - cost
    # an open far above yesterday's close usually cannot be filled (circuit limit)
    df["tradable"] = (entry / df["close"] - 1) < max_gap
    # unknown future stays NaN, never silently 0
    df["target"] = np.where(df["net_ret"].notna(),
                            (df["net_ret"] > min_profit).astype(float), np.nan)
    return df


def split_dates(df):
    dates = np.sort(df["date"].unique())
    return dates[int(len(dates) * TRAIN_FRAC)], dates[int(len(dates) * (TRAIN_FRAC + VAL_FRAC))]


def split(df, dates=None):
    """Date-based split. A row is dropped if its outcome window crosses a boundary."""
    val_start, test_start = dates if dates is not None else split_dates(df)
    d = df.dropna(subset=["target"])
    train = d[(d["date"] < val_start) & (d["exit_date"] < val_start)]
    val = d[(d["date"] >= val_start) & (d["date"] < test_start) & (d["exit_date"] < test_start)]
    test = d[d["date"] >= test_start]
    return train, val, test


def make_model(train, val):
    return SeedEnsemble([_fit_one(train, val, seed) for seed in SEEDS])


def _fit_one(train, val, seed):
    try:
        import xgboost as xgb
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        if seed == SEEDS[0]:
            print("WARNING: xgboost not installed -> using sklearn HistGradientBoosting")
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=200,
                                           min_samples_leaf=200, l2_regularization=5.0,
                                           early_stopping=False, random_state=seed)
        return m.fit(train[FEATURES], train["target"])
    # early_stopping_rounds in the constructor needs xgboost >= 1.6
    m = xgb.XGBClassifier(n_estimators=1000, max_depth=3, learning_rate=0.02,
                          subsample=0.7, colsample_bytree=0.7, min_child_weight=50,
                          reg_lambda=5.0, eval_metric="auc", early_stopping_rounds=50,
                          random_state=seed)
    return m.fit(train[FEATURES], train["target"],
                 eval_set=[(val[FEATURES], val["target"])], verbose=False)


def backtest(part, score, horizon, top_k, min_score=-np.inf):
    """
    Non-overlapping periods: every `horizon` days buy up to top_k names, equal slots,
    unused slots stay in cash. Averaged over all start offsets to remove start-day luck.
    """
    d = part.assign(score=np.asarray(score, dtype=float))
    dates = np.sort(d["date"].unique())
    picks = d[d["tradable"] & (d["score"] >= min_score)]
    picks = picks.sort_values(["date", "score"], ascending=[True, False]).groupby("date").head(top_k)
    daily = picks.groupby("date")["net_ret"].sum() / top_k
    daily = daily.reindex(dates, fill_value=0.0)
    totals, dds = [], []
    for off in range(horizon):
        eq = np.concatenate([[1.0], np.cumprod(1 + daily.iloc[off::horizon].to_numpy())])
        totals.append(eq[-1] - 1)
        dds.append((eq / np.maximum.accumulate(eq) - 1).min())
    return {"total_return": float(np.mean(totals)), "worst_start": float(np.min(totals)),
            "max_drawdown": float(np.mean(dds)), "trades": int(len(picks)),
            "win_rate": float((picks["net_ret"] > 0).mean()) if len(picks) else 0.0,
            "avg_net_per_trade": float(picks["net_ret"].mean()) if len(picks) else 0.0}


def feature_importance(model, val):
    """AUC lost on validation when one feature is shuffled. Near 0 or below = not useful."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    base = roc_auc_score(val["target"], model.predict_proba(val[FEATURES])[:, 1])
    drop = {}
    for f in FEATURES:
        X = val[FEATURES].copy()
        X[f] = rng.permutation(X[f].to_numpy())
        drop[f] = base - roc_auc_score(val["target"], model.predict_proba(X)[:, 1])
    print("\n===== FEATURE IMPORTANCE (validation AUC drop) =====")
    print(pd.Series(drop).sort_values(ascending=False).to_string(float_format=lambda v: f"{v:+.4f}"))


def pick_threshold(val, prob, args):
    grid = np.unique(np.quantile(prob, [0.5, 0.7, 0.8, 0.9, 0.95]))
    return float(max(grid, key=lambda t: backtest(
        val, prob, args.horizon, args.top_k, t)["total_return"]))


def walk_forward(df, test_start, args, folds=4):
    """Repeat train -> tune -> evaluate on successive periods BEFORE the test set."""
    d = df.dropna(subset=["target"])
    d = d[(d["date"] < test_start) & (d["exit_date"] < test_start)]
    dates = np.sort(d["date"].unique())
    cuts = np.linspace(int(len(dates) * 0.5), len(dates), folds + 1).astype(int)
    rows = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        start, end = dates[a], dates[b - 1]
        past = d[d["exit_date"] < start]
        past_dates = np.sort(past["date"].unique())
        inner = past_dates[int(len(past_dates) * 0.8)]
        tr = past[past["exit_date"] < inner]
        va = past[past["date"] >= inner]
        ev = d[(d["date"] >= start) & (d["date"] <= end)]
        if min(len(tr), len(va), len(ev)) < 100 or tr["target"].nunique() < 2:
            continue
        model = make_model(tr, va)
        thr = pick_threshold(va, model.predict_proba(va[FEATURES])[:, 1], args)
        prob = model.predict_proba(ev[FEATURES])[:, 1]
        m = backtest(ev, prob, args.horizon, args.top_k, thr)
        mkt = backtest(ev, np.zeros(len(ev)), args.horizon,
                       int(ev.groupby("date").size().max()))
        rows.append({"from": f"{pd.Timestamp(start):%Y-%m-%d}",
                     "to": f"{pd.Timestamp(end):%Y-%m-%d}",
                     "model": m["total_return"], "market": mkt["total_return"],
                     "trades": m["trades"]})
    if rows:
        print("\n===== WALK-FORWARD (net of cost) =====")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return rows


def judge(res, folds):
    m = res["model"]
    checks = {
        "test AUC > 0.52": res["auc"] > 0.52,
        "test return > 0": m["total_return"] > 0,
        "beats momentum rule": m["total_return"] > res["momentum rule"]["total_return"],
        "beats buy-all": m["total_return"] > res["buy all (market)"]["total_return"],
        "at least 30 test trades": m["trades"] >= 30,
        "profitable in >=3 walk-forward folds": sum(f["model"] > 0 for f in folds) >= 3,
    }
    print("\n===== VERDICT =====")
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    approved = all(checks.values())
    return bool(approved)


def report(name, part, prob, args, threshold):
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(part["target"], prob) if part["target"].nunique() > 1 else float("nan")
    print(f"\n===== {name}: {part['date'].min():%Y-%m-%d} .. {part['date'].max():%Y-%m-%d} "
          f"| rows={len(part)} | AUC={auc:.3f} =====")
    n_all = int(part.groupby("date").size().max())
    rows = {
        "model": backtest(part, prob, args.horizon, args.top_k, threshold),
        "momentum rule": backtest(part, part["ret_20"], args.horizon, args.top_k, 0.0),
        "buy all (market)": backtest(part, np.zeros(len(part)), args.horizon, n_all),
    }
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.4f}"))
    return {"auc": float(auc), **rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./csv/mongodb.csv")
    p.add_argument("--out", default="./csv/signal_model")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--cost", type=float, default=0.01, help="round-trip cost, 0.01 = 1%%")
    p.add_argument("--min-profit", type=float, default=0.01)
    p.add_argument("--max-gap", type=float, default=0.09)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-value", type=float, default=0.0,
                   help="minimum 20-day average traded value (close*volume)")
    p.add_argument("--final-test", action="store_true")
    args = p.parse_args()

    df = add_features(load_prices(args.data))
    df = add_labels(df, args.horizon, args.cost, args.min_profit, args.max_gap)
    df = df.dropna(subset=["ret_20", "dist_sma50"])
    if not args.final_test:
        df = df[df["avg_value_20"] >= args.min_value]
    train, val, test = split(df)
    if min(len(train), len(val), len(test)) < 100:
        raise SystemExit(f"Not enough rows: train={len(train)} val={len(val)} test={len(test)}")

    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, "model.joblib")

    if args.final_test:
        saved = joblib.load(model_path)           # frozen model + threshold
        for k, v in saved["settings"].items():    # same settings as at training time
            setattr(args, k, v)
        df = add_features(load_prices(args.data))
        df = add_labels(df, args.horizon, args.cost, args.min_profit, args.max_gap)
        df = df.dropna(subset=["ret_20", "dist_sma50"])
        df = df[df["avg_value_20"] >= args.min_value]
        test = split(df, saved["split_dates"])[2]
        if saved["test_runs"]:
            print("WARNING: this test set was already viewed; the result is now optimistic")
        res = report("TEST", test, saved["model"].predict_proba(test[FEATURES])[:, 1],
                     args, saved["threshold"])
        verdict = judge(res, saved["walk_forward"])
        if not saved["test_runs"]:                # only the first look can approve a model
            saved["approved"] = verdict
        print("Final status: " + ("APPROVED" if saved["approved"] else
                                  "NOT APPROVED -> signals stay PAPER only"))
        saved["test_runs"] += 1
        joblib.dump(saved, model_path)
    else:
        dates = split_dates(df)
        folds = walk_forward(df, dates[1], args)
        model = make_model(train, val)
        prob = model.predict_proba(val[FEATURES])[:, 1]
        threshold = pick_threshold(val, prob, args)   # VALIDATION only
        res = report("VALIDATION (optimistic: threshold tuned here)", val, prob, args, threshold)
        feature_importance(model, val)
        settings = {k: getattr(args, k) for k in
                    ("horizon", "cost", "min_profit", "max_gap", "top_k", "min_value")}
        # keep the count if the test period is unchanged, so retries cannot reset it
        runs = 0
        if os.path.exists(model_path):
            old = joblib.load(model_path)
            if old.get("split_dates", (None, None))[1] == dates[1]:
                runs = old.get("test_runs", 0)
        joblib.dump({"model": model, "threshold": threshold, "features": FEATURES,
                     "settings": settings, "split_dates": dates, "walk_forward": folds,
                     "approved": False, "test_runs": runs}, model_path)
        print(f"\nSaved to {args.out} (threshold={threshold:.3f}). Not approved until --final-test.")

    with open(os.path.join(args.out, "test.json" if args.final_test else "validation.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
