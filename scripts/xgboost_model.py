# ===============================
# XGBOOST TECHNICAL TRADING MODEL
# ===============================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
warnings.filterwarnings("ignore")


# =========================================================
# MODEL CLASS
# =========================================================

class XGBoostTradingModel:

    def __init__(self, n_estimators=1000, max_depth=5, learning_rate=0.01):
        self.model = None
        self.regression_model = None
        self.feature_importance = None

        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }

    # -----------------------------------------------------
    # DATA PREPARATION WITH TECHNICAL INDICATORS
    # -----------------------------------------------------
    def prepare_data_with_technical_indicators(self, market_data, trade_data):

        market_data = market_data.copy()
        trade_data = trade_data.copy()

        market_data["date"] = pd.to_datetime(market_data["date"])
        trade_data["date"] = pd.to_datetime(trade_data["date"])

        market_data = market_data.sort_values("date")

        symbol = market_data["symbol"].iloc[0] if len(market_data) else "UNKNOWN"
        print(f"📊 {symbol} | Preparing technical dataset...")

        samples = []

        for _, trade in trade_data.iterrows():

            buy_date = trade["date"]
            buy_price = trade["buy"]

            sl_price = trade.get("SL", buy_price * 0.95)
            tp_price = trade.get("tp", buy_price * 1.10)

            if sl_price <= 0:
                sl_price = buy_price * 0.95
            if tp_price <= buy_price:
                tp_price = buy_price * 1.10

            buy_day_data = market_data[market_data["date"] == buy_date]
            if buy_day_data.empty:
                continue

            buy_row = buy_day_data.iloc[0]

            for days_ahead in range(1, 11):

                future_date = buy_date + pd.Timedelta(days=days_ahead)
                future_row = market_data[market_data["date"] == future_date]

                if future_row.empty:
                    continue

                row = future_row.iloc[0]

                high = row["high"]
                low = row["low"]
                close = row["close"]

                sl_hit = low <= sl_price
                tp_hit = high >= tp_price
                pnl = (close - buy_price) / buy_price

                reward = 0.0

                if sl_hit:
                    reward = -1.0 * (1 - days_ahead / 20)
                elif tp_hit:
                    reward = 1.0 * (0.5 + days_ahead / 20)
                else:
                    reward = pnl * (0.2 if pnl > 0 else 0.5)

                features = buy_row.to_dict()

                features.update({
                    "days_ahead": days_ahead,
                    "buy_price": buy_price,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "sl_distance_pct": max((buy_price - sl_price) / buy_price, 0.001),
                    "tp_distance_pct": (tp_price - buy_price) / buy_price,
                    "risk_reward_ratio": (tp_price - buy_price) / max(buy_price - sl_price, 0.001),
                    "current_profit_loss_pct": pnl,
                    "sl_hit": int(sl_hit),
                    "tp_hit": int(tp_hit),
                    "reward": reward,
                    "symbol": symbol
                })

                rsi = features.get("rsi", 50)
                features["rsi_oversold"] = int(rsi < 30)
                features["rsi_overbought"] = int(rsi > 70)
                features["rsi_neutral"] = int(30 <= rsi <= 70)

                features["macd_cross"] = int(
                    features.get("macd", 0) > features.get("macd_signal", 0)
                )

                for p in ["Hammer", "BullishEngulfing", "MorningStar", "Doji"]:
                    features[f"{p}_present"] = int(features.get(p, 0))

                samples.append(features)

        if not samples:
            return pd.DataFrame(), []

        df = pd.DataFrame(samples)

        df["signal_binary"] = (df["reward"] > 0).astype(int)
        df["reward_regression"] = df["reward"]

        feature_cols = [
            "open", "high", "low", "close", "volume",
            "buy_price", "sl_price", "tp_price", "days_ahead",
            "sl_distance_pct", "tp_distance_pct", "risk_reward_ratio",
            "rsi", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower", "atr",
            "rsi_oversold", "rsi_overbought", "rsi_neutral",
            "macd_cross",
            "Hammer_present", "BullishEngulfing_present",
            "MorningStar_present", "Doji_present"
        ]

        available_features = [f for f in feature_cols if f in df.columns]

        df[available_features] = df[available_features].fillna(df[available_features].median())

        return df, available_features

    # -----------------------------------------------------
    # TRAIN MODEL
    # -----------------------------------------------------
    def train_with_technical_indicators(self, market_data, trade_data):

        data, features = self.prepare_data_with_technical_indicators(market_data, trade_data)

        if len(data) < 20 or len(features) < 8:
            return 0.0, 0.0

        X = data[features]
        y = data["reward_regression"]
        y_bin = data["signal_binary"]

        X_train, X_test, y_train, y_test, yb_train, yb_test = train_test_split(
            X, y, y_bin, test_size=0.3, random_state=42,
            stratify=y_bin if y_bin.nunique() > 1 else None
        )

        self.regression_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        )

        self.regression_model.fit(X_train, y_train)

        preds = self.regression_model.predict(X_test)

        r2 = r2_score(y_test, preds)
        acc = accuracy_score(yb_test, (preds > 0).astype(int))

        self.feature_importance = pd.DataFrame({
            "feature": features,
            "importance": self.regression_model.feature_importances_
        }).sort_values("importance", ascending=False)

        return max(0, r2), acc

    # -----------------------------------------------------
    # SAVE / LOAD
    # -----------------------------------------------------
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.regression_model.save_model(path)

    def load_model(self, path):
        self.regression_model = xgb.XGBRegressor()
        self.regression_model.load_model(path)


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./csv", exist_ok=True)

    market_data = pd.read_csv("./csv/mongodb.csv")
    trade_data = pd.read_csv("./csv/trade_stock.csv")

    symbols = sorted(set(market_data["symbol"]) & set(trade_data["symbol"]))

    results = []

    for symbol in symbols:

        print(f"\n🔄 Training {symbol}")

        mkt = market_data[market_data["symbol"] == symbol]
        trd = trade_data[trade_data["symbol"] == symbol]

        model = XGBoostTradingModel()

        r2, acc = model.train_with_technical_indicators(mkt, trd)

        results.append({
            "symbol": symbol,
            "r2": r2,
            "accuracy": acc
        })

        if r2 > 0.3 and acc > 0.6:
            model.save_model(f"./models/xgb_{symbol}.json")

    pd.DataFrame(results).to_csv("./csv/xgboost_training_summary.csv", index=False)

    print("\n✅ ALL DONE")