import os
import sys
import joblib
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# ------------------ logging (ঐচ্ছিক) ------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------


class XGBoostTradingModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = [
            "open",
            "close",
            "high",
            "low",
            "volume",
            "rsi",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_lower",
            "atr",
            "returns",
            "volatility",
            "volume_change",
        ]

    # ------------------------------------------------------------------
    # 1. CSV খোঁজা
    # ------------------------------------------------------------------
    def _resolve_csv_paths(self) -> tuple[Path, Path]:
        """স্ক্রিপ্টের অবস্থান থেকে csv/ ফোল্ডার খোঁজা"""
        script_dir = Path(__file__).resolve().parent
        csv_dir = script_dir / ".." / "csv"  # scripts/ থেকে csv/
        mongodb_path = csv_dir / "mongodb.csv"
        trade_stock_path = csv_dir / "trade_stock.csv"

        if not mongodb_path.exists():
            raise FileNotFoundError(f"mongodb.csv পাওয়া যায়নি → {mongodb_path}")
        if not trade_stock_path.exists():
            raise FileNotFoundError(f"trade_stock.csv পাওয়া যায়নি → {trade_stock_path}")

        log.info("✅ CSV files found: %s, %s", mongodb_path, trade_stock_path)
        return mongodb_path, trade_stock_path

    # ------------------------------------------------------------------
    # 2. লোড + প্রিপেয়ার
    # ------------------------------------------------------------------
    def load_and_prepare_data(self):
        log.info("📊 Loading & preparing data ...")
        mongodb_path, trade_stock_path = self._resolve_csv_paths()

        market_df = pd.read_csv(mongodb_path)
        trade_df = pd.read_csv(trade_stock_path)

        log.info("Market data shape: %s", market_df.shape)
        log.info("Trade data shape: %s", trade_df.shape)

        # --- ফিচার ইঞ্জিনিয়ারিং ---
        market_df["returns"] = market_df["close"].pct_change()
        market_df["volatility"] = market_df["returns"].rolling(20).std()
        market_df["volume_change"] = market_df["volume"].pct_change()

        # --- মার্জ ---
        required_on = ["symbol", "date"]
        for col in required_on:
            if col not in market_df.columns:
                raise KeyError(f"'{col}' কলাম মার্কেট ডেটায় নেই")
            if col not in trade_df.columns:
                raise KeyError(f"'{col}' কলাম ট্রেড ডেটায় নেই")

        trade_small = trade_df[required_on + ["buy"]].drop_duplicates()
        merged = pd.merge(
            market_df, trade_small, on=required_on, how="left"
        )
        merged["target"] = merged["buy"].notna().astype(int)
        log.info("Buy signals: %d / %d", merged["target"].sum(), len(merged))

        # --- সিম্বল এনকোড ---
        if "symbol" in merged.columns:
            merged["symbol_encoded"] = self.label_encoder.fit_transform(
                merged["symbol"]
            )
            # ফিচার লিস্টে যোগ
            if "symbol_encoded" not in self.feature_names:
                self.feature_names.append("symbol_encoded")

        # --- ক্লিন ---
        merged = merged.dropna(subset=self.feature_names + ["target"])
        log.info("Final samples: %d", len(merged))

        return merged[self.feature_names], merged["target"]

    # ------------------------------------------------------------------
    # 3. ট্রেন
    # ------------------------------------------------------------------
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> float:
        if len(X) < 20:
            log.warning("Too few samples (%d) – skipping train", len(X))
            return 0.0

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        self.model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        log.info("📈 Accuracy: %.4f", acc)
        print(classification_report(y_test, y_pred, zero_division=0))
        return acc

    # ------------------------------------------------------------------
    # 4. সেভ
    # ------------------------------------------------------------------
    def save_model(self):
        if self.model is None:
            log.warning("No model to save")
            return

        script_dir = Path(__file__).resolve().parent
        models_dir = script_dir / ".." / "csv" / "models"
        models_dir.mkdir(exist_ok=True)

        save_path = models_dir / "xgboost_model.pkl"
        joblib.dump(
            {
                "model": self.model,
                "label_encoder": self.label_encoder,
                "feature_names": self.feature_names,
            },
            save_path,
        )
        log.info("✅ Model saved → %s", save_path)


# ----------------------------------------------------------------------
# মেইন
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("XGBoost Trading Model – Training")
    print("=" * 60)

    try:
        model = XGBoostTradingModel()
        X, y = model.load_and_prepare_data()

        if len(X):
            acc = model.train(X, y)
            if acc > 0:
                model.save_model()
                print(f"\n✅ Training complete – Accuracy: {acc:.4f}")
            else:
                print("\n⚠️ Training skipped (insufficient data or poor accuracy)")
        else:
            print("\n❌ No data to train")
    except Exception as e:
        log.exception("❌ Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
