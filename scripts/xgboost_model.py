
# src/xgboost_model_v2.py  (strategy-aware)
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------
# Config fallback
# ------------------------------------------------------
try:
    from config import MAX_POSITION_SIZE, MAX_PORTFOLIO_RISK, STOP_LOSS_PCT
except ImportError:
    MAX_POSITION_SIZE = 0.02
    MAX_PORTFOLIO_RISK = 0.01
    STOP_LOSS_PCT = 0.02


# ==============================================================================
# XGBoost Model (strategy-aware)
# ==============================================================================
class XGBoostTradingModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = [
            "open", "close", "high", "low", "volume", "rsi", "macd",
            "macd_signal", "bb_upper", "bb_lower", "atr", "returns",
            "volatility", "volume_change", "symbol_encoded",
            "RRR", "SL_pct", "TP_pct"  # <-- NEW strategy features
        ]

    # ------------------------------------------------------------------
    # 1. Resolve CSV paths
    # ------------------------------------------------------------------
    def _resolve_csv_paths(self) -> tuple[Path, Path]:
        script_dir = Path(__file__).resolve().parent
        csv_dir = script_dir / ".." / "csv"
        mongodb_path = csv_dir / "mongodb.csv"
        trade_stock_path = csv_dir / "trade_stock.csv"

        if not mongodb_path.exists():
            raise FileNotFoundError(f"mongodb.csv not found → {mongodb_path}")
        if not trade_stock_path.exists():
            raise FileNotFoundError(f"trade_stock.csv not found → {trade_stock_path}")

        log.info("✅ CSV files found: %s, %s", mongodb_path, trade_stock_path)
        return mongodb_path, trade_stock_path

    # ------------------------------------------------------------------
    # 2. Load + prepare (with strategy features)
    # ------------------------------------------------------------------
    def load_and_prepare_data(self):
        log.info("📊 Loading & preparing data (strategy-aware)...")
        mongodb_path, trade_stock_path = self._resolve_csv_paths()

        market_df = pd.read_csv(mongodb_path, parse_dates=["date"])
        trade_df = pd.read_csv(trade_stock_path, parse_dates=["date"])

        log.info("Market data shape: %s", market_df.shape)
        log.info("Trade data shape: %s", trade_df.shape)

        # --- Feature engineering (market) ---
        market_df["returns"] = market_df["close"].pct_change()
        market_df["volatility"] = market_df["returns"].rolling(20).std()
        market_df["volume_change"] = market_df["volume"].pct_change()

        # --- Strategy features ---
        trade_feats = trade_df[["symbol", "date", "buy", "SL", "tp", "RRR"]].copy()
        trade_feats["SL_pct"] = (trade_feats["SL"] - trade_feats["buy"]) / trade_feats["buy"]
        trade_feats["TP_pct"] = (trade_feats["tp"] - trade_feats["buy"]) / trade_feats["buy"]
        trade_feats = trade_feats[["symbol", "date", "RRR", "SL_pct", "TP_pct"]]

        # --- Merge ---
        merged = pd.merge(market_df, trade_feats, on=["symbol", "date"], how="left")
        merged["RRR"] = merged["RRR"].fillna(0.0)
        merged["SL_pct"] = merged["SL_pct"].fillna(0.0)
        merged["TP_pct"] = merged["TP_pct"].fillna(0.0)

        # --- Label (buy signal) ---
        merged["target"] = merged["buy"].notna().astype(int)
        log.info("Buy signals: %d / %d", merged["target"].sum(), len(merged))

        # --- Symbol encoding ---
        if "symbol" in merged.columns:
            merged["symbol_encoded"] = self.label_encoder.fit_transform(merged["symbol"])
        # --- Drop NA ---
        merged = merged.dropna(subset=self.feature_names + ["target"])
        log.info("Final samples: %d", len(merged))
        return merged[self.feature_names], merged["target"]

    # ------------------------------------------------------------------
    # 3. Train
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
    # 4. Save
    # ------------------------------------------------------------------
    def save_model(self):
        if self.model is None:
            log.warning("No model to save")
            return

        script_dir = Path(__file__).resolve().parent
        models_dir = script_dir / ".." / "csv" / "models"
        models_dir.mkdir(exist_ok=True)

        save_path = models_dir / "xgboost_strategy_model.pkl"
        joblib.dump(
            {
                "model": self.model,
                "label_encoder": self.label_encoder,
                "feature_names": self.feature_names,
            },
            save_path,
        )
        log.info("✅ Strategy-aware model saved → %s", save_path)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("XGBoost Trading Model – Strategy-Aware Training")
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
