# src/portfolio_allocator.py
import math
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Helper: resolve csv path relative to this file
# ------------------------------------------------------------------------------
CSV_FILE = Path(__file__).resolve().parent.parent / "csv" / "mongodb.csv"


# ==============================================================================
# Portfolio Allocator
# ==============================================================================
class PortfolioAllocator:
    """
    Mean-Variance (Markowitz) based allocator with signal-weight overlay.
    Supports long-only, fully-invested portfolio.
    """

    def __init__(self, risk_aversion: float = 1.0):
        """
        risk_aversion = 0  → max return
        risk_aversion → ∞  → min variance
        """
        self.risk_aversion = risk_aversion

    # --------------------------------------------------------------------------
    # 1. Core allocation
    # --------------------------------------------------------------------------
    def calculate_allocation(
        self,
        market_data: pd.DataFrame,
        trade_signals: dict[str, float],
        account_balance: float,
        min_weight: float = 0.01,  # 1%
    ) -> dict[str, dict]:
        """
        Returns: dict[symbol, {weight, shares, notional, price}]
        """
        if account_balance <= 0:
            log.warning("Account balance <= 0 → no allocation")
            return {}

        symbols = list(trade_signals.keys())
        if not symbols:
            return {}

        # Build returns matrix
        returns_df = self._build_returns(market_data, symbols)
        if returns_df.empty or returns_df.isna().all().all():
            log.warning("Insufficient price data → equal-weight fallback")
            weights = np.array([1.0 / len(symbols)] * len(symbols))
        else:
            try:
                weights = self._markowitz_weights(returns_df)
            except Exception as e:
                log.exception("Markowitz failed → equal-weight fallback")
                weights = np.array([1.0 / len(symbols)] * len(symbols))

        # Apply signal strength
        signal_arr = np.array([trade_signals[s] for s in symbols])
        weights *= signal_arr
        weights = np.where(weights < 0, 0, weights)  # long only

        # Normalize
        total = weights.sum()
        if total <= 0:
            log.warning("All signal weights zero → no allocation")
            return {}
        weights /= total

        # Create allocation dict
        allocations = {}
        cash_left = account_balance
        for sym, w in zip(symbols, weights):
            if w < min_weight:
                continue
            price = self._current_price(market_data, sym)
            if price <= 0:
                log.warning("Price for %s is <= 0 → skipped", sym)
                continue

            notional = account_balance * w
            # respect cash left
            notional = min(notional, cash_left)
            shares = math.floor(notional / price)
            if shares == 0:
                continue

            notional = shares * price  # adjust for integer shares
            cash_left -= notional

            allocations[sym] = {
                "weight": notional / account_balance,
                "shares": shares,
                "notional": notional,
                "price": price,
            }

        return allocations

    # --------------------------------------------------------------------------
    # 2. Rebalance
    # --------------------------------------------------------------------------
    def rebalance(
        self,
        current_portfolio: dict[str, dict],
        target_allocations: dict[str, dict],
        transaction_cost: float = 0.001,
    ) -> list[dict]:
        """
        Returns list of orders: [{'symbol': 'X', 'side': 'BUY'|'SELL', 'shares': int, 'value': float}]
        """
        orders = []

        if not target_allocations:
            return orders

        # total current value
        total_value = sum(
            pos.get("current_value", pos.get("value", 0))
            for pos in current_portfolio.values()
        )
        if total_value <= 0:
            log.warning("Current portfolio value is zero → full rebalance as fresh")
            total_value = sum(t["notional"] for t in target_allocations.values())

        # target weights
        tgt_weights = {s: t["weight"] for s, t in target_allocations.items()}
        tgt_prices = {s: t["price"] for s, t in target_allocations.items()}

        # current weights
        cur_weights = {}
        for sym, pos in current_portfolio.items():
            val = pos.get("current_value", pos.get("value", 0))
            cur_weights[sym] = val / total_value if total_value else 0

        # symbols union
        symbols = set(cur_weights.keys()) | set(tgt_weights.keys())

        for sym in symbols:
            tgt_w = tgt_weights.get(sym, 0.0)
            tgt_val = total_value * tgt_w
            cur_val = total_value * cur_weights.get(sym, 0.0)
            diff_val = tgt_val - cur_val

            # ignore tiny diff
            if abs(diff_val) < total_value * 0.01:
                continue

            side = "BUY" if diff_val > 0 else "SELL"
            price = tgt_prices.get(sym, 0.0)
            if price <= 0:
                log.warning("Price missing for %s → skipped", sym)
                continue

            # apply transaction cost
            adj_value = abs(diff_val) * (1 - transaction_cost)
            shares = math.floor(adj_value / price)
            if shares == 0:
                continue

            orders.append(
                {
                    "symbol": sym,
                    "side": side,
                    "shares": shares,
                    "value": shares * price,
                    "price": price,
                }
            )

        return orders

    # --------------------------------------------------------------------------
    # 3. Private helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _build_returns(market_data: pd.DataFrame, symbols: list) -> pd.DataFrame:
        """Return DataFrame of daily returns for symbols."""
        returns = {}
        for sym in symbols:
            df = market_data[market_data["symbol"] == sym].copy()
            if len(df) < 2:
                continue
            df["return"] = df["close"].pct_change()
            returns[sym] = df["return"].dropna()
        return pd.DataFrame(returns)

    @staticmethod
    def _current_price(market_data: pd.DataFrame, symbol: str) -> float:
        """Latest close price for symbol."""
        df = market_data[market_data["symbol"] == symbol]
        if df.empty:
            return 0.0
        return float(df["close"].iloc[-1])

    # --------------------------------------------------------------------------
    # 4. Markowitz optimizer (long-only, fully invested)
    # --------------------------------------------------------------------------
    def _markowitz_weights(self, returns_df: pd.DataFrame) -> np.ndarray:
        mu = returns_df.mean().values
        cov = returns_df.cov().values
        n = len(mu)

        # objective: negative utility = -(mu'w - 0.5 * gamma * w'Cov w)
        def objective(w):
            return -(w @ mu - 0.5 * self.risk_aversion * w @ cov @ w)

        # constraints: sum(w) = 1, w >= 0
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n
        x0 = np.ones(n) / n

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-9, "disp": False},
        )
        if not result.success:
            log.warning("Optimization failed: %s → fallback equal weight", result.message)
            return x0
        return result.x


# ==============================================================================
# Quick test / pytest entry
# ==============================================================================
if __name__ == "__main__":
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Sample CSV not found at {CSV_FILE}")

    market_data = pd.read_csv(CSV_FILE)
    allocator = PortfolioAllocator(risk_aversion=1.0)

    # dummy signals
    signals = {"GSPFINANCE": 0.8, "ACI": 0.6}
    allocs = allocator.calculate_allocation(market_data, signals, account_balance=100_000)

    print("Allocations:")
    for sym, info in allocs.items():
        print(f"{sym}: weight={info['weight']:.2%}  shares={info['shares']}  notional={info['notional']:.2f}")

    # rebalance example
    current = {
        "GSPFINANCE": {"current_value": 40_000},
        "ACI": {"current_value": 20_000},
    }
    targets = {s: {"weight": allocs[s]["weight"], "price": allocs[s]["price"]} for s in allocs}
    orders = allocator.rebalance(current, targets, transaction_cost=0.001)
    print("\nRebalance orders:")
    for o in orders:
        print(o)
