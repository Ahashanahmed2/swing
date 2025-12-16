# src/risk_engine.py
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import logging

# --------------- path resolver ---------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# config fallback
try:
    from config import MAX_POSITION_SIZE, MAX_PORTFOLIO_RISK, STOP_LOSS_PCT
except ImportError:  # config.py missing
    MAX_POSITION_SIZE = 0.02
    MAX_PORTFOLIO_RISK = 0.01
    STOP_LOSS_PCT = 0.02

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)
# ---------------------------------------------


class RiskEngine:
    """
    Responsible for:
    - position sizing (Kelly / fixed-fraction)
    - risk-reward check
    - portfolio-level limits (concentration, drawdown)
    - per-trade stop-loss / take-profit
    """

    def __init__(
        self,
        max_position_size: float = MAX_POSITION_SIZE,
        max_portfolio_risk: float = MAX_PORTFOLIO_RISK,
        stop_loss_pct: float = STOP_LOSS_PCT,
        max_drawdown_limit: float = 0.15,
        min_risk_reward: float = 1.0,
    ):
        self.max_pos_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_pct = stop_loss_pct
        self.max_dd_limit = max_drawdown_limit
        self.min_rr = min_risk_reward
        self.positions: dict[str, dict] = {}  # symbol -> info

    # ------------------------------------------------------------------
    # 1. Position sizing (fixed-fractional)
    # ------------------------------------------------------------------
    def calc_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> int:
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0:
            log.warning("Risk per share is zero → position size 0")
            return 0

        max_risk_amount = account_balance * self.max_portfolio_risk
        shares = max_risk_amount / risk_per_share

        # cap by max position size (notional)
        max_notional = account_balance * self.max_pos_size
        shares = min(shares, max_notional / entry_price)

        return int(shares)

    # ------------------------------------------------------------------
    # 2. Single-trade validation
    # ------------------------------------------------------------------
    def validate_trade(
        self,
        trade_data: dict,
        account_balance: float,
        portfolio_value: float,
    ) -> tuple[bool, str | dict]:
        """
        trade_data = {
            'symbol': str,
            'entry'/'buy': float,
            'SL'?: float,        # optional
            'tp'?: float,        # optional
        }
        Returns (is_valid, reason_string | dict_with_details)
        """
        try:
            symbol = trade_data["symbol"]
            entry = float(trade_data["buy"] if "buy" in trade_data else trade_data["entry"])
        except KeyError as e:
            return False, f"Missing key: {e}"

        stop = trade_data.get("SL", entry * (1 - self.stop_loss_pct))
        take = trade_data.get("tp", entry * (1 + 2 * self.stop_loss_pct))

        risk = abs(entry - stop)
        reward = abs(take - entry)

        if risk == 0:
            return False, "Zero risk (entry == stop-loss)"

        rr = reward / risk
        if rr < self.min_rr:
            return False, f"Risk-reward {rr:.2f} < min {self.min_rr}"

        shares = self.calc_position_size(account_balance, entry, stop)
        if shares == 0:
            return False, "Calculated position size is zero"

        notional = shares * entry

        # portfolio concentration
        current_exposure = sum(pos["value"] for pos in self.positions.values())
        max_exposure = portfolio_value * 0.30
        if current_exposure + notional > max_exposure:
            return False, f"Portfolio exposure limit exceeded: {current_exposure + notional:.2f} > {max_exposure:.2f}"

        return True, {
            "shares": shares,
            "notional": notional,
            "risk_amount": shares * risk,
            "risk_reward": rr,
            "stop_loss": stop,
            "take_profit": take,
        }

    # ------------------------------------------------------------------
    # 3. Portfolio-level risk metrics
    # ------------------------------------------------------------------
    def portfolio_metrics(self, portfolio: dict, market_df: pd.DataFrame) -> dict:
        """
        portfolio = {
            'SYMBOL': {'shares': int, 'entry': float, ...},
            ...
        }
        market_df must have: ['date', 'symbol', 'close']
        """
        if not portfolio:
            return {}

        # last 30 trading days
        dates = sorted(market_df["date"].unique())[-30:]
        pv_series = []  # portfolio value per day

        for d in dates:
            day_df = market_df[market_df["date"] == d]
            if day_df.empty:
                continue
            day_val = 0.0
            for sym, pos in portfolio.items():
                sym_day = day_df[day_df["symbol"] == sym]
                if sym_day.empty:
                    continue
                day_val += pos["shares"] * sym_day["close"].iloc[0]
            pv_series.append(day_val)

        if len(pv_series) < 2:
            return {"volatility": 0, "sharpe": 0, "max_drawdown": 0, "var_95": 0}

        returns = np.diff(pv_series) / np.array(pv_series[:-1])
        vol = np.std(returns, ddof=1) * np.sqrt(252)
        sharpe = (np.mean(returns) * 252) / (vol + 1e-8)
        var_95 = np.percentile(returns, 5) * pv_series[-1]
        max_dd = self._max_drawdown(pv_series)

        return {
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "var_95": var_95,
        }

    # ------------------------------------------------------------------
    # 4. Drawdown
    # ------------------------------------------------------------------
    @staticmethod
    def _max_drawdown(values: list | np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        peak = np.fmax.accumulate(values)
        dd = (values - peak) / peak
        return float(np.min(dd))

    # ------------------------------------------------------------------
    # 5. Update positions
    # ------------------------------------------------------------------
    def update_position(self, symbol: str, info: dict):
        self.positions[symbol] = info

    def remove_position(self, symbol: str):
        self.positions.pop(symbol, None)

    def reset(self):
        self.positions.clear()


# ============================================================================
# Quick self-test / pytest compatible
# ============================================================================
if __name__ == "__main__":
    engine = RiskEngine()

    trade = {"symbol": "GSPFINANCE", "buy": 1.8, "SL": 1.4, "tp": 2.3}
    ok, res = engine.validate_trade(trade, account_balance=100_000, portfolio_value=150_000)
    print("Trade valid:", ok)
    if ok:
        print("Details:", res)
        engine.update_position(trade["symbol"], res)

    # dummy market data for portfolio metrics
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    market_df = pd.DataFrame({
        "date": dates.repeat(2),
        "symbol": ["GSPFINANCE", "ACI"] * 30,
        "close": np.random.uniform(1.5, 2.5, 60),
    })
    portfolio = {
        "GSPFINANCE": {"shares": res["shares"]},
        "ACI": {"shares": 500},
    }
    print("Portfolio metrics:", engine.portfolio_metrics(portfolio, market_df))
