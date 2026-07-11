"""
config.py
─────────
Central configuration for the agent squad.
All values can be overridden via environment variables in .env
"""

import os
import dataclasses
from dotenv import load_dotenv

load_dotenv()


@dataclasses.dataclass
class AgentConfig:
    # ── Webull credentials (loaded from .env) ─────────────────
    app_key:    str = dataclasses.field(
                    default_factory=lambda: os.getenv("WEBULL_APP_KEY", "REPLACE_ME"))
    app_secret: str = dataclasses.field(
                    default_factory=lambda: os.getenv("WEBULL_APP_SECRET", "REPLACE_ME"))
    webull_env: str = dataclasses.field(
                    default_factory=lambda: os.getenv("WEBULL_ENV", "test"))

    # ── Database ───────────────────────────────────────────────
    db_path: str = dataclasses.field(
                    default_factory=lambda: os.getenv("DB_PATH", "options_flow.db"))

    # ── Scanning cadence ──────────────────────────────────────
    cycle_seconds: int = dataclasses.field(
                    default_factory=lambda: int(os.getenv("CYCLE_SECONDS", "60")))

    # ── Flow detection thresholds ─────────────────────────────
    min_sweep_size:  int   = dataclasses.field(
                    default_factory=lambda: int(os.getenv("MIN_SWEEP_SIZE", "500")))
    flow_lookback:   int   = dataclasses.field(
                    default_factory=lambda: int(os.getenv("FLOW_LOOKBACK_MINS", "60")))

    # ── Entry quality filter ──────────────────────────────────
    min_tech_score: int = dataclasses.field(
                    default_factory=lambda: int(os.getenv("MIN_TECH_SCORE", "2")))

    # ── Position sizing (used by RiskManagerAgent) ────────────
    account_size: float = dataclasses.field(
                    default_factory=lambda: float(os.getenv("ACCOUNT_SIZE", "10000")))
    risk_pct:     float = dataclasses.field(
                    default_factory=lambda: float(os.getenv("RISK_PCT", "0.01")))

    # ── Exit parameters ───────────────────────────────────────
    chandelier_mult: float = dataclasses.field(
                    default_factory=lambda: float(os.getenv("CHANDELIER_MULT", "2.0")))

    def __post_init__(self):
        if self.account_size <= 0:
            raise ValueError("ACCOUNT_SIZE must be positive")
        if not (0 < self.risk_pct < 1):
            raise ValueError("RISK_PCT must be between 0 and 1")
        if self.chandelier_mult <= 0:
            raise ValueError("CHANDELIER_MULT must be positive")
        if self.cycle_seconds < 5:
            raise ValueError("CYCLE_SECONDS must be at least 5")

    @property
    def is_live(self) -> bool:
        """True if real Webull credentials are present."""
        return (
            self.app_key not in ("REPLACE_ME", "", None)
            and self.app_secret not in ("REPLACE_ME", "", None)
        )

    def summary(self) -> str:
        mode = "LIVE 🟢" if self.is_live else "SIMULATION 🟡"
        lines = [
            f"  Mode          : {mode}",
            f"  Webull env    : {self.webull_env}",
            f"  DB path       : {self.db_path}",
            f"  Cycle         : {self.cycle_seconds}s",
            f"  Min sweep     : {self.min_sweep_size:,} contracts",
            f"  Flow window   : {self.flow_lookback} min",
            f"  Tech score    : {self.min_tech_score}/3",
            f"  Account size  : ${self.account_size:,.0f}",
            f"  Risk per trade: {self.risk_pct*100:.1f}%",
            f"  Chandelier    : {self.chandelier_mult}× ATR",
        ]
        return "\n".join(lines)


DEFAULT_CONFIG = AgentConfig()
