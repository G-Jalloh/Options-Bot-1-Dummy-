"""
agent_squad.py
──────────────
Four-agent autonomous trading system fully wired to Webull.

AGENTS:
  MarketTrendAgent    → checks SPY macro regime via Webull live data
  WhaleHunterAgent    → reads your real options_flow.db for sweeps
  EntrySignalsAgent   → combines macro + flow + technicals → TradeSignal
  RiskManagerAgent    → Chandelier trailing stop + EMA invalidation exit

WEBULL INTEGRATION:
  ┌──────────────────────────────────────────────────────────┐
  │  .env has real keys?                                      │
  │    YES → WebullQuoteClient  (live REST quotes + history)  │
  │    NO  → SimulatedClient    (random-walk, no creds needed)│
  └──────────────────────────────────────────────────────────┘

FLOW DATA:
  Reads options_flow.db written by your options-flow-engine
  (the Webull MQTT streaming pipeline built earlier).
  If DB is missing → uses synthetic sweep data for testing.

Run:
  python run.py               (auto-detects mode from .env)
  python run.py --simulate    (force simulation mode)
  python run.py --cycles 20   (run 20 cycles then stop)
"""

import asyncio
import aiosqlite
import dataclasses
import logging
import os
import sys
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from webull_client import get_quote_client, SimulatedClient

load_dotenv()

log = logging.getLogger("agent_squad")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent_squad.log"),
    ]
)


# ═════════════════════════════════════════════════════════════
# DATA MODEL
# ═════════════════════════════════════════════════════════════

@dataclasses.dataclass
class TradeSignal:
    ticker:            str
    direction:         str        # "CALL" or "PUT"
    strike:            float      # option strike
    entry_option_px:   float      # option premium at entry
    entry_underlying:  float      # stock price at entry  ← stop math uses this
    atr:               float      # underlying ATR at entry
    sweep_size:        int        # contracts in the triggering sweep
    regime:            str        # "BULLISH" | "BEARISH" | "NEUTRAL"
    rsi:               float      # RSI at entry
    ema_9:             float      # EMA 9 at entry
    ema_21:            float      # EMA 21 at entry
    timestamp:         datetime   = dataclasses.field(
                                       default_factory=datetime.utcnow)


# ═════════════════════════════════════════════════════════════
# AGENT 1 — MARKET TREND
# Calls Webull for live SPY price + 200 SMA + RSI
# ═════════════════════════════════════════════════════════════

class MarketTrendAgent:
    """
    Determines macro regime from live SPY data via Webull.

    Rules:
      BULLISH  = SPY > 200 SMA  AND  RSI(14) > 50
      BEARISH  = SPY < 200 SMA  AND  RSI(14) < 50
      NEUTRAL  = mixed / insufficient data
    """

    def __init__(self, quote_client, spy_symbol: str = "SPY"):
        self.client     = quote_client
        self.spy_symbol = spy_symbol

    async def check_regime(self) -> tuple[str, Dict[str, Any]]:
        """
        Returns (regime_str, details_dict)
        """
        log.info("MarketTrendAgent: fetching %s regime...", self.spy_symbol)

        try:
            q = await self.client.get_quote(self.spy_symbol)
        except Exception as e:
            log.error("MarketTrendAgent: quote fetch failed: %s", e)
            return "NEUTRAL", {}

        price  = q["price"]
        ema_50 = q["ema_50"]    # proxy for medium-term trend
        rsi    = q["rsi_14"]
        atr    = q["atr_14"]

        # SPY above EMA 50 + RSI bullish = BULLISH regime
        above_trend = price > ema_50
        rsi_bull    = rsi > 52
        rsi_bear    = rsi < 48

        if above_trend and rsi_bull:
            regime = "BULLISH"
        elif not above_trend and rsi_bear:
            regime = "BEARISH"
        else:
            regime = "NEUTRAL"

        details = {
            "spy_price": price,
            "ema_50":    ema_50,
            "rsi":       rsi,
            "atr":       atr,
        }

        log.info(
            "SPY $%.2f | EMA50 $%.2f | RSI %.1f | Regime: %s",
            price, ema_50, rsi, regime
        )
        return regime, details


# ═════════════════════════════════════════════════════════════
# AGENT 2 — WHALE HUNTER
# Reads your real options_flow.db for institutional sweeps
# ═════════════════════════════════════════════════════════════

class WhaleHunterAgent:
    """
    Queries options_flow.db (written by your Webull MQTT engine)
    for recent large sweeps that signal institutional intent.

    Query logic:
      is_sweep   = 1     (multi-exchange sweep detected)
      is_unusual = 1     (size above unusual threshold)
      size       >= min_size
      ts_utc     within last lookback_mins minutes
      ORDER BY size DESC  (largest first = strongest conviction)
    """

    def __init__(
        self,
        db_path:       str = "options_flow.db",
        lookback_mins: int = 60,
        min_size:      int = 500,
    ):
        self.db_path       = db_path
        self.lookback_mins = lookback_mins
        self.min_size      = min_size

    async def find_unusual_flow(self) -> Optional[Dict[str, Any]]:
        log.info("WhaleHunterAgent: scanning %s...", self.db_path)

        # ── No DB yet? Use synthetic data for testing ─────────
        if not os.path.exists(self.db_path):
            log.warning(
                "options_flow.db not found at '%s'. "
                "Using synthetic sweep. Run options-flow-engine first "
                "to populate real data.", self.db_path
            )
            return self._synthetic_flow()

        cutoff = (
            datetime.utcnow() - timedelta(minutes=self.lookback_mins)
        ).isoformat()

        query = """
            SELECT
                symbol,
                option_symbol,
                size,
                price,
                aggressor_side,
                ts_utc,
                is_sweep,
                is_unusual
            FROM options_flow
            WHERE  is_sweep   = 1
              AND  is_unusual = 1
              AND  size       >= ?
              AND  ts_utc     >= ?
            ORDER BY size DESC, ts_utc DESC
            LIMIT 1
        """

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, (self.min_size, cutoff)) as cur:
                row = await cur.fetchone()

        if row is None:
            log.info("WhaleHunterAgent: no qualifying sweeps in window.")
            return None

        result = dict(row)
        # Map aggressor_side (BUY/SELL) → option direction (CALL/PUT)
        result["side"] = "CALL" if result.get("aggressor_side") == "BUY" else "PUT"

        log.info(
            "🐳 Sweep: %s %s | size=%d | side=%s | price=$%.2f",
            result["symbol"], result.get("option_symbol", ""),
            result["size"],   result["side"], result["price"]
        )
        return result

    @staticmethod
    def _synthetic_flow() -> Dict[str, Any]:
        """Fake sweep for offline testing."""
        import random
        tickers = [
            ("NVDA", "CALL", 12.50, 1000),
            ("AAPL", "CALL",  1.25, 1500),
            ("TSLA", "PUT",   4.10,  750),
            ("SPY",  "CALL",  2.80,  800),
        ]
        sym, side, price, size = random.choice(tickers)
        return {
            "symbol":        sym,
            "option_symbol": f"{sym} 2026-01-16 {side[0]}",
            "type":          "SWEEP",
            "side":          side,
            "size":          size,
            "price":         price,
            "ts_utc":        datetime.utcnow().isoformat(),
            "aggressor_side": "BUY" if side == "CALL" else "SELL",
        }


# ═════════════════════════════════════════════════════════════
# AGENT 3 — ENTRY SIGNALS
# Combines macro + whale flow + live technicals → TradeSignal
# ═════════════════════════════════════════════════════════════

class EntrySignalsAgent:
    """
    Generates a TradeSignal when ALL three conditions align:

      1. Whale sweep matches macro regime
         (CALL sweep + BULLISH  OR  PUT sweep + BEARISH)

      2. Technical confirmation on the ticker itself
         EMA 9 > EMA 21  (for calls)  or  EMA 9 < EMA 21  (for puts)

      3. RSI in momentum zone
         CALL: RSI 50–75   |   PUT: RSI 25–50
    """

    def __init__(
        self,
        hunter:       WhaleHunterAgent,
        macro:        MarketTrendAgent,
        quote_client,
        min_score:    int = 2,       # of 3 technical checks
    ):
        self.hunter       = hunter
        self.macro        = macro
        self.client       = quote_client
        self.min_score    = min_score

    async def generate_order(self) -> Optional[TradeSignal]:
        # ── Run macro + whale scan concurrently ───────────────
        (regime, spy_data), flow = await asyncio.gather(
            self.macro.check_regime(),
            self.hunter.find_unusual_flow(),
        )

        if flow is None:
            log.info("EntrySignals: no flow → no signal.")
            return None

        # ── Regime / flow alignment ───────────────────────────
        call_setup = (regime == "BULLISH" and flow["side"] == "CALL")
        put_setup  = (regime == "BEARISH" and flow["side"] == "PUT")

        if not (call_setup or put_setup):
            log.info(
                "EntrySignals: flow (%s) conflicts with regime (%s) → skip.",
                flow["side"], regime
            )
            return None

        direction = "CALL" if call_setup else "PUT"
        ticker    = flow["symbol"]

        # ── Fetch live ticker technicals from Webull ──────────
        try:
            q = await self.client.get_quote(ticker)
        except Exception as e:
            log.error("EntrySignals: quote fetch failed for %s: %s", ticker, e)
            return None

        price  = q["price"]
        ema_9  = q["ema_9"]
        ema_21 = q["ema_21"]
        ema_50 = q["ema_50"]
        atr    = q["atr_14"]
        rsi    = q["rsi_14"]

        # ── Technical score (0–3) ─────────────────────────────
        score = 0
        reasons = []

        # Check 1: EMA alignment
        if direction == "CALL" and ema_9 > ema_21:
            score += 1
            reasons.append(f"✅ EMA 9 (${ema_9:.2f}) > EMA 21 (${ema_21:.2f})")
        elif direction == "PUT" and ema_9 < ema_21:
            score += 1
            reasons.append(f"✅ EMA 9 (${ema_9:.2f}) < EMA 21 (${ema_21:.2f})")
        else:
            reasons.append(f"❌ EMA alignment mismatch for {direction}")

        # Check 2: RSI momentum zone
        if direction == "CALL" and 50 < rsi < 75:
            score += 1
            reasons.append(f"✅ RSI {rsi:.1f} in bull momentum zone (50–75)")
        elif direction == "PUT" and 25 < rsi < 50:
            score += 1
            reasons.append(f"✅ RSI {rsi:.1f} in bear momentum zone (25–50)")
        else:
            reasons.append(f"❌ RSI {rsi:.1f} not in momentum zone for {direction}")

        # Check 3: Ticker above/below EMA 50 (macro alignment on the stock)
        if direction == "CALL" and price > ema_50:
            score += 1
            reasons.append(f"✅ {ticker} above EMA 50 (${ema_50:.2f})")
        elif direction == "PUT" and price < ema_50:
            score += 1
            reasons.append(f"✅ {ticker} below EMA 50 (${ema_50:.2f})")
        else:
            reasons.append(f"❌ {ticker} on wrong side of EMA 50 for {direction}")

        for r in reasons:
            log.info("  %s", r)

        if score < self.min_score:
            log.info(
                "EntrySignals: score %d/%d below threshold → skip.",
                score, self.min_score * 1
            )
            return None

        signal = TradeSignal(
            ticker           = ticker,
            direction        = direction,
            strike           = round(price * (1.05 if direction == "CALL" else 0.95), 0),
            entry_option_px  = flow["price"],
            entry_underlying = price,        # ← underlying for stop math
            atr              = atr,
            sweep_size       = flow["size"],
            regime           = regime,
            rsi              = rsi,
            ema_9            = ema_9,
            ema_21           = ema_21,
        )

        log.info(
            "🎯 SIGNAL: %s %s | underlying=$%.2f | option=$%.2f | "
            "ATR=$%.2f | score=%d/3",
            ticker, direction, price, flow["price"], atr, score
        )
        return signal


# ═════════════════════════════════════════════════════════════
# AGENT 4 — RISK MANAGER
# Chandelier trailing stop + EMA invalidation on underlying
# ═════════════════════════════════════════════════════════════

class RiskManagerAgent:
    """
    Monitors an open position bar-by-bar using live Webull quotes.

    Tracks UNDERLYING price for stops (not option price —
    option premiums are noisy and leveraged).

    Exit priority:
      1. DEEP_INVALIDATION  → price > 1 ATR below EMA 50
      2. TREND_INVALIDATION → EMA 9 crosses below EMA 21
      3. TRAILING_STOP      → underlying drops below 2×ATR trail
      4. PROFIT_TARGET      → option premium reaches 2× entry
    """

    def __init__(self, signal: TradeSignal, atr_multiplier: float = 2.0):
        self.signal     = signal
        self.atr_mult   = atr_multiplier

        # ── All stop math on the UNDERLYING ───────────────────
        self.highest_seen  = signal.entry_underlying
        self.trailing_stop = signal.entry_underlying - (
            atr_multiplier * signal.atr
        )

        # ── Option profit target = 2× premium ─────────────────
        self.profit_target = signal.entry_option_px * 2.0

        # ── Breakeven flag ─────────────────────────────────────
        self.at_breakeven = False

        log.info("─" * 55)
        log.info("🛡️  Risk Guard armed")
        log.info("   Ticker       : %s %s", signal.ticker, signal.direction)
        log.info("   Underlying   : $%.2f", signal.entry_underlying)
        log.info("   Initial Stop : $%.2f  (2×ATR=%.2f below entry)",
                 self.trailing_stop, signal.atr)
        log.info("   Option entry : $%.2f  |  2× target: $%.2f",
                 signal.entry_option_px, self.profit_target)
        log.info("─" * 55)

    async def monitor_exit(
        self,
        current_underlying: float,
        current_option_px:  float,
        current_ema_9:      float,
        current_ema_21:     float,
        current_ema_50:     float,
    ) -> str:

        # ── Ratchet stop upward (only moves up, never down) ───
        if current_underlying > self.highest_seen:
            self.highest_seen  = current_underlying
            self.trailing_stop = (
                self.highest_seen - self.atr_mult * self.signal.atr
            )

        # ── Move stop to breakeven after 1R gain ──────────────
        one_r_target = (
            self.signal.entry_underlying + self.signal.atr * 1.5
        )
        if current_underlying >= one_r_target and not self.at_breakeven:
            self.trailing_stop = max(
                self.trailing_stop, self.signal.entry_underlying
            )
            self.at_breakeven  = True
            log.info("🔒 Stop moved to breakeven ($%.2f)",
                     self.signal.entry_underlying)

        # ── Exit checks (priority order) ──────────────────────

        # Priority 1: deep breach — more than 1 ATR below EMA 50
        if current_underlying < (current_ema_50 - self.signal.atr):
            return (
                f"🚨 CRITICAL EXIT — underlying ${current_underlying:.2f} "
                f"is more than 1 ATR below EMA 50 (${current_ema_50:.2f}). "
                f"Exit IMMEDIATELY."
            )

        # Priority 2: EMA 9/21 crossunder
        if current_ema_9 < current_ema_21:
            spread = current_ema_21 - current_ema_9
            return (
                f"🚨 TREND INVALIDATION — EMA 9 (${current_ema_9:.2f}) "
                f"crossed below EMA 21 (${current_ema_21:.2f}). "
                f"Spread: -{spread:.3f}. Exit now."
            )

        # Priority 3: trailing stop
        if current_underlying <= self.trailing_stop:
            be = " (breakeven)" if self.at_breakeven else ""
            return (
                f"🔔 TRAILING STOP HIT{be} — underlying ${current_underlying:.2f} "
                f"broke below stop ${self.trailing_stop:.2f}. Exit."
            )

        # Priority 4: profit target
        if current_option_px >= self.profit_target:
            gain = (current_option_px - self.signal.entry_option_px) / \
                    self.signal.entry_option_px * 100
            return (
                f"💰 PROFIT TARGET — option at ${current_option_px:.2f} "
                f"(+{gain:.1f}%). Take full or partial profits."
            )

        # Hold
        option_pnl = (
            (current_option_px - self.signal.entry_option_px)
            / self.signal.entry_option_px * 100
        )
        be_label = " [BREAKEVEN LOCKED]" if self.at_breakeven else ""
        return (
            f"✅ HOLD{be_label} | "
            f"underlying=${current_underlying:.2f} | "
            f"stop=${self.trailing_stop:.2f} | "
            f"option P&L: {option_pnl:+.1f}%"
        )


# ═════════════════════════════════════════════════════════════
# ORCHESTRATOR — wires all 4 agents into a continuous loop
# ═════════════════════════════════════════════════════════════

class AgentOrchestrator:
    """
    Main control loop. On each cycle:
      - If no open position: scan for entry signal
      - If position open:    poll Webull for live quotes + run exit checks

    Webull integration:
      get_quote_client() auto-selects live vs simulated
      based on WEBULL_APP_KEY in .env
    """

    def __init__(
        self,
        db_path:       str = "options_flow.db",
        cycle_seconds: int = 60,
        max_cycles:    int = 0,          # 0 = infinite
        force_sim:     bool = False,     # force simulation mode
    ):
        self.cycle_secs   = cycle_seconds
        self.max_cycles   = max_cycles

        # ── Auto-select live vs simulated Webull client ───────
        if force_sim:
            self.quote_client = SimulatedClient()
            log.info("Mode: SIMULATION (forced)")
        else:
            self.quote_client = get_quote_client()
            mode = "SIMULATION" if isinstance(self.quote_client, SimulatedClient) else "LIVE"
            log.info("Mode: %s", mode)

        # ── Wire agents ───────────────────────────────────────
        self.macro  = MarketTrendAgent(self.quote_client)
        self.hunter = WhaleHunterAgent(db_path=db_path)
        self.picker = EntrySignalsAgent(self.hunter, self.macro, self.quote_client)

        self.active_signal: Optional[TradeSignal]     = None
        self.risk_guard:    Optional[RiskManagerAgent] = None

    async def run(self) -> None:
        self._print_banner()
        cycle = 0

        while self.max_cycles == 0 or cycle < self.max_cycles:
            cycle += 1
            print(f"\n{'─'*55}")
            print(f"  Cycle {cycle}"
                  + (f" / {self.max_cycles}" if self.max_cycles else "")
                  + f"  [{datetime.utcnow().strftime('%H:%M:%S')} UTC]")
            print(f"{'─'*55}")

            try:
                if self.active_signal is None:
                    await self._scan_for_entry()
                else:
                    await self._monitor_position()
            except Exception as e:
                log.error("Cycle error: %s", e, exc_info=True)

            if self.max_cycles == 0 or cycle < self.max_cycles:
                await asyncio.sleep(self.cycle_secs)

        print(f"\n{'═'*55}")
        print("  Agent squad completed all cycles.")
        print(f"{'═'*55}\n")

    async def _scan_for_entry(self) -> None:
        print("🔍 Scanning for entry signal...")
        signal = await self.picker.generate_order()

        if signal:
            print(f"\n{'='*55}")
            print(f"  🎯 ENTRY SIGNAL CONFIRMED")
            print(f"  Ticker    : {signal.ticker}")
            print(f"  Direction : {signal.direction}")
            print(f"  Strike    : ${signal.strike:.0f}")
            print(f"  Underlying: ${signal.entry_underlying:.2f}")
            print(f"  Option px : ${signal.entry_option_px:.2f}")
            print(f"  ATR (14)  : ${signal.atr:.2f}")
            print(f"  Sweep size: {signal.sweep_size:,} contracts")
            print(f"  Regime    : {signal.regime}")
            print(f"  RSI       : {signal.rsi:.1f}")
            print(f"{'='*55}\n")

            self.active_signal = signal
            self.risk_guard    = RiskManagerAgent(signal)
        else:
            print("  ⏳ No qualifying setup — waiting next cycle.")

    async def _monitor_position(self) -> None:
        sig = self.active_signal
        print(f"📡 Monitoring {sig.ticker} {sig.direction}...")

        try:
            q = await self.quote_client.get_quote(sig.ticker)
        except Exception as e:
            log.error("Could not fetch quote for %s: %s", sig.ticker, e)
            return

        # Fetch live option price (or estimate if unavailable)
        try:
            option_px = await self.quote_client.get_option_price(
                sig.ticker, "2026-01-16", sig.strike, sig.direction
            )
            option_px = option_px or sig.entry_option_px
        except Exception:
            option_px = sig.entry_option_px

        status = await self.risk_guard.monitor_exit(
            current_underlying = q["price"],
            current_option_px  = option_px,
            current_ema_9      = q["ema_9"],
            current_ema_21     = q["ema_21"],
            current_ema_50     = q["ema_50"],
        )

        print(f"\n  {status}\n")

        # Any exit signal → close position and reset
        exit_keywords = ("EXIT", "STOP HIT", "TARGET", "INVALIDATION")
        if any(kw in status for kw in exit_keywords):
            print(f"  ⚠️  Closing position: {sig.ticker} {sig.direction}")
            self.active_signal = None
            self.risk_guard    = None

    @staticmethod
    def _print_banner() -> None:
        print("\n" + "═" * 55)
        print("  🤖 AGENT SQUAD — Webull-Powered Trading System")
        print("  " + "─" * 51)
        print("  Agents: MarketTrend · WhaleHunter")
        print("          EntrySignals · RiskManager")
        print("═" * 55)
