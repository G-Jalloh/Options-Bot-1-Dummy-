"""
webull_client.py
────────────────
Webull OpenAPI client that provides:

  1. WebullQuoteClient   — live underlying quotes + computed EMAs
  2. WebullOptionClient  — live option premium lookup
  3. SimulatedClient     — drop-in fallback (no creds needed for testing)

MODE SELECTION (automatic):
  If WEBULL_APP_KEY / WEBULL_APP_SECRET are set in .env → uses Webull live data
  If keys are missing or say REPLACE_ME              → falls back to simulation

Webull API used:
  REST  → GET /quote/v1/ticker/getBatchStockQuotes  (live price)
  REST  → GET /quote/v1/option/getOptionQuote        (option premium)
  MQTT  → options tick stream (handled by options-flow-engine separately)

Docs: https://developer.webull.com/apis/docs/market-data-api/
"""

import os
import time
import logging
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("webull_client")


# ─────────────────────────────────────────────────────────────
# Webull REST endpoints
# ─────────────────────────────────────────────────────────────

ENDPOINTS = {
    "prod": "https://api.webullfintech.com",
    "test": "https://testapi.webullfintech.com",
}

AUTH_PATH    = "/oauth2/v1/token"
QUOTE_PATH   = "/quote/v1/ticker/getBatchStockQuotes"
OPTION_PATH  = "/quote/v1/option/getOptionQuote"
HIST_PATH    = "/quote/v1/ticker/getKlineHistory"


# ─────────────────────────────────────────────────────────────
# Auth token manager
# ─────────────────────────────────────────────────────────────

class _WebullAuth:
    """Fetches and auto-refreshes the short-lived REST access token."""

    def __init__(self, app_key: str, app_secret: str, base_url: str):
        self._app_key    = app_key
        self._app_secret = app_secret
        self._base_url   = base_url
        self._token:      Optional[str] = None
        self._expires_at: float         = 0.0

    async def get_token(self) -> str:
        if self._token and time.time() < self._expires_at - 60:
            return self._token
        await self._refresh()
        return self._token

    async def _refresh(self) -> None:
        url     = self._base_url + AUTH_PATH
        payload = {
            "appKey":    self._app_key,
            "appSecret": self._app_secret,
            "grantType": "client_credentials",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Webull auth failed ({resp.status}): {text}"
                    )
                data = await resp.json()

        # SDK response shape: { "accessToken": "...", "expiresIn": 3600 }
        self._token      = data["accessToken"]
        self._expires_at = time.time() + data.get("expiresIn", 3600)
        log.info("Webull token refreshed. Expires in %ds", data.get("expiresIn", 3600))


# ─────────────────────────────────────────────────────────────
# Quote client — underlying prices + EMA computation
# ─────────────────────────────────────────────────────────────

class WebullQuoteClient:
    """
    Fetches live stock quotes and computes EMA 9 / 21 / 50
    from recent historical bars.

    Usage:
        client = WebullQuoteClient.from_env()
        quote  = await client.get_quote("NVDA")
        # quote = {
        #   "symbol": "NVDA", "price": 138.50,
        #   "ema_9": 137.20, "ema_21": 135.80, "ema_50": 130.00,
        #   "atr_14": 3.25, "rsi_14": 61.2
        # }
    """

    def __init__(self, app_key: str, app_secret: str, env: str = "test"):
        self._base   = ENDPOINTS.get(env, ENDPOINTS["test"])
        self._auth   = _WebullAuth(app_key, app_secret, self._base)
        self._env    = env

    @classmethod
    def from_env(cls) -> "WebullQuoteClient":
        key    = os.getenv("WEBULL_APP_KEY",    "")
        secret = os.getenv("WEBULL_APP_SECRET", "")
        env    = os.getenv("WEBULL_ENV",        "test")
        if not key or key == "REPLACE_ME":
            raise ValueError(
                "WEBULL_APP_KEY not set in .env — "
                "use SimulatedClient for offline testing."
            )
        return cls(key, secret, env)

    async def _headers(self) -> Dict[str, str]:
        token = await self._auth.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }

    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Returns live price + computed technical levels for one symbol.
        """
        # Step 1: get live price
        price = await self._fetch_live_price(symbol)

        # Step 2: get recent bars for EMA/ATR/RSI computation
        bars  = await self._fetch_history(symbol, interval="m15", count=250)

        if bars is None or len(bars) < 50:
            log.warning("%s: insufficient history — returning price only", symbol)
            return {"symbol": symbol, "price": price,
                    "ema_9": price, "ema_21": price,
                    "ema_50": price, "atr_14": price * 0.02, "rsi_14": 50.0}

        # Add live price as the latest close
        bars.loc[bars.index[-1], "close"] = price

        # Compute indicators inline (avoid circular imports)
        c     = bars["close"]
        ema9  = c.ewm(span=9,   adjust=False).mean().iloc[-1]
        ema21 = c.ewm(span=21,  adjust=False).mean().iloc[-1]
        ema50 = c.ewm(span=50,  adjust=False).mean().iloc[-1]
        atr   = self._compute_atr(bars).iloc[-1]
        rsi   = self._compute_rsi(c).iloc[-1]

        return {
            "symbol":  symbol,
            "price":   round(price, 4),
            "ema_9":   round(ema9,  4),
            "ema_21":  round(ema21, 4),
            "ema_50":  round(ema50, 4),
            "atr_14":  round(atr,   4),
            "rsi_14":  round(rsi,   2),
        }

    async def get_quotes_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch multiple symbols concurrently."""
        results = await asyncio.gather(
            *[self.get_quote(s) for s in symbols],
            return_exceptions=True
        )
        return {
            sym: (res if not isinstance(res, Exception) else None)
            for sym, res in zip(symbols, results)
        }

    # ── Internal REST helpers ─────────────────────────────────

    async def _fetch_live_price(self, symbol: str) -> float:
        url     = self._base + QUOTE_PATH
        headers = await self._headers()
        params  = {"symbols": symbol, "regionId": "us"}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, params=params,
                timeout=aiohttp.ClientTimeout(total=8)
            ) as resp:
                data = await resp.json()

        # Navigate Webull response: data["data"][0]["close"]
        try:
            return float(data["data"][0]["close"])
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Could not parse price for {symbol}: {e} | {data}")

    async def _fetch_history(
        self, symbol: str, interval: str = "m15", count: int = 250
    ) -> Optional[pd.DataFrame]:
        """
        Fetches recent OHLCV bars.

        Webull kline intervals: m1, m5, m15, m30, h1, h2, h4, d1, w1
        """
        url     = self._base + HIST_PATH
        headers = await self._headers()
        params  = {
            "symbol":   symbol,
            "type":     interval,
            "count":    count,
            "regionId": "us",
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, params=params,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                data = await resp.json()

        bars = data.get("data", [])
        if not bars:
            return None

        df = pd.DataFrame(bars)
        df.rename(columns={
            "open": "open", "high": "high",
            "low":  "low",  "close": "close", "volume": "volume"
        }, inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["close"])

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"]  - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_g = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_l = loss.ewm(alpha=1/period, adjust=False).mean()
        rs    = avg_g / avg_l.replace(0, np.nan)
        return 100 - (100 / (1 + rs))


# ─────────────────────────────────────────────────────────────
# Option client — live option premium
# ─────────────────────────────────────────────────────────────

class WebullOptionClient:
    """
    Fetches a live option premium for a given contract.

    Usage:
        client = WebullOptionClient.from_env()
        px     = await client.get_option_price(
                     symbol="NVDA", expiry="2026-01-16",
                     strike=1000.0, option_type="CALL"
                 )
    """

    def __init__(self, app_key: str, app_secret: str, env: str = "test"):
        self._base = ENDPOINTS.get(env, ENDPOINTS["test"])
        self._auth = _WebullAuth(app_key, app_secret, self._base)

    @classmethod
    def from_env(cls) -> "WebullOptionClient":
        key    = os.getenv("WEBULL_APP_KEY",    "")
        secret = os.getenv("WEBULL_APP_SECRET", "")
        env    = os.getenv("WEBULL_ENV",        "test")
        if not key or key == "REPLACE_ME":
            raise ValueError("WEBULL_APP_KEY not set.")
        return cls(key, secret, env)

    async def get_option_price(
        self,
        symbol:      str,
        expiry:      str,     # "YYYY-MM-DD"
        strike:      float,
        option_type: str,     # "CALL" or "PUT"
    ) -> Optional[float]:
        """Returns the mid-price (bid+ask)/2 of the option contract."""
        url     = self._base + OPTION_PATH
        headers = await self._auth.get_token()
        headers = {"Authorization": f"Bearer {headers}", "Content-Type": "application/json"}

        params = {
            "symbol":      symbol,
            "expireDate":  expiry,
            "strikePrice": str(strike),
            "direction":   option_type.upper(),
            "regionId":    "us",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, params=params,
                    timeout=aiohttp.ClientTimeout(total=8)
                ) as resp:
                    data = await resp.json()

            bid = float(data["data"]["bid"])
            ask = float(data["data"]["ask"])
            return round((bid + ask) / 2, 4)

        except Exception as e:
            log.warning("Could not fetch option price for %s: %s", symbol, e)
            return None


# ─────────────────────────────────────────────────────────────
# Simulated client — no credentials needed
# ─────────────────────────────────────────────────────────────

class SimulatedClient:
    """
    Drop-in replacement for WebullQuoteClient when credentials
    are not yet available.

    Generates realistic random-walk price data so you can test
    the entire agent loop without a Webull subscription.

    Automatically used when WEBULL_APP_KEY = REPLACE_ME or missing.
    """

    def __init__(self, seed: int = 42):
        self._rng    = np.random.default_rng(seed)
        self._prices: Dict[str, float] = {}

    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        # Start each symbol at a sensible price
        defaults = {"SPY": 510.0, "NVDA": 138.0, "AAPL": 195.0,
                    "TSLA": 250.0, "QQQ": 440.0}
        if symbol not in self._prices:
            self._prices[symbol] = defaults.get(symbol, 100.0)

        # Small random walk on each call
        self._prices[symbol] *= (1 + self._rng.normal(0, 0.003))
        price = round(self._prices[symbol], 4)

        # Simulated EMA levels around the price
        ema9  = round(price * (1 + self._rng.uniform(-0.003,  0.003)), 4)
        ema21 = round(price * (1 + self._rng.uniform(-0.006,  0.002)), 4)
        ema50 = round(price * (1 + self._rng.uniform(-0.015, -0.002)), 4)
        atr   = round(price * self._rng.uniform(0.012, 0.025), 4)
        rsi   = round(self._rng.uniform(45, 72), 2)

        return {
            "symbol": symbol, "price": price,
            "ema_9":  ema9,   "ema_21": ema21,
            "ema_50": ema50,  "atr_14": atr, "rsi_14": rsi,
        }

    async def get_option_price(self, symbol, expiry, strike, option_type) -> float:
        """Returns a simulated option premium."""
        base = self._prices.get(symbol, 100.0)
        moneyness = abs(base - strike) / base
        premium   = max(0.05, base * 0.03 * (1 - moneyness))
        return round(premium * (1 + self._rng.normal(0, 0.05)), 2)

    async def get_quotes_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        results = await asyncio.gather(*[self.get_quote(s) for s in symbols])
        return {sym: res for sym, res in zip(symbols, results)}


# ─────────────────────────────────────────────────────────────
# Auto-factory: returns live or simulated based on .env
# ─────────────────────────────────────────────────────────────

def get_quote_client():
    """
    Returns WebullQuoteClient if credentials are configured,
    otherwise returns SimulatedClient automatically.

    No manual switching needed — just fill in .env when ready.
    """
    key = os.getenv("WEBULL_APP_KEY", "")
    if not key or key == "REPLACE_ME":
        log.warning(
            "WEBULL_APP_KEY not set → using SimulatedClient (offline mode). "
            "Fill in .env with real credentials to switch to live data."
        )
        return SimulatedClient()

    env = os.getenv("WEBULL_ENV", "test")
    log.info("Webull credentials found → using WebullQuoteClient (env=%s)", env)
    return WebullQuoteClient.from_env()
