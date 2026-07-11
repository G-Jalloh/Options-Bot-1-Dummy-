# ═══════════════════════════════════════════════════════════
# Agent Squad — Python dependencies
# Install with:  pip install -r requirements.txt
# ═══════════════════════════════════════════════════════════

# Async SQLite — reads options_flow.db without blocking the event loop
aiosqlite>=0.19.0

# HTTP client — Webull REST auth & quote endpoints
aiohttp>=3.9.0

# MQTT — Webull real-time tick streaming
paho-mqtt>=1.6.1

# Environment file loader
python-dotenv>=1.0.0

# Numerical / data
numpy>=1.26.0
pandas>=2.1.0

# Webull official SDK (optional — provides MdClient for MQTT auth)
# Uncomment if you want to use the official SDK token helper.
# webull>=0.3.14

# ── Dev / test extras (optional) ──────────────────────────
# pytest>=7.4.0
# pytest-asyncio>=0.23.0
