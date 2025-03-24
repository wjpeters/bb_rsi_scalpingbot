"""
Configuration file for the Bitcoin RSI+Bollinger Bands Scalping Bot
"""

# Trading parameters
SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
POSITION_SIZE = 0.005  # BTC amount to trade (5x)
PROFIT_TARGET = 20     # Daily profit target in USD

# Compounding settings
ENABLE_COMPOUNDING = False  # Whether to enable position size compounding
INITIAL_CAPITAL = 1000     # Initial capital in USD for compounding calculations
COMPOUND_INTERVAL = "DAILY" # When to compound: DAILY, WEEKLY, or MONTHLY
MAX_POSITION_SIZE = 0.02   # Maximum position size in BTC (for safety)
COMPOUND_RATE = 0.5        # Percentage of profits to compound (0.5 = 50%)

# Strategy parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 58  # Even more aggressive overbought level
RSI_OVERSOLD = 42    # Even more aggressive oversold level
BB_PERIOD = 20
BB_STD = 1.3        # Even tighter Bollinger Bands
VOLUME_THRESHOLD = 1.05  # Lower volume threshold further
STOP_LOSS_PCT = 0.15   # Tighter stop loss for larger position
MAX_POSITIONS = 3

# Backtesting parameters
BACKTEST_DAYS = 90     # Number of days to backtest

# API credentials (replace with your own)
BYBIT_API_KEY = ""
BYBIT_API_SECRET = ""
TESTNET = True         # Set to False for real trading

# Data settings
DATA_PROVIDER = "ccxt"  # Alternative: "yfinance"

# Logging
LOG_LEVEL = "INFO"     # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Position parameters
TAKE_PROFIT_PCT = 0.25  # Lower take profit for quicker gains
MAX_DAILY_TRADES = 20   # Increased max trades
MAX_DAILY_LOSS_PCT = 2.0
COOLDOWN_MINUTES = 3    # Shorter cooldown between trades

# Technical
UPDATE_INTERVAL = 5  # Seconds between updates 