# Bitcoin RSI + Bollinger Bands Scalping Bot

A sophisticated Bitcoin trading bot that implements a scalping strategy using RSI (Relative Strength Index) and Bollinger Bands indicators. The bot aims to generate small, frequent profits targeting approximately $20 per day through scalping Bitcoin on the Bybit exchange.

## Features

- 📈 RSI + Bollinger Bands strategy implementation
- 🔄 Real-time trading on Bybit exchange
- 📊 Beautiful Apple-inspired visualization
- 📱 Modern command-line interface with live updates
- 📈 Comprehensive backtesting capabilities
- 🎯 Daily profit target monitoring
- 📊 Detailed performance metrics and analytics
- 🔒 Support for testnet trading
- 🎨 Professional-grade charts and dashboards

## Strategy Overview

The bot implements a long-only strategy with the following rules:

### Entry Conditions (Buy)
- RSI (14-period) is below 30 (oversold)
- Price is below the lower Bollinger Band

### Exit Conditions (Sell)
- Price reaches or exceeds the middle Bollinger Band, OR
- RSI crosses above 50, OR
- Price drops 1% below entry price (stop loss)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bitcoin-scalping-bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Bybit API credentials:
   - Create a `.env` file in the project root
   - Add your API credentials:
   ```
   BYBIT_API_KEY=your_api_key
   BYBIT_API_SECRET=your_api_secret
   TESTNET=True  # Set to False for live trading
   ```

## Usage

### Backtesting
Run the strategy backtest:
```bash
python app.py backtest --days 90
```

This will:
- Fetch historical data
- Run the strategy simulation
- Generate performance metrics
- Create visualization charts
- Save results to `backtest_results.png`

### Live Trading
Start live trading:
```bash
python app.py live
```

Features:
- Real-time price monitoring
- Live trade execution
- Beautiful dashboard interface
- Performance tracking
- Automatic daily profit target monitoring

## Configuration

Edit `config.py` to customize:
- Trading parameters
- Position size
- Risk management settings
- API credentials
- Indicator parameters
- Daily profit target

## Safety Features

- Small default position size (0.001 BTC)
- 1% stop loss per trade
- Daily profit target monitoring
- Support for testnet trading
- Comprehensive error handling
- Rate limit management

## Visualization

The bot includes professional-grade visualization tools:
- Live trading dashboard
- Backtest results charts
- Performance metrics
- Trade analysis
- Profit distribution
- Win rate analysis by day

## Disclaimer

This trading bot is for educational purposes only. Cryptocurrency trading carries significant risks. Always:
- Start with small position sizes
- Test thoroughly on testnet first
- Never trade with money you can't afford to lose
- Monitor the bot's operation
- Understand the strategy and risks involved

## License

MIT License - feel free to modify and use as needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 