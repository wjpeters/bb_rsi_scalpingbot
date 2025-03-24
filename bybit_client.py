"""
Bybit Exchange Client for executing trades on Bybit
"""

import logging
import time
from datetime import datetime
from pybit.unified_trading import HTTP
import pandas as pd

import config

logger = logging.getLogger("bybit_client")

class BybitClient:
    """
    Client for interacting with Bybit exchange API
    """
    
    def __init__(self, api_key=config.BYBIT_API_KEY, api_secret=config.BYBIT_API_SECRET, testnet=config.TESTNET):
        """Initialize the Bybit client with API credentials"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Check if API credentials are provided
        if not api_key or not api_secret:
            logger.warning("API key or secret not provided. Running in read-only mode.")
            self.trading_enabled = False
        else:
            self.trading_enabled = True
        
        # Initialize client session
        try:
            self.session = HTTP(
                testnet=testnet,
                api_key=api_key,
                api_secret=api_secret
            )
            
            # Get account details to validate connection
            if self.trading_enabled:
                account_info = self.session.get_wallet_balance(accountType="UNIFIED")
                logger.info(f"Connected to Bybit {'Testnet' if testnet else 'Mainnet'}")
                logger.info(f"Account type: UNIFIED")
            else:
                logger.info(f"Connected to Bybit {'Testnet' if testnet else 'Mainnet'} in read-only mode")
                
        except Exception as e:
            logger.error(f"Error connecting to Bybit API: {str(e)}")
            raise
            
    def get_ticker(self, symbol=config.SYMBOL):
        """
        Get current ticker information
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            dict with ticker information
        """
        try:
            ticker = self.session.get_tickers(category="spot", symbol=symbol)
            return ticker['result']['list'][0]
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            raise
            
    def get_balance(self, coin="BTC"):
        """
        Get account balance for a specific coin
        
        Args:
            coin: The coin to get balance for (e.g., 'BTC')
            
        Returns:
            float: Available balance
        """
        if not self.trading_enabled:
            logger.warning("Trading not enabled. Cannot fetch balance.")
            return 0
            
        try:
            wallet = self.session.get_wallet_balance(accountType="UNIFIED", coin=coin)
            
            if not wallet['result']['list']:
                return 0
                
            coin_balance = next((item for item in wallet['result']['list'][0]['coin'] if item['coin'] == coin), None)
            
            if not coin_balance:
                return 0
                
            return float(coin_balance['free'])
            
        except Exception as e:
            logger.error(f"Error fetching {coin} balance: {str(e)}")
            raise
            
    def get_usdt_balance(self):
        """Get USDT balance"""
        return self.get_balance("USDT")
        
    def get_btc_balance(self):
        """Get BTC balance"""
        return self.get_balance("BTC")
            
    def place_market_order(self, symbol=config.SYMBOL, side="Buy", qty=config.POSITION_SIZE):
        """
        Place a market order
        
        Args:
            symbol: Trading symbol
            side: 'Buy' or 'Sell'
            qty: Order quantity
            
        Returns:
            dict: Order response
        """
        if not self.trading_enabled:
            logger.warning("Trading not enabled. Cannot place order.")
            return None
            
        try:
            response = self.session.place_order(
                category="spot",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(qty)
            )
            
            logger.info(f"Placed {side} market order for {qty} {symbol}")
            return response['result']
            
        except Exception as e:
            logger.error(f"Error placing {side} market order: {str(e)}")
            raise
            
    def place_buy_order(self, symbol=config.SYMBOL, qty=config.POSITION_SIZE):
        """Place a market buy order"""
        return self.place_market_order(symbol=symbol, side="Buy", qty=qty)
        
    def place_sell_order(self, symbol=config.SYMBOL, qty=config.POSITION_SIZE):
        """Place a market sell order"""
        return self.place_market_order(symbol=symbol, side="Sell", qty=qty)
            
    def get_klines(self, symbol=config.SYMBOL, interval=config.TIMEFRAME, limit=200):
        """
        Get historical klines/candlesticks
        
        Args:
            symbol: Trading symbol
            interval: Timeframe interval (e.g., '5m')
            limit: Number of candles to retrieve
            
        Returns:
            pandas.DataFrame with OHLCV data
        """
        try:
            # Map 5m to 5 for Bybit API
            interval_map = {
                '1m': 1,
                '3m': 3,
                '5m': 5,
                '15m': 15,
                '30m': 30,
                '1h': 60,
                '2h': 120,
                '4h': 240,
                '6h': 360,
                '12h': 720,
                '1d': 'D',
                '1w': 'W',
                '1M': 'M'
            }
            
            bybit_interval = interval_map.get(interval, interval)
            
            klines = self.session.get_kline(
                category="spot",
                symbol=symbol,
                interval=bybit_interval,
                limit=limit
            )
            
            # Convert to DataFrame
            data = klines['result']['list']
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = df[col].astype(float)
                
            # Sort by time and set index
            df.sort_values('timestamp', inplace=True)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching klines: {str(e)}")
            raise
            
    def get_open_positions(self, symbol=config.SYMBOL):
        """
        Get open positions for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            dict: Position information
        """
        if not self.trading_enabled:
            logger.warning("Trading not enabled. Cannot fetch positions.")
            return None
            
        try:
            positions = self.session.get_positions(
                category="spot",
                symbol=symbol
            )
            
            return positions['result']['list']
            
        except Exception as e:
            logger.error(f"Error fetching positions: {str(e)}")
            raise 