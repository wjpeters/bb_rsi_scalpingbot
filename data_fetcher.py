"""
Data fetcher module to obtain Bitcoin price data from various sources
"""

import datetime
import pandas as pd
import numpy as np
import ccxt
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_fetcher")

class DataFetcher:
    """Class to fetch and prepare market data for trading and backtesting"""
    
    def __init__(self, provider: str = 'ccxt'):
        """Initialize the data fetcher with specified provider"""
        self.provider = provider
        self.logger = logging.getLogger('data_fetcher')
        self.logger.info(f"Initialized DataFetcher with provider: {provider}")
        
        if provider == 'ccxt':
            self.exchange = ccxt.bybit({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
        
        self.symbol = 'BTCUSDT'
        self._latest_data = None
        self._last_update = None
        
    def get_historical_data(self, days: int) -> pd.DataFrame:
        """
        Fetch historical price data
        
        Args:
            days: Number of days of historical data to fetch
            
        Returns:
            pd.DataFrame with OHLCV data and indicators
        """
        try:
            self.logger.info(f"Fetching {days} days of {config.TIMEFRAME} data for BTCUSDT from Bybit")
            
            # Calculate timeframes
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Fetch candles
            candles = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=config.TIMEFRAME,
                since=int(start_time.timestamp() * 1000),
                limit=200
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                candles, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Process data
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add indicators
            df['rsi'] = self._calculate_rsi(df['close'].values)
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'].values)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            
            self.logger.info(f"Successfully fetched {len(df)} candles")
            self._latest_data = df
            self._last_update = datetime.now()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data from {self.provider}: {str(e)}")
            raise

    def get_latest_data(self) -> Dict:
        """
        Get latest market data
        
        Returns:
            Dict with latest OHLCV and indicator data
        """
        try:
            # Fetch latest candle
            candles = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=config.TIMEFRAME,
                limit=100
            )
            
            # Convert to arrays for technical analysis
            data = {
                'timestamp': [c[0] for c in candles],
                'open': [c[1] for c in candles],
                'high': [c[2] for c in candles],
                'low': [c[3] for c in candles],
                'close': [c[4] for c in candles],
                'volume': [c[5] for c in candles]
            }
            
            # Add indicators
            data['rsi'] = self._calculate_rsi(data['close'])
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['close'])
            data['bb_upper'] = bb_upper
            data['bb_middle'] = bb_middle
            data['bb_lower'] = bb_lower
            
            self._latest_data = data
            self._last_update = datetime.now()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching latest data: {str(e)}")
            raise

    def get_current_price(self) -> float:
        """Get current BTC price"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            self.logger.error(f"Error fetching current price: {str(e)}")
            return 0.0

    def _calculate_rsi(self, prices: List[float], period: int = config.RSI_PERIOD) -> List[float]:
        """Calculate RSI values"""
        prices = np.array(prices)
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up/down if down != 0 else 0
            rsi[i] = 100. - 100./(1. + rs)

        return rsi.tolist()

    def _calculate_bollinger_bands(
        self, 
        prices: List[float], 
        period: int = config.BB_PERIOD, 
        std: float = config.BB_STD
    ) -> tuple:
        """Calculate Bollinger Bands"""
        prices = np.array(prices)
        
        # Calculate middle band (SMA)
        middle_band = np.convolve(prices, np.ones(period)/period, mode='valid')
        
        # Calculate standard deviation
        rolling_std = np.array([np.std(prices[i:i+period]) for i in range(len(prices)-period+1)])
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * std)
        lower_band = middle_band - (rolling_std * std)
        
        # Pad bands to match input length
        padding = np.array([np.nan] * (period-1))
        middle_band = np.concatenate([padding, middle_band])
        upper_band = np.concatenate([padding, upper_band])
        lower_band = np.concatenate([padding, lower_band])
        
        return upper_band.tolist(), middle_band.tolist(), lower_band.tolist() 