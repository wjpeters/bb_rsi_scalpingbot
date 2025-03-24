"""
Technical indicators module for calculating RSI and Bollinger Bands
"""

import pandas as pd
import numpy as np
import logging

import config

logger = logging.getLogger("indicators")

def calculate_rsi(df, period=config.RSI_PERIOD):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        df: DataFrame with price data, must have 'close' column
        period: RSI period (default: 14)
        
    Returns:
        Series with RSI values
    """
    close_delta = df['close'].diff()
    
    # Make two series: one for gains and one for losses
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    # Calculate the EWMA (Exponential Weighted Moving Average)
    ma_up = up.ewm(com=period-1, adjust=True, min_periods=period).mean()
    ma_down = down.ewm(com=period-1, adjust=True, min_periods=period).mean()
    
    # Calculate RS (Relative Strength)
    rs = ma_up / ma_down
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_bollinger_bands(df, period=config.BB_PERIOD, std_dev=config.BB_STD):
    """
    Calculate Bollinger Bands
    
    Args:
        df: DataFrame with price data, must have 'close' column
        period: Bollinger Bands period (default: 20)
        std_dev: Number of standard deviations (default: 2)
        
    Returns:
        DataFrame with Bollinger Bands (upper, middle, lower)
    """
    # Calculate middle band - Simple Moving Average
    middle_band = df['close'].rolling(window=period).mean()
    
    # Calculate standard deviation
    rolling_std = df['close'].rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    # Create and return DataFrame with all Bollinger Bands
    bands = pd.DataFrame({
        'bb_upper': upper_band,
        'bb_middle': middle_band,
        'bb_lower': lower_band
    })
    
    return bands

def add_indicators(df):
    """
    Add all technical indicators to a DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional indicator columns
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Calculate RSI
    df['rsi'] = calculate_rsi(df)
    
    # Calculate Bollinger Bands
    bollinger_bands = calculate_bollinger_bands(df)
    df = pd.concat([df, bollinger_bands], axis=1)
    
    # Calculate previous RSI for crossover detection
    df['rsi_prev'] = df['rsi'].shift(1)
    
    # Log info
    logger.info(f"Added indicators to DataFrame with shape {df.shape}")
    
    return df 