"""
RSI + Bollinger Bands Scalping Strategy Implementation
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import config
from indicators import add_indicators

logger = logging.getLogger("strategy")

class RSI_BB_ScalpingStrategy:
    """
    Bitcoin scalping strategy using RSI and Bollinger Bands
    
    Strategy Rules:
    - Entry (Buy): RSI < 30 AND price < lower Bollinger Band
    - Exit (Sell): 
        1. Price > middle Bollinger Band OR
        2. RSI crosses above 50 OR
        3. Price drops below stop loss (1% below entry)
    """
    
    def __init__(self, position_size: float = config.POSITION_SIZE):
        """Initialize the strategy with given parameters"""
        self.position_size = position_size
        self.logger = logging.getLogger('strategy')
        self.logger.info(f"Initialized RSI_BB_ScalpingStrategy with position size: {position_size} BTC")
        self.current_position: Optional[Dict] = None
        self.trades_today = 0
        self.daily_pnl = 0.0
        
    def reset(self):
        """Reset the strategy state"""
        self.current_position = None
        self.trades_today = 0
        self.daily_pnl = 0.0
        
    def generate_signals(self, df):
        """
        Generate trading signals based on strategy rules
        
        Args:
            df: DataFrame with price data and indicators
            
        Returns:
            DataFrame with added signal columns
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Add indicators if they don't exist
        if 'rsi' not in df.columns:
            df = add_indicators(df)
            
        # Initialize signal columns
        df['signal'] = 0  # 1 for buy, -1 for sell, 0 for no action
        df['in_position'] = False
        df['entry_price'] = np.nan
        df['exit_price'] = np.nan
        df['profit'] = np.nan
        
        # Current position tracking
        in_position = False
        entry_price = 0
        
        # Loop through each candle
        for i in range(1, len(df)):
            current_date = df.index[i].date()
            
            # Skip first few candles until indicators are calculated
            if pd.isna(df['rsi'].iloc[i]) or pd.isna(df['bb_lower'].iloc[i]):
                continue
                
            # If not in a position, check for entry signal
            if not in_position:
                # Entry condition: RSI < 30 AND close price < lower Bollinger Band
                if (df['rsi'].iloc[i] < config.RSI_OVERSOLD and 
                    df['close'].iloc[i] < df['bb_lower'].iloc[i]):
                    df.loc[df.index[i], 'signal'] = 1
                    df.loc[df.index[i], 'in_position'] = True
                    df.loc[df.index[i], 'entry_price'] = df['close'].iloc[i]
                    in_position = True
                    entry_price = df['close'].iloc[i]
            
            # If in a position, check for exit signal
            else:
                df.loc[df.index[i], 'in_position'] = True
                
                # Calculate stop loss price
                stop_loss_price = entry_price * (1 - config.STOP_LOSS_PCT/100)
                
                # Exit condition 1: Price > middle Bollinger Band
                condition1 = df['close'].iloc[i] >= df['bb_middle'].iloc[i]
                
                # Exit condition 2: RSI crosses above 50
                condition2 = (df['rsi_prev'].iloc[i] <= config.RSI_EXIT and 
                             df['rsi'].iloc[i] > config.RSI_EXIT)
                
                # Exit condition 3: Stop loss triggered
                condition3 = df['close'].iloc[i] <= stop_loss_price
                
                # If any exit condition is met
                if condition1 or condition2 or condition3:
                    df.loc[df.index[i], 'signal'] = -1
                    df.loc[df.index[i], 'in_position'] = False
                    df.loc[df.index[i], 'exit_price'] = df['close'].iloc[i]
                    
                    # Calculate profit in quote currency (USD)
                    profit = (df['close'].iloc[i] - entry_price) * self.position_size
                    df.loc[df.index[i], 'profit'] = profit
                    
                    # Reset position
                    in_position = False
                    entry_price = 0
        
        return df
        
    def backtest(self, df, initial_balance=1000):
        """
        Run a backtest of the strategy on historical data
        
        Args:
            df: DataFrame with price data
            initial_balance: Starting balance in USD
            
        Returns:
            DataFrame with backtest results and performance metrics
        """
        # Generate signals
        df = self.generate_signals(df)
        
        # Calculate cumulative profit
        df['cumulative_profit'] = df['profit'].fillna(0).cumsum()
        
        # Calculate balance
        df['balance'] = initial_balance + df['cumulative_profit']
        
        # Calculate drawdown
        df['peak_balance'] = df['balance'].cummax()
        df['drawdown'] = (df['peak_balance'] - df['balance']) / df['peak_balance'] * 100
        
        # Extract trades
        trades = df[df['signal'] != 0].copy()
        
        # Calculate performance metrics
        total_trades = len(trades[trades['signal'] == -1])
        winning_trades = len(trades[trades['profit'] > 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        total_profit = df['profit'].fillna(0).sum()
        max_drawdown = df['drawdown'].max()
        
        # Log results
        logger.info(f"Backtest completed: {total_trades} trades, {win_rate:.2f}% win rate")
        logger.info(f"Total profit: ${total_profit:.2f}, Max drawdown: {max_drawdown:.2f}%")
        
        # Group trades by day
        df['date'] = df.index.date
        daily_profits = df.groupby('date')['profit'].sum()
        
        # Calculate daily statistics
        avg_daily_profit = daily_profits.mean()
        profitable_days = len(daily_profits[daily_profits > 0])
        total_days = len(daily_profits)
        daily_win_rate = profitable_days / total_days * 100 if total_days > 0 else 0
        
        logger.info(f"Average daily profit: ${avg_daily_profit:.2f}")
        logger.info(f"Daily win rate: {daily_win_rate:.2f}% ({profitable_days}/{total_days} days)")
        
        # Store trades for analysis
        self.trades = []
        for i in trades[trades['signal'] == -1].index:
            entry_idx = trades[trades['signal'] == 1].index[0]
            entry_price = trades.loc[entry_idx, 'entry_price']
            exit_price = trades.loc[i, 'exit_price']
            profit = trades.loc[i, 'profit']
            
            self.trades.append({
                'entry_time': entry_idx,
                'exit_time': i,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit': profit,
                'profit_pct': (exit_price / entry_price - 1) * 100
            })
            
        # Calculate metrics
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'max_drawdown': max_drawdown,
            'avg_daily_profit': avg_daily_profit,
            'daily_win_rate': daily_win_rate,
            'final_balance': df['balance'].iloc[-1] if not df.empty else initial_balance
        }
        
        return df, metrics
    
    def check_entry_conditions(self, data: Dict) -> Tuple[bool, str, float]:
        """Check if entry conditions are met."""
        try:
            # Get latest values
            rsi = data['rsi'][-1]
            bb_lower = data['bb_lower'][-1]
            bb_upper = data['bb_upper'][-1]
            close = data['close'][-1]
            volume = data['volume'][-1]
            avg_volume = sum(data['volume'][-20:]) / 20

            # Volume check
            if volume < avg_volume * config.VOLUME_THRESHOLD:
                return False, "Insufficient volume", 0.0

            # Check for long entry
            if rsi <= config.RSI_OVERSOLD and close <= bb_lower:
                return True, "LONG", close

            # Check for short entry
            if rsi >= config.RSI_OVERBOUGHT and close >= bb_upper:
                return True, "SHORT", close

            return False, "No signal", 0.0

        except Exception as e:
            self.logger.error(f"Error in check_entry_conditions: {str(e)}")
            return False, f"Error: {str(e)}", 0.0

    def check_exit_conditions(self, data: Dict, position: Dict) -> Tuple[bool, str, float]:
        """Check if exit conditions are met."""
        try:
            close = data['close'][-1]
            entry_price = position['entry_price']
            position_type = position['type']

            # Calculate profit/loss percentage
            if position_type == "LONG":
                pnl_pct = ((close - entry_price) / entry_price) * 100
            else:  # SHORT
                pnl_pct = ((entry_price - close) / entry_price) * 100

            # Take profit check
            if pnl_pct >= config.TAKE_PROFIT_PCT:
                return True, "Take profit hit", close

            # Stop loss check
            if pnl_pct <= -config.STOP_LOSS_PCT:
                return True, "Stop loss hit", close

            # Mean reversion exit for LONG
            if position_type == "LONG" and close >= data['bb_upper'][-1]:
                return True, "Upper band hit", close

            # Mean reversion exit for SHORT
            if position_type == "SHORT" and close <= data['bb_lower'][-1]:
                return True, "Lower band hit", close

            return False, "Hold position", close

        except Exception as e:
            self.logger.error(f"Error in check_exit_conditions: {str(e)}")
            return True, f"Error: {str(e)}", data['close'][-1]

    def update_position(self, should_exit: bool, exit_price: float, exit_reason: str) -> None:
        """Update position status and track performance."""
        if should_exit and self.current_position:
            entry_price = self.current_position['entry_price']
            position_type = self.current_position['type']
            
            if position_type == "LONG":
                pnl = (exit_price - entry_price) * self.position_size
            else:  # SHORT
                pnl = (entry_price - exit_price) * self.position_size
            
            self.daily_pnl += pnl
            self.current_position = None
            self.logger.info(f"Closed {position_type} position: {exit_reason}, PnL: ${pnl:.2f}")

    def can_open_position(self) -> bool:
        """Check if we can open a new position based on risk management rules."""
        return (
            self.current_position is None and
            self.trades_today < config.MAX_DAILY_TRADES and
            self.daily_pnl > -config.MAX_DAILY_LOSS_PCT
        )

    def reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self.trades_today = 0
        self.daily_pnl = 0.0
        
    def execute_buy(self, price):
        """
        Execute a buy order at the given price
        
        Args:
            price: Current market price
            
        Returns:
            dict: Order details
        """
        if self.current_position:
            logger.warning("Already in position, buy order ignored")
            return None
            
        self.current_position = {
            'type': 'LONG',
            'entry_price': price
        }
        
        logger.info(f"BUY executed at ${price:.2f}, position size: {self.position_size} BTC")
        
        return {
            'side': 'buy',
            'price': price,
            'size': self.position_size,
            'time': datetime.now()
        }
        
    def execute_sell(self, price):
        """
        Execute a sell order at the given price
        
        Args:
            price: Current market price
            
        Returns:
            dict: Order details including profit
        """
        if not self.current_position:
            logger.warning("Not in position, sell order ignored")
            return None
            
        # Calculate profit
        profit = (price - self.current_position['entry_price']) * self.position_size
        profit_pct = (price / self.current_position['entry_price'] - 1) * 100
        
        logger.info(f"SELL executed at ${price:.2f}, profit: ${profit:.2f} ({profit_pct:.2f}%)")
        
        # Record trade
        trade = {
            'entry_time': self.current_position['entry_time'],
            'exit_time': datetime.now(),
            'entry_price': self.current_position['entry_price'],
            'exit_price': price,
            'profit': profit,
            'profit_pct': profit_pct
        }
        
        self.trades.append(trade)
        
        # Update daily profit
        self.daily_pnl += profit
        
        # Reset position
        self.current_position = None
        
        return {
            'side': 'sell',
            'price': price,
            'size': self.position_size,
            'profit': profit,
            'profit_pct': profit_pct
        }
        
    def check_daily_profit_target(self):
        """
        Check if daily profit target has been reached
        
        Returns:
            bool: True if target reached, False otherwise
        """
        if self.daily_pnl >= config.PROFIT_TARGET:
            logger.info(f"Daily profit target of ${config.PROFIT_TARGET:.2f} reached!")
            return True
        return False
        
    def reset_daily_profit(self):
        """Reset daily profit counter"""
        self.daily_pnl = 0.0 