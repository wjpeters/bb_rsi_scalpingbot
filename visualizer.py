"""
Visualization module for creating beautiful trading charts with Apple-inspired design
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import seaborn as sns
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO

import config

logger = logging.getLogger("visualizer")

# Set Apple-inspired style
plt.style.use('dark_background')
APPLE_BLUE = '#007AFF'
APPLE_GREEN = '#34C759'
APPLE_RED = '#FF3B30'
APPLE_ORANGE = '#FF9500'
APPLE_PURPLE = '#AF52DE'
APPLE_GREY = '#8E8E93'
APPLE_BACKGROUND = '#000000'
APPLE_FOREGROUND = '#FFFFFF'

class TradingVisualizer:
    """
    Class for creating beautiful trading visualizations with Apple-inspired design
    """
    
    def __init__(self, dark_mode=True):
        """Initialize the visualizer with specified theme"""
        self.dark_mode = dark_mode
        
        # Set style
        plt.style.use('dark_background' if dark_mode else 'default')
        
        # Theme colors
        self.background_color = '#1C1C1E' if dark_mode else '#FFFFFF'
        self.text_color = '#FFFFFF' if dark_mode else '#000000'
        self.grid_color = '#2C2C2E' if dark_mode else '#E5E5E5'
        
        # Use system fonts instead of SF Pro Display
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
        
        self.colors = {
            'price': '#007AFF',      # Blue
            'volume': '#8E8E93',     # Gray
            'rsi': '#FF9500',        # Orange
            'bb_upper': '#AF52DE',   # Purple
            'bb_middle': '#8E8E93',  # Gray
            'bb_lower': '#AF52DE',   # Purple
            'buy': '#34C759',        # Green
            'sell': '#FF3B30',       # Red
            'grid': '#48484A'        # Dark gray
        }
        
        logger.info(f"Initialized TradingVisualizer with theme: {'dark' if dark_mode else 'light'}")
    
    def create_backtest_chart(self, df, trades=None, title='Bitcoin RSI + Bollinger Bands Scalping Strategy', 
                             save_path=None, show_trades=True, figsize=(14, 10)):
        """
        Create a beautiful backtest visualization chart
        
        Args:
            df: DataFrame with price and indicator data
            trades: List of trade dictionaries
            title: Chart title
            save_path: Path to save the figure
            show_trades: Whether to show trade markers
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        try:
            # Create figure and grid
            fig = plt.figure(figsize=figsize, facecolor=self.background_color)
            gs = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)
            
            # Price and Bollinger Bands subplot
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(df.index, df['close'], color=APPLE_BLUE, linewidth=1.5, label='BTC Price')
            ax1.plot(df.index, df['bb_upper'], color=APPLE_PURPLE, linewidth=0.8, alpha=0.7, label='Upper BB')
            ax1.plot(df.index, df['bb_middle'], color=APPLE_GREY, linewidth=0.8, alpha=0.7, label='Middle BB')
            ax1.plot(df.index, df['bb_lower'], color=APPLE_PURPLE, linewidth=0.8, alpha=0.7, label='Lower BB')
            ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'], color=APPLE_PURPLE, alpha=0.05)
            
            # Add trade markers if available
            if show_trades and 'signal' in df.columns:
                # Buy signals
                buy_signals = df[df['signal'] == 1]
                ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color=APPLE_GREEN, 
                          s=100, label='Buy', zorder=5, alpha=0.8)
                
                # Sell signals
                sell_signals = df[df['signal'] == -1]
                ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color=APPLE_RED, 
                          s=100, label='Sell', zorder=5, alpha=0.8)
            
            # Configure price chart
            ax1.set_title(title, fontsize=16, color=self.text_color, pad=10, fontweight='bold')
            ax1.set_ylabel('Price (USD)', fontsize=12, color=self.text_color)
            ax1.grid(True, alpha=0.2)
            ax1.legend(loc='upper left', facecolor=self.background_color, edgecolor=self.grid_color)
            
            # Format y-axis to show dollar values
            ax1.yaxis.set_major_formatter('${x:,.0f}')
            
            # RSI subplot
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax2.plot(df.index, df['rsi'], color=APPLE_ORANGE, linewidth=1.2, label='RSI')
            ax2.axhline(70, color=APPLE_RED, linestyle='--', linewidth=0.8, alpha=0.5)
            ax2.axhline(50, color=APPLE_GREY, linestyle='--', linewidth=0.8, alpha=0.5)
            ax2.axhline(30, color=APPLE_GREEN, linestyle='--', linewidth=0.8, alpha=0.5)
            ax2.fill_between(df.index, df['rsi'], 30, where=(df['rsi'] < 30), color=APPLE_GREEN, alpha=0.3)
            ax2.fill_between(df.index, df['rsi'], 70, where=(df['rsi'] > 70), color=APPLE_RED, alpha=0.3)
            ax2.set_ylabel('RSI', fontsize=12, color=self.text_color)
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.2)
            
            # Balance/Equity subplot
            if 'balance' in df.columns:
                ax3 = fig.add_subplot(gs[2], sharex=ax1)
                ax3.plot(df.index, df['balance'], color=APPLE_GREEN, linewidth=1.2, label='Equity')
                
                # Add drawdown shading
                if 'peak_balance' in df.columns:
                    ax3.fill_between(df.index, df['balance'], df['peak_balance'], 
                                   alpha=0.3, color=APPLE_RED, label='Drawdown')
                                   
                ax3.set_ylabel('Equity (USD)', fontsize=12, color=self.text_color)
                ax3.yaxis.set_major_formatter('${x:,.0f}')
                ax3.grid(True, alpha=0.2)
                ax3.legend(loc='upper left', facecolor=self.background_color, edgecolor=self.grid_color)
            else:
                # If no balance data, show volume instead
                ax3 = fig.add_subplot(gs[2], sharex=ax1)
                ax3.bar(df.index, df['volume'], color=APPLE_GREY, alpha=0.5, label='Volume')
                ax3.set_ylabel('Volume', fontsize=12, color=self.text_color)
                ax3.grid(True, alpha=0.2)
            
            # Format x-axis date
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
            # Only show x-axis labels on bottom subplot
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)
            
            # Add performance metrics if available
            if trades and len(trades) > 0:
                # Calculate metrics
                total_trades = len(trades)
                winning_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
                win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
                total_profit = sum(trade.get('profit', 0) for trade in trades)
                
                # Create text for metrics
                metrics_text = (
                    f"Total Trades: {total_trades}\n"
                    f"Win Rate: {win_rate:.1f}%\n"
                    f"Total Profit: ${total_profit:.2f}"
                )
                
                # Add metrics text to chart
                ax1.text(0.02, 0.05, metrics_text, transform=ax1.transAxes, 
                        fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                        bbox=dict(facecolor=self.background_color, alpha=0.7, edgecolor=self.grid_color))
            
            # Adjust layout and save/show
            plt.tight_layout()
            fig.subplots_adjust(top=0.95)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=self.background_color)
                logger.info(f"Chart saved to {save_path}")
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating backtest chart: {str(e)}")
            raise
            
    def create_dashboard(self, df, metrics, trades, title='Trading Dashboard', save_path=None, figsize=(18, 12)):
        """
        Create a comprehensive trading dashboard with multiple plots
        
        Args:
            df: DataFrame with price and indicator data
            metrics: Dictionary with performance metrics
            trades: List of trade dictionaries
            title: Dashboard title
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        try:
            # Create figure and grid
            fig = plt.figure(figsize=figsize, facecolor=self.background_color)
            gs = GridSpec(4, 4, height_ratios=[3, 1, 1, 1], hspace=0.3, wspace=0.3)
            
            # Main price chart (spans top row)
            ax_price = fig.add_subplot(gs[0, :])
            ax_price.plot(df.index, df['close'], color=APPLE_BLUE, linewidth=1.5, label='BTC Price')
            ax_price.plot(df.index, df['bb_upper'], color=APPLE_PURPLE, linewidth=0.8, alpha=0.7, label='Upper BB')
            ax_price.plot(df.index, df['bb_middle'], color=APPLE_GREY, linewidth=0.8, alpha=0.7, label='Middle BB')
            ax_price.plot(df.index, df['bb_lower'], color=APPLE_PURPLE, linewidth=0.8, alpha=0.7, label='Lower BB')
            ax_price.fill_between(df.index, df['bb_upper'], df['bb_lower'], color=APPLE_PURPLE, alpha=0.05)
            
            # Buy signals
            buy_signals = df[df['signal'] == 1]
            ax_price.scatter(buy_signals.index, buy_signals['close'], marker='^', color=APPLE_GREEN, 
                          s=100, label='Buy', zorder=5, alpha=0.8)
            
            # Sell signals
            sell_signals = df[df['signal'] == -1]
            ax_price.scatter(sell_signals.index, sell_signals['close'], marker='v', color=APPLE_RED, 
                          s=100, label='Sell', zorder=5, alpha=0.8)
            
            ax_price.set_title(title, fontsize=18, color=self.text_color, pad=10, fontweight='bold')
            ax_price.set_ylabel('Price (USD)', fontsize=12, color=self.text_color)
            ax_price.grid(True, alpha=0.2)
            ax_price.legend(loc='upper left', facecolor=self.background_color, edgecolor=self.grid_color)
            ax_price.yaxis.set_major_formatter('${x:,.0f}')
            
            # RSI subplot
            ax_rsi = fig.add_subplot(gs[1, :2], sharex=ax_price)
            ax_rsi.plot(df.index, df['rsi'], color=APPLE_ORANGE, linewidth=1.2, label='RSI')
            ax_rsi.axhline(70, color=APPLE_RED, linestyle='--', linewidth=0.8, alpha=0.5)
            ax_rsi.axhline(50, color=APPLE_GREY, linestyle='--', linewidth=0.8, alpha=0.5)
            ax_rsi.axhline(30, color=APPLE_GREEN, linestyle='--', linewidth=0.8, alpha=0.5)
            ax_rsi.fill_between(df.index, df['rsi'], 30, where=(df['rsi'] < 30), color=APPLE_GREEN, alpha=0.3)
            ax_rsi.fill_between(df.index, df['rsi'], 70, where=(df['rsi'] > 70), color=APPLE_RED, alpha=0.3)
            ax_rsi.set_ylabel('RSI', fontsize=12, color=self.text_color)
            ax_rsi.set_ylim(0, 100)
            ax_rsi.grid(True, alpha=0.2)
            
            # Volume subplot
            ax_vol = fig.add_subplot(gs[1, 2:], sharex=ax_price)
            ax_vol.bar(df.index, df['volume'], color=APPLE_GREY, alpha=0.5, label='Volume')
            ax_vol.set_ylabel('Volume', fontsize=12, color=self.text_color)
            ax_vol.grid(True, alpha=0.2)
            
            # Equity/Balance curve
            ax_equity = fig.add_subplot(gs[2, :])
            ax_equity.plot(df.index, df['balance'], color=APPLE_GREEN, linewidth=1.2, label='Equity')
            ax_equity.fill_between(df.index, df['balance'], df['peak_balance'], 
                                alpha=0.3, color=APPLE_RED, label='Drawdown')
            ax_equity.set_ylabel('Equity (USD)', fontsize=12, color=self.text_color)
            ax_equity.yaxis.set_major_formatter('${x:,.0f}')
            ax_equity.grid(True, alpha=0.2)
            ax_equity.legend(loc='upper left', facecolor=self.background_color, edgecolor=self.grid_color)
            
            # Format x-axis dates
            for ax in [ax_price, ax_rsi, ax_vol, ax_equity]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
            # Only show x-labels on bottom plots
            plt.setp(ax_price.get_xticklabels(), visible=False)
            plt.setp(ax_rsi.get_xticklabels(), visible=False)
            plt.setp(ax_vol.get_xticklabels(), visible=False)
            
            # Performance metrics plots
            if metrics:
                # Strategy performance text box
                ax_metrics = fig.add_subplot(gs[3, 0])
                ax_metrics.axis('off')
                
                metrics_text = (
                    f"Strategy Performance\n"
                    f"-------------------\n"
                    f"Total Trades: {metrics.get('total_trades', 0)}\n"
                    f"Win Rate: {metrics.get('win_rate', 0):.1f}%\n"
                    f"Total Profit: ${metrics.get('total_profit', 0):.2f}\n"
                    f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%\n"
                    f"Daily Win Rate: {metrics.get('daily_win_rate', 0):.1f}%\n"
                    f"Avg Daily Profit: ${metrics.get('avg_daily_profit', 0):.2f}"
                )
                
                ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                              fontsize=12, verticalalignment='top', fontfamily='monospace',
                              color=self.text_color)
                
                # Profit distribution histogram
                ax_profit_dist = fig.add_subplot(gs[3, 1])
                if trades and len(trades) > 0:
                    profits = [trade.get('profit', 0) for trade in trades]
                    ax_profit_dist.hist(profits, bins=15, color=APPLE_BLUE, alpha=0.7)
                    ax_profit_dist.axvline(0, color=APPLE_GREY, linestyle='--', linewidth=1)
                    ax_profit_dist.set_title('Profit Distribution', fontsize=12, color=self.text_color)
                    ax_profit_dist.set_xlabel('Profit (USD)', fontsize=10, color=self.text_color)
                    ax_profit_dist.set_ylabel('Frequency', fontsize=10, color=self.text_color)
                    ax_profit_dist.grid(True, alpha=0.2)
                
                # Win rate by day of week
                ax_weekday = fig.add_subplot(gs[3, 2])
                if trades and len(trades) > 0:
                    # Extract day of week for each trade
                    days = []
                    day_profits = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
                    
                    for trade in trades:
                        if 'entry_time' in trade and isinstance(trade['entry_time'], (datetime, pd.Timestamp)):
                            day = trade['entry_time'].weekday()
                            profit = trade.get('profit', 0)
                            days.append(day)
                            day_profits[day].append(profit)
                    
                    # Calculate win rate by day
                    day_win_rates = []
                    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    
                    for day in range(7):
                        profits = day_profits[day]
                        if profits:
                            win_rate = sum(1 for p in profits if p > 0) / len(profits) * 100
                            day_win_rates.append(win_rate)
                        else:
                            day_win_rates.append(0)
                    
                    # Plot
                    ax_weekday.bar(day_names, day_win_rates, color=APPLE_BLUE, alpha=0.7)
                    ax_weekday.set_title('Win Rate by Day', fontsize=12, color=self.text_color)
                    ax_weekday.set_ylabel('Win Rate (%)', fontsize=10, color=self.text_color)
                    ax_weekday.grid(True, alpha=0.2)
                    ax_weekday.set_ylim(0, 100)
                
                # Trade duration scatter plot
                ax_duration = fig.add_subplot(gs[3, 3])
                if trades and len(trades) > 0:
                    durations = []
                    profits = []
                    
                    for trade in trades:
                        if ('entry_time' in trade and 'exit_time' in trade and 
                            isinstance(trade['entry_time'], (datetime, pd.Timestamp)) and
                            isinstance(trade['exit_time'], (datetime, pd.Timestamp))):
                            
                            duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 60  # minutes
                            profit = trade.get('profit', 0)
                            
                            durations.append(duration)
                            profits.append(profit)
                    
                    if durations:
                        colors = [APPLE_GREEN if p > 0 else APPLE_RED for p in profits]
                        ax_duration.scatter(durations, profits, alpha=0.7, c=colors)
                        ax_duration.set_title('Profit vs Duration', fontsize=12, color=self.text_color)
                        ax_duration.set_xlabel('Duration (minutes)', fontsize=10, color=self.text_color)
                        ax_duration.set_ylabel('Profit (USD)', fontsize=10, color=self.text_color)
                        ax_duration.grid(True, alpha=0.2)
                        ax_duration.axhline(0, color=APPLE_GREY, linestyle='--', linewidth=1)
            
            # Adjust layout and save/show
            plt.tight_layout()
            fig.subplots_adjust(top=0.95)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=self.background_color)
                logger.info(f"Dashboard saved to {save_path}")
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise
            
    def create_live_chart(self, df, current_position=None, title='Live Trading', save_path=None):
        """
        Create a live trading chart focused on recent price action
        
        Args:
            df: DataFrame with recent price data
            current_position: Dictionary with current position details
            title: Chart title
            save_path: Path to save the figure
            
        Returns:
            matplotlib figure
        """
        try:
            # Create figure and grid (simpler for live view)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, 
                                           sharex=True, facecolor=self.background_color)
            
            # Plot price and Bollinger Bands
            ax1.plot(df.index, df['close'], color=APPLE_BLUE, linewidth=1.5, label='BTC Price')
            ax1.plot(df.index, df['bb_upper'], color=APPLE_PURPLE, linewidth=0.8, alpha=0.7, label='Upper BB')
            ax1.plot(df.index, df['bb_middle'], color=APPLE_GREY, linewidth=0.8, alpha=0.7, label='Middle BB')
            ax1.plot(df.index, df['bb_lower'], color=APPLE_PURPLE, linewidth=0.8, alpha=0.7, label='Lower BB')
            ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'], color=APPLE_PURPLE, alpha=0.05)
            
            # Add current price line
            if not df.empty:
                current_price = df['close'].iloc[-1]
                ax1.axhline(current_price, color=APPLE_GREEN, linestyle='--', linewidth=1, alpha=0.8)
                ax1.text(df.index[-1], current_price, f' ${current_price:.2f}', 
                       color=APPLE_GREEN, va='center', fontweight='bold')
            
            # Plot RSI
            ax2.plot(df.index, df['rsi'], color=APPLE_ORANGE, linewidth=1.2, label='RSI')
            ax2.axhline(70, color=APPLE_RED, linestyle='--', linewidth=0.8, alpha=0.5)
            ax2.axhline(50, color=APPLE_GREY, linestyle='--', linewidth=0.8, alpha=0.5)
            ax2.axhline(30, color=APPLE_GREEN, linestyle='--', linewidth=0.8, alpha=0.5)
            ax2.fill_between(df.index, df['rsi'], 30, where=(df['rsi'] < 30), color=APPLE_GREEN, alpha=0.3)
            ax2.fill_between(df.index, df['rsi'], 70, where=(df['rsi'] > 70), color=APPLE_RED, alpha=0.3)
            
            # Set titles and labels
            ax1.set_title(f"{title} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                         fontsize=16, color=self.text_color, pad=10, fontweight='bold')
            ax1.set_ylabel('Price (USD)', fontsize=12, color=self.text_color)
            ax2.set_ylabel('RSI', fontsize=12, color=self.text_color)
            ax2.set_ylim(0, 100)
            
            # Format axes
            ax1.grid(True, alpha=0.2)
            ax2.grid(True, alpha=0.2)
            ax1.legend(loc='upper left', facecolor=self.background_color, edgecolor=self.grid_color)
            ax1.yaxis.set_major_formatter('${x:,.0f}')
            
            # Format x-axis date
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax1.get_xticklabels(), visible=False)
            
            # Add position info if available
            if current_position and current_position.get('in_position'):
                entry_price = current_position.get('entry_price', 0)
                entry_time = current_position.get('entry_time', datetime.now())
                position_size = current_position.get('position_size', config.POSITION_SIZE)
                
                # Calculate current P&L if we have price data
                if not df.empty:
                    current_price = df['close'].iloc[-1]
                    pnl = (current_price - entry_price) * position_size
                    pnl_pct = (current_price / entry_price - 1) * 100
                    
                    position_text = (
                        f"Current Position\n"
                        f"--------------\n"
                        f"Entry Price: ${entry_price:.2f}\n"
                        f"Current Price: ${current_price:.2f}\n"
                        f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%)\n"
                        f"Position Size: {position_size} BTC\n"
                        f"Entry Time: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    
                    # Mark entry price on chart
                    ax1.axhline(entry_price, color=APPLE_GREEN, linestyle='--', linewidth=1)
                    ax1.text(df.index[0], entry_price, f' Entry: ${entry_price:.2f}', 
                           color=APPLE_GREEN, va='center')
                    
                    # Add position text box
                    ax1.text(0.02, 0.05, position_text, transform=ax1.transAxes, 
                           fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                           bbox=dict(facecolor=self.background_color, alpha=0.7, edgecolor=self.grid_color),
                           fontfamily='monospace', color=self.text_color)
            
            # Adjust layout and save/show
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=self.background_color)
                logger.info(f"Live chart saved to {save_path}")
                
            return fig
            
        except Exception as e:
            logger.error(f"Error creating live chart: {str(e)}")
            raise

    def create_backtest_report(self, df: pd.DataFrame, trades: List[dict], save_path: str) -> None:
        """
        Create a comprehensive PDF report of backtest results.
        
        Args:
            df (pd.DataFrame): DataFrame containing price and indicator data
            trades (List[dict]): List of trade dictionaries
            save_path (str): Path where the PDF report will be saved
        """
        try:
            # Calculate metrics
            total_trades = len([t for t in trades if 'exit_price' in t])
            winning_trades = len([t for t in trades if t.get('pnl') is not None and t.get('pnl', 0) > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl') is not None)
            avg_profit = total_profit / total_trades if total_trades > 0 else 0
            
            # Calculate max drawdown
            cumulative_pnl = [0]
            for trade in trades:
                if 'pnl' in trade and trade['pnl'] is not None:
                    cumulative_pnl.append(cumulative_pnl[-1] + trade['pnl'])
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdowns = [(peak - trough) / peak * 100 if peak > 0 else 0 
                        for peak, trough in zip(running_max, cumulative_pnl)]
            max_drawdown = min(drawdowns) if drawdowns else 0
            
            # Calculate trading days using pandas datetime index
            trading_days = len(pd.Series(df.index).dt.date.unique())
            avg_daily_profit = total_profit / trading_days if trading_days > 0 else 0
            
            # Store metrics in a dictionary
            metrics = {
                'Total Trades': total_trades,
                'Win Rate': f"{win_rate:.2f}%",
                'Total Profit': f"${total_profit:.2f}",
                'Average Profit': f"${avg_profit:.2f}",
                'Max Drawdown': f"{max_drawdown:.2f}%",
                'Trading Days': trading_days,
                'Average Daily Profit': f"${avg_daily_profit:.2f}"
            }
            
            # Create visualization with buy/sell signals and save to bytes
            img_data = BytesIO()
            
            # Create figure with entry/exit points
            fig = plt.figure(figsize=(10, 7), facecolor=self.background_color)
            gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
            
            # Price and Bollinger Bands subplot
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(df.index, df['close'], color=APPLE_BLUE, linewidth=1.5, label='BTC Price')
            ax1.plot(df.index, df['bb_upper'], color=APPLE_PURPLE, linewidth=0.8, alpha=0.7, label='Upper BB')
            ax1.plot(df.index, df['bb_middle'], color=APPLE_GREY, linewidth=0.8, alpha=0.7, label='Middle BB')
            ax1.plot(df.index, df['bb_lower'], color=APPLE_PURPLE, linewidth=0.8, alpha=0.7, label='Lower BB')
            ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'], color=APPLE_PURPLE, alpha=0.05)
            
            # Add entry/exit points
            for trade in trades:
                if 'entry_time' in trade and 'exit_time' in trade:
                    entry_time = trade['entry_time']
                    exit_time = trade['exit_time']
                    entry_price = trade['entry_price']
                    exit_price = trade['exit_price']
                    trade_type = trade.get('type', 'unknown')
                    
                    # Entry points
                    if trade_type == 'long':
                        ax1.scatter(entry_time, entry_price, marker='^', color=APPLE_GREEN, s=100, zorder=5, label='Long Entry' if 'Long Entry' not in ax1.get_legend_handles_labels()[1] else '')
                    else:
                        ax1.scatter(entry_time, entry_price, marker='v', color=APPLE_RED, s=100, zorder=5, label='Short Entry' if 'Short Entry' not in ax1.get_legend_handles_labels()[1] else '')
                    
                    # Exit points
                    ax1.scatter(exit_time, exit_price, marker='o', color=APPLE_GREY, s=80, zorder=5, label='Exit' if 'Exit' not in ax1.get_legend_handles_labels()[1] else '')
            
            # Configure price chart
            ax1.set_title('Backtest Results', fontsize=16, color=self.text_color, pad=10, fontweight='bold')
            ax1.set_ylabel('Price (USD)', fontsize=12, color=self.text_color)
            ax1.grid(True, alpha=0.2)
            ax1.legend(loc='upper left', facecolor=self.background_color, edgecolor=self.grid_color)
            ax1.yaxis.set_major_formatter('${x:,.0f}')
            
            # RSI subplot
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax2.plot(df.index, df['rsi'], color=APPLE_ORANGE, linewidth=1.2, label='RSI')
            ax2.axhline(70, color=APPLE_RED, linestyle='--', linewidth=0.8, alpha=0.5)
            ax2.axhline(50, color=APPLE_GREY, linestyle='--', linewidth=0.8, alpha=0.5)
            ax2.axhline(30, color=APPLE_GREEN, linestyle='--', linewidth=0.8, alpha=0.5)
            ax2.fill_between(df.index, df['rsi'], 30, where=(df['rsi'] < 30), color=APPLE_GREEN, alpha=0.3)
            ax2.fill_between(df.index, df['rsi'], 70, where=(df['rsi'] > 70), color=APPLE_RED, alpha=0.3)
            ax2.set_ylabel('RSI', fontsize=12, color=self.text_color)
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.2)
            
            # Format x-axis date
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # Save figure
            plt.tight_layout()
            fig.savefig(img_data, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            img_data.seek(0)
            
            # Create PDF
            doc = SimpleDocTemplate(
                save_path,
                pagesize=landscape(letter),
                rightMargin=30,
                leftMargin=30,
                topMargin=30,
                bottomMargin=30
            )
            
            # Prepare content
            story = []
            styles = getSampleStyleSheet()
            
            # Add title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1
            )
            story.append(Paragraph("Backtest Results Report", title_style))
            
            # Add date range
            date_range = f"Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
            story.append(Paragraph(date_range, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Add configuration parameters
            story.append(Paragraph("Strategy Configuration", styles['Heading2']))
            story.append(Spacer(1, 10))
            
            config_data = [
                ["Parameter", "Value"],
                ["Position Size", str(config.POSITION_SIZE)],
                ["RSI Period", str(config.RSI_PERIOD)],
                ["RSI Overbought", str(config.RSI_OVERBOUGHT)],
                ["RSI Oversold", str(config.RSI_OVERSOLD)],
                ["BB Period", str(config.BB_PERIOD)],
                ["BB Standard Dev", str(config.BB_STD)],
                ["Stop Loss %", f"{config.STOP_LOSS_PCT:.1f}%"],
                ["Take Profit %", f"{config.TAKE_PROFIT_PCT:.1f}%"],
                ["Max Daily Trades", str(config.MAX_DAILY_TRADES)],
                ["Max Daily Loss %", f"{config.MAX_DAILY_LOSS_PCT:.1f}%"],
                ["Cooldown Minutes", str(config.COOLDOWN_MINUTES)]
            ]
            
            config_table = Table(config_data, colWidths=[200, 200])
            config_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 12),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(config_table)
            story.append(Spacer(1, 20))
            
            # Add performance summary
            story.append(Paragraph("Performance Summary", styles['Heading2']))
            story.append(Spacer(1, 10))
            
            # Create metrics table
            data = [[Paragraph("Metric", styles['Heading3']), Paragraph("Value", styles['Heading3'])]]
            for metric, value in metrics.items():
                data.append([metric, str(value)])
            
            table = Table(data, colWidths=[200, 200])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 12),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
            
            # Add trade details section
            story.append(Paragraph("Trade Details", styles['Heading2']))
            story.append(Spacer(1, 10))
            
            # Create trade details table with more information
            trade_data = [["Date", "Type", "Entry Price", "Exit Price", "PnL", "Duration", "Exit Reason"]]
            
            for trade in trades:
                if 'entry_time' in trade and 'entry_price' in trade:
                    entry_time = trade['entry_time'].strftime('%Y-%m-%d %H:%M') if isinstance(trade['entry_time'], (datetime, pd.Timestamp)) else 'Unknown'
                    trade_type = trade.get('type', 'Unknown').upper()
                    entry_price = f"${trade.get('entry_price', 0):.2f}"
                    exit_price = f"${trade['exit_price']:.2f}" if trade.get('exit_price') is not None else 'N/A'
                    pnl = f"${trade['pnl']:.2f}" if trade.get('pnl') is not None else 'N/A'
                    
                    # Calculate duration if we have both entry and exit times
                    duration = 'N/A'
                    if 'exit_time' in trade and isinstance(trade['exit_time'], (datetime, pd.Timestamp)):
                        duration_mins = int((trade['exit_time'] - trade['entry_time']).total_seconds() / 60)
                        duration = f"{duration_mins} min"
                    
                    exit_reason = trade.get('exit_reason', 'Unknown')
                    
                    trade_data.append([
                        entry_time,
                        trade_type,
                        entry_price,
                        exit_price,
                        pnl,
                        duration,
                        exit_reason
                    ])
            
            trade_table = Table(trade_data, colWidths=[100, 60, 80, 80, 70, 60, 100])
            trade_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(trade_table)
            story.append(Spacer(1, 20))
            
            # Add chart
            # Scale image to fit page width while maintaining aspect ratio
            img = Image(img_data)
            aspect = img.imageHeight / float(img.imageWidth)
            available_width = doc.width * 0.9
            img.drawWidth = available_width
            img.drawHeight = available_width * aspect
            story.append(img)
            
            # Build PDF
            doc.build(story)
            logger.info(f"PDF report saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating PDF report: {str(e)}")
            raise

    def plot_backtest_results(self, df: pd.DataFrame, trades: List[Dict], save_path: str = 'backtest_results.pdf') -> None:
        """
        Create a PDF report of backtest results
        
        Args:
            df: DataFrame with price and indicator data
            trades: List of trade dictionaries
            save_path: Path to save the report (should end with .pdf)
        """
        try:
            # Convert Path to string if needed
            save_path = str(save_path)
            
            # Ensure save_path ends with .pdf
            if not save_path.endswith('.pdf'):
                save_path = save_path.replace('.png', '.pdf')
                if not save_path.endswith('.pdf'):
                    save_path += '.pdf'
            
            # Create the PDF report directly
            self.create_backtest_report(df, trades, save_path)
            
        except Exception as e:
            logger.error(f"Error creating backtest report: {str(e)}")
            raise 