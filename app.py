"""
Bitcoin RSI + Bollinger Bands Scalping Bot with Modern UI
"""

import logging
import time
from datetime import datetime, timedelta
import threading
from typing import Dict, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live

import config
from data_fetcher import DataFetcher
from strategy import RSI_BB_ScalpingStrategy
from bybit_client import BybitClient
from visualizer import TradingVisualizer

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('app')

# Initialize Rich console
console = Console()

class TradingBot:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.strategy = RSI_BB_ScalpingStrategy()
        self.client = BybitClient()
        self.visualizer = TradingVisualizer()
        self.last_update = datetime.now()
        self.running = False
        self.layout = Layout()
        self.setup_dashboard()

    def setup_dashboard(self):
        """Create the dashboard layout"""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        self.layout["main"].split_row(
            Layout(name="status", ratio=1),
            Layout(name="position", ratio=1)
        )

    def create_status_table(self, current_price: float) -> Table:
        """Create status table for the dashboard"""
        table = Table(title="Bot Status", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Status", "🟢 Running" if self.running else "🔴 Stopped")
        table.add_row("Current Price", f"${current_price:,.2f}")
        table.add_row("Daily Profit", f"${self.strategy.daily_pnl:,.2f}")
        table.add_row("Trades Today", str(self.strategy.trades_today))
        
        return table

    def create_position_table(self) -> Table:
        """Create position table for the dashboard"""
        table = Table(title="Current Position", show_header=True)
        table.add_column("Detail", style="cyan")
        table.add_column("Value", style="green")
        
        if self.strategy.current_position:
            pos = self.strategy.current_position
            entry_price = pos['entry_price']
            current_price = self.data_fetcher.get_current_price()
            pnl = (current_price - entry_price) * self.strategy.position_size
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            table.add_row("Type", pos['type'])
            table.add_row("Entry Price", f"${entry_price:,.2f}")
            table.add_row("Current Price", f"${current_price:,.2f}")
            table.add_row("PnL", f"${pnl:,.2f} ({pnl_pct:,.2f}%)")
        else:
            table.add_row("Status", "No Active Position")
        
        return table

    def update_dashboard(self, live: Live):
        """Update the dashboard with latest data"""
        try:
            current_price = self.data_fetcher.get_current_price()
            
            self.layout["header"].update(
                Panel("Bitcoin Scalping Bot", style="bold magenta")
            )
            
            self.layout["status"].update(
                self.create_status_table(current_price)
            )
            
            self.layout["position"].update(
                self.create_position_table()
            )
            
            self.layout["footer"].update(
                Panel(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                      style="dim")
            )
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {str(e)}")

    def run_backtest(self, days: int = 30):
        """Run backtest simulation"""
        try:
            console.print("[bold green]Starting backtest...[/bold green]")
            
            # Fetch historical data
            console.print("Fetching historical data...")
            df = self.data_fetcher.get_historical_data(days)
            
            # Calculate indicators
            console.print("Calculating indicators...")
            data = {
                'close': df['close'].values,
                'high': df['high'].values,
                'low': df['low'].values,
                'volume': df['volume'].values,
                'rsi': df['rsi'].values,
                'bb_upper': df['bb_upper'].values,
                'bb_middle': df['bb_middle'].values,
                'bb_lower': df['bb_lower'].values
            }
            
            # Run backtest simulation
            console.print("Running backtest...")
            trades = []
            daily_pnl = []
            current_day = None
            
            for i in range(len(df)):
                # Get current candle data
                current_data = {k: v[:i+1] for k, v in data.items()}
                current_time = df.index[i]
                
                # Reset daily stats at start of new day
                if current_day != current_time.date():
                    current_day = current_time.date()
                    self.strategy.reset_daily_stats()
                    daily_pnl.append(self.strategy.daily_pnl)
                
                # Check for exit if in position
                if self.strategy.current_position:
                    should_exit, reason, price = self.strategy.check_exit_conditions(
                        current_data, self.strategy.current_position
                    )
                    if should_exit:
                        self.strategy.update_position(True, price, reason)
                        trades.append({
                            'exit_time': current_time,
                            'exit_price': price,
                            'pnl': self.strategy.daily_pnl
                        })
                
                # Check for entry if not in position
                elif self.strategy.can_open_position():
                    should_enter, signal_type, price = self.strategy.check_entry_conditions(current_data)
                    if should_enter:
                        self.strategy.current_position = {
                            'type': signal_type,
                            'entry_price': price,
                            'entry_time': current_time
                        }
                        self.strategy.trades_today += 1
                        trades.append({
                            'entry_time': current_time,
                            'entry_price': price,
                            'type': signal_type
                        })
            
            # Calculate backtest metrics
            total_trades = len([t for t in trades if 'exit_price' in t])
            winning_trades = len([t for t in trades if 'pnl' in t and t['pnl'] > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_profit = sum([t['pnl'] for t in trades if 'pnl' in t])
            max_drawdown = min([0] + [t['pnl'] for t in trades if 'pnl' in t])
            avg_daily_profit = total_profit / days if days > 0 else 0
            trading_days = len([pnl for pnl in daily_pnl if pnl != 0])
            daily_win_rate = (trading_days / days * 100) if days > 0 else 0
            
            # Print results
            console.print("\n[bold cyan]Backtest Results[/bold cyan]")
            results = Table(show_header=True, header_style="bold magenta")
            results.add_column("Metric", style="cyan")
            results.add_column("Value", style="green")
            
            results.add_row("Total Trades", str(total_trades))
            results.add_row("Win Rate", f"{win_rate:.2f}%")
            results.add_row("Total Profit", f"${total_profit:.2f}")
            results.add_row("Max Drawdown", f"{max_drawdown:.2f}%")
            results.add_row("Avg Daily Profit", f"${avg_daily_profit:.2f}")
            results.add_row("Daily Win Rate", f"{daily_win_rate:.2f}%")
            
            console.print(results)
            
            # Create visualization
            console.print("\nCreating visualization...")
            self.visualizer.plot_backtest_results(df, trades)
            console.print("[bold green]Backtest completed! Results saved to backtest_results.png[/bold green]")
            
        except Exception as e:
            logger.error(f"Backtest error: {str(e)}")
            console.print(f"[bold red]Error during backtest: {str(e)}[/bold red]")

    def run_live(self):
        """Run live trading"""
        self.running = True
        
        with Live(self.layout, refresh_per_second=1) as live:
            while self.running:
                try:
                    # Get latest data
                    data = self.data_fetcher.get_latest_data()
                    
                    # Check for exit if in position
                    if self.strategy.current_position:
                        should_exit, reason, price = self.strategy.check_exit_conditions(
                            data, self.strategy.current_position
                        )
                        if should_exit:
                            # Execute exit order
                            order = self.client.create_order(
                                'sell' if self.strategy.current_position['type'] == 'LONG' else 'buy',
                                self.strategy.position_size,
                                price
                            )
                            if order:
                                self.strategy.update_position(True, price, reason)
                    
                    # Check for entry if not in position
                    elif self.strategy.can_open_position():
                        should_enter, signal_type, price = self.strategy.check_entry_conditions(data)
                        if should_enter:
                            # Execute entry order
                            order = self.client.create_order(
                                'buy' if signal_type == 'LONG' else 'sell',
                                self.strategy.position_size,
                                price
                            )
                            if order:
                                self.strategy.current_position = {
                                    'type': signal_type,
                                    'entry_price': price,
                                    'entry_time': datetime.now()
                                }
                                self.strategy.trades_today += 1
                    
                    # Update dashboard
                    self.update_dashboard(live)
                    
                    # Sleep until next update
                    time.sleep(config.UPDATE_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"Error in live trading loop: {str(e)}")
                    time.sleep(config.UPDATE_INTERVAL)

    def stop(self):
        """Stop the trading bot"""
        self.running = False

@click.group()
def cli():
    """Bitcoin Scalping Trading Bot"""
    pass

@cli.command()
@click.option('--days', default=30, help='Number of days to backtest')
def backtest(days):
    """Run backtest simulation"""
    bot = TradingBot()
    bot.run_backtest(days)

@cli.command()
def live():
    """Run live trading"""
    bot = TradingBot()
    try:
        bot.run_live()
    except KeyboardInterrupt:
        bot.stop()
        console.print("[bold red]Bot stopped by user[/bold red]")

if __name__ == '__main__':
    cli() 