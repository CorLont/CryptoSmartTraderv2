#!/usr/bin/env python3
"""
Async Data Updater - Background data refresh for Streamlit dashboard
"""

import asyncio
import threading
import time
import logging
from typing import Dict, Any, Callable, List
from datetime import datetime, timedelta
import streamlit as st


class AsyncDataUpdater:
    """
    Asynchronous data updater for Streamlit dashboard
    
    Features:
    - Background data refresh
    - Non-blocking updates
    - Configurable refresh intervals
    - Error handling and retry logic
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.update_thread = None
        self.is_running = False
        self.update_tasks = []
        self.last_update_times = {}
        
        # Update intervals (seconds)
        self.update_intervals = {
            'market_data': 30,
            'portfolio_data': 60,
            'agent_status': 45,
            'system_health': 120,
            'trading_signals': 15
        }
    
    def start_background_updates(self):
        """Start background update thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self.update_thread.start()
        self.logger.info("Background update thread started")
    
    def stop_background_updates(self):
        """Stop background update thread"""
        self.is_running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
        self.logger.info("Background update thread stopped")
    
    def _update_loop(self):
        """Main update loop running in background thread"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if auto-refresh is enabled
                if not st.session_state.get('auto_refresh', True):
                    time.sleep(5)
                    continue
                
                # Update market data
                if self._should_update('market_data', current_time):
                    self._update_market_data()
                    self.last_update_times['market_data'] = current_time
                
                # Update portfolio data
                if self._should_update('portfolio_data', current_time):
                    self._update_portfolio_data()
                    self.last_update_times['portfolio_data'] = current_time
                
                # Update agent status
                if self._should_update('agent_status', current_time):
                    self._update_agent_status()
                    self.last_update_times['agent_status'] = current_time
                
                # Update system health
                if self._should_update('system_health', current_time):
                    self._update_system_health()
                    self.last_update_times['system_health'] = current_time
                
                # Update trading signals
                if self._should_update('trading_signals', current_time):
                    self._update_trading_signals()
                    self.last_update_times['trading_signals'] = current_time
                
                # Sleep for a short interval
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _should_update(self, data_type: str, current_time: float) -> bool:
        """Check if data type should be updated based on interval"""
        if data_type not in self.update_intervals:
            return False
        
        last_update = self.last_update_times.get(data_type, 0)
        interval = self.update_intervals[data_type]
        
        # Respect user's refresh interval setting
        user_interval = st.session_state.get('refresh_interval', 30)
        effective_interval = max(interval, user_interval)
        
        return (current_time - last_update) >= effective_interval
    
    def _update_market_data(self):
        """Update market data in background"""
        try:
            # Import here to avoid circular imports
            from dashboards.utils.cache_manager import cache_manager
            
            # Clear market data cache to force refresh
            symbols = st.session_state.get('market_overview_symbols', ['BTC/USD'])
            
            # This will trigger a cache miss and fresh data fetch
            # Note: In a real implementation, this would call the actual API
            market_data = cache_manager.get_market_data(symbols)
            
            # Update session state with new data
            if not market_data.empty:
                st.session_state.market_data_last_update = datetime.utcnow()
                self.logger.debug("Market data updated in background")
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    def _update_portfolio_data(self):
        """Update portfolio data in background"""
        try:
            from dashboards.utils.cache_manager import cache_manager
            
            portfolio_data = cache_manager.get_portfolio_data()
            
            # Update session state
            st.session_state.portfolio_value = portfolio_data.get('total_value', 0)
            st.session_state.daily_pnl = portfolio_data.get('daily_pnl', 0)
            st.session_state.active_positions = len(portfolio_data.get('positions', []))
            
            self.logger.debug("Portfolio data updated in background")
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio data: {e}")
    
    def _update_agent_status(self):
        """Update agent status in background"""
        try:
            from dashboards.utils.cache_manager import cache_manager
            
            agent_data = cache_manager.get_agent_status()
            
            # Update session state
            agents = agent_data.get('agents', [])
            healthy_agents = sum(1 for agent in agents if agent['status'] == 'healthy')
            
            st.session_state.healthy_agents = healthy_agents
            st.session_state.total_agents = len(agents)
            st.session_state.agents_last_update = datetime.utcnow()
            
            self.logger.debug("Agent status updated in background")
            
        except Exception as e:
            self.logger.error(f"Error updating agent status: {e}")
    
    def _update_system_health(self):
        """Update system health in background"""
        try:
            from dashboards.utils.cache_manager import cache_manager
            
            health_data = cache_manager.get_system_health()
            
            # Update session state
            st.session_state.system_health_score = health_data.get('overall_score', 0)
            st.session_state.system_health_grade = health_data.get('grade', 'F')
            st.session_state.trading_enabled = health_data.get('trading_enabled', False)
            
            self.logger.debug("System health updated in background")
            
        except Exception as e:
            self.logger.error(f"Error updating system health: {e}")
    
    def _update_trading_signals(self):
        """Update trading signals in background"""
        try:
            # Simulate signal updates
            import random
            
            current_signals = st.session_state.get('signals_today', 0)
            
            # Randomly generate new signals
            if random.random() < 0.1:  # 10% chance of new signal
                st.session_state.signals_today = current_signals + 1
                
                # Add to recent signals list
                new_signal = {
                    'timestamp': datetime.utcnow(),
                    'symbol': random.choice(['BTC/USD', 'ETH/USD', 'BNB/USD']),
                    'type': random.choice(['buy', 'sell']),
                    'confidence': random.uniform(0.7, 0.95)
                }
                
                recent_signals = st.session_state.get('recent_signals', [])
                recent_signals.append(new_signal)
                
                # Keep only last 10 signals
                st.session_state.recent_signals = recent_signals[-10:]
                
                self.logger.debug(f"New trading signal generated: {new_signal}")
            
        except Exception as e:
            self.logger.error(f"Error updating trading signals: {e}")
    
    def trigger_refresh(self):
        """Manually trigger data refresh"""
        try:
            # Reset update times to force immediate updates
            current_time = time.time()
            for data_type in self.update_intervals.keys():
                self.last_update_times[data_type] = 0
            
            self.logger.info("Manual data refresh triggered")
            
        except Exception as e:
            self.logger.error(f"Error triggering refresh: {e}")
    
    def set_update_interval(self, data_type: str, interval_seconds: int):
        """Set custom update interval for data type"""
        if data_type in self.update_intervals:
            self.update_intervals[data_type] = interval_seconds
            self.logger.info(f"Update interval for {data_type} set to {interval_seconds}s")
    
    def get_update_status(self) -> Dict[str, Any]:
        """Get status of background updates"""
        current_time = time.time()
        
        status = {
            'is_running': self.is_running,
            'thread_alive': self.update_thread.is_alive() if self.update_thread else False,
            'auto_refresh_enabled': st.session_state.get('auto_refresh', False),
            'last_updates': {}
        }
        
        for data_type, last_update in self.last_update_times.items():
            if last_update > 0:
                seconds_ago = int(current_time - last_update)
                status['last_updates'][data_type] = {
                    'seconds_ago': seconds_ago,
                    'last_update': datetime.fromtimestamp(last_update).isoformat()
                }
        
        return status
    
    def add_custom_updater(self, name: str, update_func: Callable, interval_seconds: int):
        """Add custom update function"""
        self.update_intervals[name] = interval_seconds
        # Store function reference for custom updates
        # This would need more sophisticated implementation for real use
        self.logger.info(f"Custom updater '{name}' added with {interval_seconds}s interval")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_background_updates()


# Global async updater instance
async_updater = AsyncDataUpdater()