import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from pathlib import Path

class WhaleDetectorAgent:
    """Whale Detection Agent for large transaction monitoring"""
    
    def __init__(self, config_manager, data_manager, cache_manager):
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        # Agent state
        self.active = False
        self.last_update = None
        self.processed_count = 0
        self.error_count = 0
        
        # Whale detection data
        self.whale_activities = {}
        self.whale_alerts = []
        self.whale_statistics = {}
        self._lock = threading.Lock()
        
        # Detection thresholds
        self.volume_threshold_multiplier = 5.0  # 5x normal volume
        self.large_trade_threshold = 1000000  # $1M USD
        self.unusual_activity_threshold = 3.0  # 3 standard deviations
        
        # Whale data path
        self.whale_path = Path("whale_data")
        self.whale_path.mkdir(exist_ok=True)
        
        # Start agent if enabled
        if self.config_manager.get("agents", {}).get("whale_detector", {}).get("enabled", True):
            self.start()
    
    def start(self):
        """Start the whale detector agent"""
        if not self.active:
            self.active = True
            self.agent_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.agent_thread.start()
            self.logger.info("Whale Detector Agent started")
    
    def stop(self):
        """Stop the whale detector agent"""
        self.active = False
        self.logger.info("Whale Detector Agent stopped")
    
    def _detection_loop(self):
        """Main whale detection loop"""
        while self.active:
            try:
                # Get update interval from config
                interval = self.config_manager.get("agents", {}).get("whale_detector", {}).get("update_interval", 180)
                
                # Detect whale activities
                self._detect_whale_activities()
                
                # Analyze whale patterns
                self._analyze_whale_patterns()
                
                # Update statistics
                self._update_whale_statistics()
                
                # Update last update time
                self.last_update = datetime.now()
                
                # Sleep until next detection
                time.sleep(interval)
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Whale detection error: {str(e)}")
                time.sleep(60)  # Sleep 1 minute on error
    
    def _detect_whale_activities(self):
        """Detect whale activities across all monitored symbols"""
        try:
            symbols = self.data_manager.get_supported_symbols()
            
            for symbol in symbols[:100]:  # Monitor top 100 symbols
                try:
                    whale_activity = self._analyze_symbol_for_whales(symbol)
                    if whale_activity:
                        self._process_whale_activity(symbol, whale_activity)
                        self.processed_count += 1
                
                except Exception as e:
                    self.logger.error(f"Error detecting whales for {symbol}: {str(e)}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Whale detection error: {str(e)}")
    
    def _analyze_symbol_for_whales(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a symbol for whale activity"""
        try:
            # Get current market data
            current_data = self.data_manager.get_market_data(symbol=symbol)
            if current_data is None or current_data.empty:
                return None
            
            # Get historical data for baseline comparison
            historical_data = self.data_manager.get_historical_data(symbol, days=7)
            if historical_data is None or len(historical_data) < 20:
                return None
            
            # Analyze volume patterns
            volume_analysis = self._analyze_volume_patterns(current_data, historical_data)
            
            # Analyze price movements
            price_analysis = self._analyze_price_movements(current_data, historical_data)
            
            # Detect large transactions
            large_transactions = self._detect_large_transactions(current_data, historical_data)
            
            # Combine analyses
            whale_activity = self._combine_whale_indicators(
                symbol, volume_analysis, price_analysis, large_transactions
            )
            
            return whale_activity
            
        except Exception as e:
            self.logger.error(f"Whale analysis error for {symbol}: {str(e)}")
            return None
    
    def _analyze_volume_patterns(self, current_data: pd.DataFrame, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns for whale activity"""
        try:
            current_volume = current_data.iloc[0].get('volume', 0)
            
            # Calculate historical volume statistics
            historical_volumes = historical_data['volume'].values
            avg_volume = np.mean(historical_volumes)
            std_volume = np.std(historical_volumes)
            
            # Volume spike detection
            volume_multiplier = current_volume / avg_volume if avg_volume > 0 else 0
            volume_z_score = (current_volume - avg_volume) / std_volume if std_volume > 0 else 0
            
            # Determine volume anomaly level
            if volume_multiplier > self.volume_threshold_multiplier:
                anomaly_level = 'high'
            elif volume_z_score > 2.0:
                anomaly_level = 'medium'
            elif volume_z_score > 1.5:
                anomaly_level = 'low'
            else:
                anomaly_level = 'normal'
            
            return {
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_multiplier': volume_multiplier,
                'volume_z_score': volume_z_score,
                'anomaly_level': anomaly_level,
                'is_unusual': volume_multiplier > self.volume_threshold_multiplier
            }
            
        except Exception as e:
            self.logger.error(f"Volume analysis error: {str(e)}")
            return {}
    
    def _analyze_price_movements(self, current_data: pd.DataFrame, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price movements for whale signatures"""
        try:
            current_price = current_data.iloc[0].get('price', 0)
            
            # Calculate price statistics
            historical_prices = historical_data['price'].values
            price_changes = np.diff(historical_prices) / historical_prices[:-1]
            
            # Get recent price change
            if len(historical_prices) > 0:
                recent_change = (current_price - historical_prices[-1]) / historical_prices[-1]
            else:
                recent_change = 0
            
            # Calculate volatility metrics
            price_volatility = np.std(price_changes) if len(price_changes) > 0 else 0
            z_score = abs(recent_change) / price_volatility if price_volatility > 0 else 0
            
            # Detect unusual price movements
            if abs(recent_change) > 0.1:  # 10% price change
                movement_significance = 'high'
            elif z_score > 2.0:
                movement_significance = 'medium'
            elif z_score > 1.5:
                movement_significance = 'low'
            else:
                movement_significance = 'normal'
            
            return {
                'current_price': current_price,
                'recent_change': recent_change,
                'price_volatility': price_volatility,
                'z_score': z_score,
                'movement_significance': movement_significance,
                'is_unusual': abs(recent_change) > 0.05  # 5% threshold
            }
            
        except Exception as e:
            self.logger.error(f"Price analysis error: {str(e)}")
            return {}
    
    def _detect_large_transactions(self, current_data: pd.DataFrame, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect large transactions based on price and volume"""
        try:
            current_row = current_data.iloc[0]
            current_price = current_row.get('price', 0)
            current_volume = current_row.get('volume', 0)
            
            # Calculate transaction value
            transaction_value = current_price * current_volume
            
            # Historical transaction values
            historical_values = historical_data['price'] * historical_data['volume']
            avg_transaction_value = np.mean(historical_values)
            
            # Large transaction detection
            is_large_transaction = transaction_value > self.large_trade_threshold
            value_multiplier = transaction_value / avg_transaction_value if avg_transaction_value > 0 else 0
            
            # Classify transaction size
            if transaction_value > 10000000:  # $10M
                transaction_class = 'mega_whale'
            elif transaction_value > 5000000:  # $5M
                transaction_class = 'large_whale'
            elif transaction_value > 1000000:  # $1M
                transaction_class = 'whale'
            elif value_multiplier > 5:
                transaction_class = 'unusual_large'
            else:
                transaction_class = 'normal'
            
            return {
                'transaction_value': transaction_value,
                'avg_transaction_value': avg_transaction_value,
                'value_multiplier': value_multiplier,
                'is_large_transaction': is_large_transaction,
                'transaction_class': transaction_class
            }
            
        except Exception as e:
            self.logger.error(f"Large transaction detection error: {str(e)}")
            return {}
    
    def _combine_whale_indicators(self, symbol: str, volume_analysis: Dict, 
                                 price_analysis: Dict, transaction_analysis: Dict) -> Optional[Dict[str, Any]]:
        """Combine all whale indicators into a unified assessment"""
        try:
            timestamp = datetime.now()
            
            # Calculate whale probability score
            whale_score = 0
            confidence_factors = []
            
            # Volume contribution
            if volume_analysis.get('is_unusual', False):
                volume_weight = min(0.4, volume_analysis.get('volume_multiplier', 0) / 10)
                whale_score += volume_weight
                confidence_factors.append('unusual_volume')
            
            # Price movement contribution
            if price_analysis.get('is_unusual', False):
                price_weight = min(0.3, abs(price_analysis.get('recent_change', 0)) * 5)
                whale_score += price_weight
                confidence_factors.append('unusual_price_movement')
            
            # Transaction size contribution
            if transaction_analysis.get('is_large_transaction', False):
                trans_weight = min(0.5, transaction_analysis.get('value_multiplier', 0) / 20)
                whale_score += trans_weight
                confidence_factors.append('large_transaction')
            
            # Determine whale activity level
            if whale_score > 0.7:
                activity_level = 'high'
                alert_level = 'critical'
            elif whale_score > 0.4:
                activity_level = 'medium'
                alert_level = 'warning'
            elif whale_score > 0.2:
                activity_level = 'low'
                alert_level = 'info'
            else:
                activity_level = 'none'
                alert_level = 'none'
            
            # Only return activity if there's something significant
            if whale_score > 0.2:
                return {
                    'timestamp': timestamp.isoformat(),
                    'symbol': symbol,
                    'whale_score': whale_score,
                    'activity_level': activity_level,
                    'alert_level': alert_level,
                    'confidence_factors': confidence_factors,
                    'volume_analysis': volume_analysis,
                    'price_analysis': price_analysis,
                    'transaction_analysis': transaction_analysis,
                    'estimated_whale_type': transaction_analysis.get('transaction_class', 'unknown')
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Whale indicator combination error: {str(e)}")
            return None
    
    def _process_whale_activity(self, symbol: str, activity: Dict[str, Any]):
        """Process and store detected whale activity"""
        with self._lock:
            # Store activity
            if symbol not in self.whale_activities:
                self.whale_activities[symbol] = []
            
            self.whale_activities[symbol].append(activity)
            
            # Keep only last 48 hours of activities
            cutoff_time = datetime.now() - timedelta(hours=48)
            self.whale_activities[symbol] = [
                act for act in self.whale_activities[symbol]
                if datetime.fromisoformat(act['timestamp']) > cutoff_time
            ]
            
            # Generate alert if significant
            if activity['alert_level'] in ['warning', 'critical']:
                self._generate_whale_alert(symbol, activity)
    
    def _generate_whale_alert(self, symbol: str, activity: Dict[str, Any]):
        """Generate whale activity alert"""
        alert = {
            'timestamp': activity['timestamp'],
            'symbol': symbol,
            'alert_level': activity['alert_level'],
            'whale_score': activity['whale_score'],
            'activity_level': activity['activity_level'],
            'estimated_whale_type': activity['estimated_whale_type'],
            'message': self._create_alert_message(symbol, activity),
            'confidence_factors': activity['confidence_factors']
        }
        
        self.whale_alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.whale_alerts) > 100:
            self.whale_alerts = self.whale_alerts[-100:]
        
        # Log the alert
        self.logger.warning(f"Whale alert: {alert['message']}")
    
    def _create_alert_message(self, symbol: str, activity: Dict[str, Any]) -> str:
        """Create human-readable alert message"""
        whale_type = activity['estimated_whale_type']
        score = activity['whale_score']
        factors = activity['confidence_factors']
        
        # Get specific metrics
        volume_mult = activity.get('volume_analysis', {}).get('volume_multiplier', 0)
        price_change = activity.get('price_analysis', {}).get('recent_change', 0)
        trans_value = activity.get('transaction_analysis', {}).get('transaction_value', 0)
        
        message_parts = [f"{whale_type.replace('_', ' ').title()} detected on {symbol}"]
        
        if 'unusual_volume' in factors:
            message_parts.append(f"Volume: {volume_mult:.1f}x normal")
        
        if 'unusual_price_movement' in factors:
            direction = "up" if price_change > 0 else "down"
            message_parts.append(f"Price moved {abs(price_change)*100:.1f}% {direction}")
        
        if 'large_transaction' in factors:
            message_parts.append(f"Transaction value: ${trans_value:,.0f}")
        
        message_parts.append(f"Whale score: {score:.2f}")
        
        return " | ".join(message_parts)
    
    def _analyze_whale_patterns(self):
        """Analyze patterns in whale activities"""
        try:
            with self._lock:
                if not self.whale_activities:
                    return
                
                # Analyze recent whale patterns
                recent_activities = []
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                for symbol, activities in self.whale_activities.items():
                    for activity in activities:
                        if datetime.fromisoformat(activity['timestamp']) > cutoff_time:
                            recent_activities.append(activity)
                
                if not recent_activities:
                    return
                
                # Pattern analysis
                self._analyze_whale_clustering(recent_activities)
                self._analyze_whale_market_impact(recent_activities)
                
        except Exception as e:
            self.logger.error(f"Whale pattern analysis error: {str(e)}")
    
    def _analyze_whale_clustering(self, activities: List[Dict[str, Any]]):
        """Analyze clustering of whale activities"""
        try:
            # Group activities by time windows
            time_windows = {}
            
            for activity in activities:
                timestamp = datetime.fromisoformat(activity['timestamp'])
                window_key = timestamp.replace(minute=0, second=0, microsecond=0)
                
                if window_key not in time_windows:
                    time_windows[window_key] = []
                
                time_windows[window_key].append(activity)
            
            # Detect clustering
            clustered_windows = {k: v for k, v in time_windows.items() if len(v) > 3}
            
            if clustered_windows:
                self.logger.info(f"Detected whale clustering in {len(clustered_windows)} time windows")
                
        except Exception as e:
            self.logger.error(f"Whale clustering analysis error: {str(e)}")
    
    def _analyze_whale_market_impact(self, activities: List[Dict[str, Any]]):
        """Analyze market impact of whale activities"""
        try:
            # Group by symbols
            symbol_impacts = {}
            
            for activity in activities:
                symbol = activity['symbol']
                if symbol not in symbol_impacts:
                    symbol_impacts[symbol] = []
                
                symbol_impacts[symbol].append(activity)
            
            # Analyze impact for each symbol
            for symbol, symbol_activities in symbol_impacts.items():
                if len(symbol_activities) > 1:
                    # Multiple whale activities on same symbol
                    total_score = sum(act['whale_score'] for act in symbol_activities)
                    if total_score > 2.0:
                        self.logger.info(f"High whale activity concentration on {symbol}: {total_score:.2f}")
                        
        except Exception as e:
            self.logger.error(f"Whale market impact analysis error: {str(e)}")
    
    def _update_whale_statistics(self):
        """Update whale detection statistics"""
        try:
            with self._lock:
                total_activities = sum(len(activities) for activities in self.whale_activities.values())
                
                # Count by whale type
                whale_type_counts = {}
                alert_level_counts = {}
                
                for activities in self.whale_activities.values():
                    for activity in activities:
                        whale_type = activity.get('estimated_whale_type', 'unknown')
                        alert_level = activity.get('alert_level', 'none')
                        
                        whale_type_counts[whale_type] = whale_type_counts.get(whale_type, 0) + 1
                        alert_level_counts[alert_level] = alert_level_counts.get(alert_level, 0) + 1
                
                self.whale_statistics = {
                    'timestamp': datetime.now().isoformat(),
                    'total_activities': total_activities,
                    'monitored_symbols': len(self.whale_activities),
                    'recent_alerts': len(self.whale_alerts),
                    'whale_type_distribution': whale_type_counts,
                    'alert_level_distribution': alert_level_counts
                }
                
        except Exception as e:
            self.logger.error(f"Whale statistics update error: {str(e)}")
    
    def get_whale_activities(self, symbol: str = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get whale activities for a symbol or all symbols"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            if symbol:
                activities = self.whale_activities.get(symbol, [])
                return [
                    act for act in activities
                    if datetime.fromisoformat(act['timestamp']) > cutoff_time
                ]
            else:
                all_activities = []
                for activities in self.whale_activities.values():
                    all_activities.extend([
                        act for act in activities
                        if datetime.fromisoformat(act['timestamp']) > cutoff_time
                    ])
                
                # Sort by timestamp, most recent first
                all_activities.sort(key=lambda x: x['timestamp'], reverse=True)
                return all_activities
    
    def get_whale_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent whale alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.whale_alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
    
    def get_whale_summary(self) -> Dict[str, Any]:
        """Get whale detection summary"""
        with self._lock:
            recent_activities = self.get_whale_activities(hours=24)
            recent_alerts = self.get_whale_alerts(hours=24)
            
            # Count by activity level
            activity_levels = {}
            for activity in recent_activities:
                level = activity.get('activity_level', 'none')
                activity_levels[level] = activity_levels.get(level, 0) + 1
            
            # Count by alert level
            alert_levels = {}
            for alert in recent_alerts:
                level = alert.get('alert_level', 'none')
                alert_levels[level] = alert_levels.get(level, 0) + 1
            
            return {
                'total_activities_24h': len(recent_activities),
                'total_alerts_24h': len(recent_alerts),
                'activity_level_distribution': activity_levels,
                'alert_level_distribution': alert_levels,
                'monitored_symbols': len(self.whale_activities),
                'statistics': self.whale_statistics,
                'last_update': self.last_update.isoformat() if self.last_update else None
            }
    
    def get_top_whale_symbols(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get symbols with highest whale activity"""
        with self._lock:
            symbol_scores = {}
            
            for symbol, activities in self.whale_activities.items():
                recent_activities = [
                    act for act in activities
                    if datetime.fromisoformat(act['timestamp']) > datetime.now() - timedelta(hours=24)
                ]
                
                if recent_activities:
                    total_score = sum(act['whale_score'] for act in recent_activities)
                    avg_score = total_score / len(recent_activities)
                    
                    symbol_scores[symbol] = {
                        'symbol': symbol,
                        'total_score': total_score,
                        'avg_score': avg_score,
                        'activity_count': len(recent_activities),
                        'latest_activity': recent_activities[0]['timestamp']
                    }
            
            # Sort by total score
            top_symbols = sorted(symbol_scores.values(), key=lambda x: x['total_score'], reverse=True)
            
            return top_symbols[:limit]
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            'active': self.active,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'monitored_symbols': len(self.whale_activities),
            'total_activities': sum(len(activities) for activities in self.whale_activities.values()),
            'total_alerts': len(self.whale_alerts)
        }
