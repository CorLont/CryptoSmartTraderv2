import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from pathlib import Path


class TradeExecutorAgent:
    """Trade Executor Agent for signal generation and risk management"""

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

        # Trading state
        self.trading_signals = {}
        self.portfolio_positions = {}
        self.risk_metrics = {}
        self.trade_history = []
        self._lock = threading.Lock()

        # Risk management parameters
        self.max_position_size = 0.1  # 10% of portfolio per position
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.15  # 15% take profit
        self.max_portfolio_risk = 0.2  # 20% max portfolio risk

        # Trading history path
        self.trades_path = Path("trades")
        self.trades_path.mkdir(exist_ok=True)

        # Start agent if enabled
        if self.config_manager.get("agents", {}).get("trade_executor", {}).get("enabled", True):
            self.start()

    def start(self):
        """Start the trade executor agent"""
        if not self.active:
            self.active = True
            self.agent_thread = threading.Thread(target=self._execution_loop, daemon=True)
            self.agent_thread.start()
            self.logger.info("Trade Executor Agent started")

    def stop(self):
        """Stop the trade executor agent"""
        self.active = False
        self.logger.info("Trade Executor Agent stopped")

    def _execution_loop(self):
        """Main execution loop"""
        while self.active:
            try:
                # Get update interval from config
                interval = (
                    self.config_manager.get("agents", {})
                    .get("trade_executor", {})
                    .get("update_interval", 120)
                )

                # Generate trading signals
                self._generate_trading_signals()

                # Evaluate portfolio risk
                self._evaluate_portfolio_risk()

                # Execute risk management
                self._execute_risk_management()

                # Update last update time
                self.last_update = datetime.now()

                # Sleep until next execution
                time.sleep(interval)

            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Trade execution error: {str(e)}")
                time.sleep(60)  # Sleep 1 minute on error

    def _generate_trading_signals(self):
        """Generate trading signals based on all available analysis"""
        try:
            symbols = self.data_manager.get_supported_symbols()

            for symbol in symbols[:50]:  # Limit for efficiency
                try:
                    signal = self._analyze_symbol_for_trading(symbol)
                    if signal:
                        self._process_trading_signal(symbol, signal)
                        self.processed_count += 1

                except Exception as e:
                    self.logger.error(f"Error generating signal for {symbol}: {str(e)}")
                    continue

        except Exception as e:
            self.logger.error(f"Signal generation error: {str(e)}")

    def _analyze_symbol_for_trading(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a symbol for trading opportunities"""
        try:
            # Get various analysis results
            technical_analysis = self._get_technical_analysis(symbol)
            sentiment_analysis = self._get_sentiment_analysis(symbol)
            ml_prediction = self._get_ml_prediction(symbol)
            backtest_results = self._get_backtest_results(symbol)

            # Combine all signals
            signal = self._combine_signals(
                symbol, technical_analysis, sentiment_analysis, ml_prediction, backtest_results
            )

            return signal

        except Exception as e:
            self.logger.error(f"Symbol analysis error for {symbol}: {str(e)}")
            return None

    def _get_technical_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get technical analysis from cache or default"""
        cache_key = f"technical_{symbol.replace('/', '_')}"
        technical_data = self.cache_manager.get(cache_key)

        if technical_data:
            return technical_data.get("signal", {})

        return {"signal": "hold", "confidence": 0.1, "strength": 0.5}

    def _get_sentiment_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment analysis from cache or default"""
        base_currency = symbol.split("/")[0] if "/" in symbol else symbol
        cache_key = f"sentiment_{base_currency.lower()}"
        sentiment_data = self.cache_manager.get(cache_key)

        if sentiment_data:
            return {
                "score": sentiment_data.get("composite_score", 0.5),
                "category": sentiment_data.get("category", "neutral"),
                "confidence": sentiment_data.get("confidence", 0.1),
            }

        return {"score": 0.5, "category": "neutral", "confidence": 0.1}

    def _get_ml_prediction(self, symbol: str) -> Dict[str, Any]:
        """Get ML prediction from cache or default"""
        cache_key = f"ml_prediction_{symbol.replace('/', '_')}"
        ml_data = self.cache_manager.get(cache_key)

        if ml_data:
            predictions = ml_data.get("predictions", {})
            if "1d" in predictions:
                pred = predictions["1d"]
                return {
                    "predicted_change": pred.get("predicted_change_percent", 0),
                    "confidence": pred.get("confidence", 0.1),
                }

        return {"predicted_change": 0, "confidence": 0.1}

    def _get_backtest_results(self, symbol: str) -> Dict[str, Any]:
        """Get backtest results from cache or default"""
        # This would get results from the backtest agent
        # For now, return default values
        return {
            "best_strategy": "moving_average_crossover",
            "avg_return": 0.05,
            "sharpe_ratio": 0.8,
            "win_rate": 0.6,
        }

    def _combine_signals(
        self, symbol: str, technical: Dict, sentiment: Dict, ml_pred: Dict, backtest: Dict
    ) -> Dict[str, Any]:
        """Combine all signals into a unified trading signal"""
        timestamp = datetime.now()

        # Extract individual signals
        tech_signal = technical.get("signal", "hold")
        tech_confidence = technical.get("confidence", 0.1)

        sent_score = sentiment.get("score", 0.5)
        sent_confidence = sentiment.get("confidence", 0.1)

        ml_change = ml_pred.get("predicted_change", 0)
        ml_confidence = ml_pred.get("confidence", 0.1)

        # Convert signals to numeric scores
        tech_score = 1 if tech_signal == "buy" else -1 if tech_signal == "sell" else 0
        sent_score_norm = (sent_score - 0.5) * 2  # Convert 0-1 to -1 to 1
        ml_score = np.clip(ml_change / 10, -1, 1)  # Normalize ML prediction

        # Weighted combination
        weights = {"technical": 0.4, "sentiment": 0.3, "ml": 0.3}

        combined_score = (
            tech_score * weights["technical"] * tech_confidence
            + sent_score_norm * weights["sentiment"] * sent_confidence
            + ml_score * weights["ml"] * ml_confidence
        )

        # Calculate overall confidence
        overall_confidence = (
            tech_confidence * weights["technical"]
            + sent_confidence * weights["sentiment"]
            + ml_confidence * weights["ml"]
        )

        # Determine final signal
        if combined_score > 0.2 and overall_confidence > 0.3:
            final_signal = "buy"
        elif combined_score < -0.2 and overall_confidence > 0.3:
            final_signal = "sell"
        else:
            final_signal = "hold"

        # Calculate signal strength
        signal_strength = abs(combined_score)

        # Risk adjustment based on backtest results
        risk_adjustment = min(1.0, backtest.get("sharpe_ratio", 0.5) / 1.0)

        return {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "signal": final_signal,
            "strength": signal_strength,
            "confidence": overall_confidence,
            "combined_score": combined_score,
            "risk_adjustment": risk_adjustment,
            "components": {
                "technical": {"signal": tech_signal, "confidence": tech_confidence},
                "sentiment": {"score": sent_score, "confidence": sent_confidence},
                "ml": {"predicted_change": ml_change, "confidence": ml_confidence},
            },
            "backtest_metrics": backtest,
        }

    def _process_trading_signal(self, symbol: str, signal: Dict[str, Any]):
        """Process and store trading signal"""
        with self._lock:
            if symbol not in self.trading_signals:
                self.trading_signals[symbol] = []

            self.trading_signals[symbol].append(signal)

            # Keep only last 24 hours of signals
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.trading_signals[symbol] = [
                s
                for s in self.trading_signals[symbol]
                if datetime.fromisoformat(s["timestamp"]) > cutoff_time
            ]

    def _evaluate_portfolio_risk(self):
        """Evaluate current portfolio risk metrics"""
        try:
            with self._lock:
                # Calculate portfolio metrics
                total_positions = len(self.portfolio_positions)
                total_exposure = sum(
                    pos.get("value", 0) for pos in self.portfolio_positions.values()
                )

                # Calculate risk metrics
                portfolio_var = self._calculate_portfolio_var()
                correlation_risk = self._calculate_correlation_risk()
                concentration_risk = self._calculate_concentration_risk()

                self.risk_metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "total_positions": total_positions,
                    "total_exposure": total_exposure,
                    "portfolio_var": portfolio_var,
                    "correlation_risk": correlation_risk,
                    "concentration_risk": concentration_risk,
                    "risk_score": (portfolio_var + correlation_risk + concentration_risk) / 3,
                }

        except Exception as e:
            self.logger.error(f"Portfolio risk evaluation error: {str(e)}")

    def _calculate_portfolio_var(self) -> float:
        """Calculate portfolio Value at Risk"""
        try:
            if not self.portfolio_positions:
                return 0.0

            # Simple VaR calculation based on historical volatility
            total_risk = 0
            for symbol, position in self.portfolio_positions.items():
                # Get historical data to calculate volatility
                historical_data = self.data_manager.get_historical_data(symbol, days=30)

                if historical_data is not None and len(historical_data) > 1:
                    returns = historical_data["price"].pct_change().dropna()
                    volatility = returns.std()
                    position_value = position.get("value", 0)
                    position_var = position_value * volatility * 2.33  # 99% confidence
                    total_risk += position_var**2

            return np.sqrt(total_risk)

        except Exception:
            return 0.5  # Default moderate risk

    def _calculate_correlation_risk(self) -> float:
        """Calculate correlation risk among positions"""
        try:
            if len(self.portfolio_positions) < 2:
                return 0.0

            # Simple correlation risk based on sector concentration
            # In a real implementation, this would calculate actual correlations
            symbols = list(self.portfolio_positions.keys())
            crypto_pairs = sum(1 for s in symbols if "/" in s)

            # Higher correlation risk if too many similar assets
            correlation_risk = min(1.0, crypto_pairs / 10)

            return correlation_risk

        except Exception:
            return 0.3  # Default moderate correlation risk

    def _calculate_concentration_risk(self) -> float:
        """Calculate concentration risk"""
        try:
            if not self.portfolio_positions:
                return 0.0

            total_value = sum(pos.get("value", 0) for pos in self.portfolio_positions.values())

            if total_value == 0:
                return 0.0

            # Calculate concentration using Herfindahl index
            concentrations = []
            for position in self.portfolio_positions.values():
                weight = position.get("value", 0) / total_value
                concentrations.append(weight**2)

            herfindahl_index = sum(concentrations)

            # Convert to risk score (higher concentration = higher risk)
            concentration_risk = min(1.0, herfindahl_index * 2)

            return concentration_risk

        except Exception:
            return 0.4  # Default moderate concentration risk

    def _execute_risk_management(self):
        """Execute risk management rules"""
        try:
            with self._lock:
                # Check position sizes
                self._check_position_sizes()

                # Check stop losses and take profits
                self._check_stop_loss_take_profit()

                # Check overall portfolio risk
                self._check_portfolio_risk_limits()

        except Exception as e:
            self.logger.error(f"Risk management error: {str(e)}")

    def _check_position_sizes(self):
        """Check and adjust position sizes"""
        for symbol, position in list(self.portfolio_positions.items()):
            try:
                current_value = position.get("value", 0)
                portfolio_value = sum(
                    pos.get("value", 0) for pos in self.portfolio_positions.values()
                )

                if portfolio_value > 0:
                    position_weight = current_value / portfolio_value

                    if position_weight > self.max_position_size:
                        # Position too large, generate sell signal
                        self._generate_risk_management_signal(
                            symbol,
                            "reduce_position",
                            f"Position size {position_weight:.2%} exceeds limit {self.max_position_size:.2%}",
                        )

            except Exception as e:
                self.logger.error(f"Position size check error for {symbol}: {str(e)}")

    def _check_stop_loss_take_profit(self):
        """Check stop loss and take profit levels"""
        for symbol, position in list(self.portfolio_positions.items()):
            try:
                entry_price = position.get("entry_price", 0)
                current_data = self.data_manager.get_market_data(symbol=symbol)

                if current_data is not None and not current_data.empty and entry_price > 0:
                    current_price = current_data.iloc[0].get("price", 0)

                    if current_price > 0:
                        price_change = (current_price - entry_price) / entry_price

                        if price_change <= -self.stop_loss_pct:
                            # Stop loss triggered
                            self._generate_risk_management_signal(
                                symbol, "stop_loss", f"Stop loss triggered: {price_change:.2%}"
                            )

                        elif price_change >= self.take_profit_pct:
                            # Take profit triggered
                            self._generate_risk_management_signal(
                                symbol, "take_profit", f"Take profit triggered: {price_change:.2%}"
                            )

            except Exception as e:
                self.logger.error(f"Stop loss/take profit check error for {symbol}: {str(e)}")

    def _check_portfolio_risk_limits(self):
        """Check overall portfolio risk limits"""
        try:
            risk_score = self.risk_metrics.get("risk_score", 0)

            if risk_score > self.max_portfolio_risk:
                # Portfolio risk too high, reduce positions
                self._generate_portfolio_risk_signal(
                    "reduce_overall_risk",
                    f"Portfolio risk {risk_score:.2%} exceeds limit {self.max_portfolio_risk:.2%}",
                )

        except Exception as e:
            self.logger.error(f"Portfolio risk check error: {str(e)}")

    def _generate_risk_management_signal(self, symbol: str, action: str, reason: str):
        """Generate risk management signal"""
        risk_signal = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "reason": reason,
            "signal_type": "risk_management",
        }

        # Add to trade history
        self.trade_history.append(risk_signal)

        # Log the signal
        self.logger.warning(f"Risk management signal: {action} for {symbol} - {reason}")

    def _generate_portfolio_risk_signal(self, action: str, reason: str):
        """Generate portfolio-level risk management signal"""
        risk_signal = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "reason": reason,
            "signal_type": "portfolio_risk_management",
        }

        # Add to trade history
        self.trade_history.append(risk_signal)

        # Log the signal
        self.logger.warning(f"Portfolio risk signal: {action} - {reason}")

    def get_trading_signals(self, symbol: str = None) -> Dict[str, Any]:
        """Get current trading signals"""
        with self._lock:
            if symbol:
                return self.trading_signals.get(symbol, [])
            else:
                return self.trading_signals.copy()

    def get_latest_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest trading signal for a symbol"""
        with self._lock:
            signals = self.trading_signals.get(symbol, [])
            return signals[-1] if signals else None

    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of all trading signals"""
        with self._lock:
            if not self.trading_signals:
                return {
                    "total_symbols": 0,
                    "buy_signals": 0,
                    "sell_signals": 0,
                    "hold_signals": 0,
                    "high_confidence_signals": 0,
                }

            latest_signals = {}
            for symbol, signals in self.trading_signals.items():
                if signals:
                    latest_signals[symbol] = signals[-1]

            buy_count = sum(1 for signal in latest_signals.values() if signal["signal"] == "buy")
            sell_count = sum(1 for signal in latest_signals.values() if signal["signal"] == "sell")
            hold_count = len(latest_signals) - buy_count - sell_count
            high_confidence = sum(
                1 for signal in latest_signals.values() if signal["confidence"] > 0.7
            )

            return {
                "total_symbols": len(latest_signals),
                "buy_signals": buy_count,
                "sell_signals": sell_count,
                "hold_signals": hold_count,
                "high_confidence_signals": high_confidence,
                "last_update": self.last_update.isoformat() if self.last_update else None,
            }

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        with self._lock:
            return {
                "positions": self.portfolio_positions.copy(),
                "risk_metrics": self.risk_metrics.copy(),
                "total_positions": len(self.portfolio_positions),
                "risk_management_active": True,
            }

    def get_trade_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent trade history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            trade
            for trade in self.trade_history
            if datetime.fromisoformat(trade["timestamp"]) > cutoff_time
        ]

    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "active": self.active,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "tracked_signals": len(self.trading_signals),
            "portfolio_positions": len(self.portfolio_positions),
            "trade_history_count": len(self.trade_history),
        }
