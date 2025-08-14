#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - AI-Driven Portfolio Optimization Engine
Advanced portfolio allocation using ML models and risk optimization
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import json
from pathlib import Path
import warnings

# Core ML imports
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.covariance import LedoitWolf
    from sklearn.model_selection import cross_val_score
    import scipy.optimize as optimize
    from scipy import stats

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn(
        "Scikit-learn not available, portfolio optimization will have limited functionality"
    )

# Advanced optimization
try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


@dataclass
class PortfolioConfig:
    """Configuration for AI portfolio optimization"""

    # Risk parameters
    max_position_size: float = 0.25  # Max 25% in single asset
    min_position_size: float = 0.01  # Min 1% position
    risk_tolerance: float = 0.15  # Maximum portfolio volatility

    # Optimization settings
    lookback_days: int = 90
    rebalance_frequency: str = "daily"  # daily, weekly, monthly

    # AI/ML settings
    use_ml_returns: bool = True
    use_clustering: bool = True
    cluster_count: int = 5

    # Risk models
    risk_model: str = "sample"  # sample, ledoit_wolf, factor
    return_model: str = "ml_ensemble"  # historical, ml_ensemble, black_litterman

    # Constraints
    max_assets: int = 20
    min_correlation_threshold: float = 0.8  # Avoid highly correlated assets

    # Performance tracking
    track_performance: bool = True
    performance_benchmark: str = "equal_weight"


@dataclass
class AssetSignal:
    """Individual asset signal and prediction"""

    symbol: str
    expected_return: float
    expected_volatility: float
    confidence: float
    ml_prediction: float
    sentiment_score: float
    technical_score: float
    fundamental_score: float
    risk_score: float
    timestamp: datetime


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result"""

    allocations: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    optimization_method: str
    constraints_active: List[str]
    rebalance_cost: float
    confidence_score: float
    creation_timestamp: datetime


class MLReturnPredictor:
    """ML-based return prediction for portfolio optimization"""

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.is_trained = False

    def train_models(self, price_data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """Train ML models for return prediction"""

        if not HAS_SKLEARN:
            logging.warning("Scikit-learn not available, using simple historical returns")
            return {"method": "historical", "models_trained": 0}

        try:
            # Calculate returns
            returns = price_data.pct_change().dropna()

            results = {"models_trained": 0, "performance": {}}

            for symbol in returns.columns:
                if symbol in features.columns:
                    # Prepare data
                    X = features[symbol].dropna()
                    y = returns[symbol].dropna()

                    # Align data
                    common_idx = X.index.intersection(y.index)
                    if len(common_idx) < 50:  # Need minimum data
                        continue

                    X_aligned = X.loc[common_idx].values.reshape(-1, 1)
                    y_aligned = y.loc[common_idx].values

                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_aligned)

                    # Train model
                    from sklearn.ensemble import RandomForestRegressor

                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_scaled, y_aligned)

                    # Cross-validation
                    cv_scores = cross_val_score(model, X_scaled, y_aligned, cv=5)

                    # Store model
                    self.models[symbol] = model
                    self.scalers[symbol] = scaler
                    results["models_trained"] += 1
                    results["performance"][symbol] = np.mean(cv_scores)

            self.is_trained = True
            return results

        except Exception as e:
            logging.error(f"ML model training failed: {e}")
            return {"error": str(e)}

    def predict_returns(
        self, current_features: pd.Series, horizon_days: int = 1
    ) -> Dict[str, float]:
        """Predict expected returns using ML models"""

        predictions = {}

        if not self.is_trained:
            logging.warning("Models not trained, using zero returns")
            return {symbol: 0.0 for symbol in current_features.index}

        try:
            for symbol, model in self.models.items():
                if symbol in current_features.index and not pd.isna(current_features[symbol]):
                    # Scale feature
                    feature_scaled = self.scalers[symbol].transform([[current_features[symbol]]])

                    # Predict
                    prediction = model.predict(feature_scaled)[0]

                    # Adjust for horizon
                    predictions[symbol] = prediction * np.sqrt(horizon_days)
                else:
                    predictions[symbol] = 0.0

            return predictions

        except Exception as e:
            logging.error(f"Return prediction failed: {e}")
            return {symbol: 0.0 for symbol in current_features.index}


class RiskModel:
    """Advanced risk modeling for portfolio optimization"""

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.covariance_matrix = None
        self.risk_factors = None

    def estimate_risk(self, returns: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Estimate risk model (covariance matrix)"""

        if not HAS_SKLEARN:
            # Simple sample covariance
            return returns.cov().values, {"method": "sample"}

        try:
            if self.config.risk_model == "ledoit_wolf":
                return self._ledoit_wolf_covariance(returns)
            elif self.config.risk_model == "factor":
                return self._factor_model_covariance(returns)
            else:
                return self._sample_covariance(returns)

        except Exception as e:
            logging.error(f"Risk estimation failed: {e}")
            return returns.cov().values, {"method": "sample", "error": str(e)}

    def _ledoit_wolf_covariance(self, returns: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Ledoit-Wolf shrinkage estimator"""

        cov_estimator = LedoitWolf()
        cov_matrix = cov_estimator.fit(returns.dropna()).covariance_

        return cov_matrix, {"method": "ledoit_wolf", "shrinkage": cov_estimator.shrinkage_}

    def _factor_model_covariance(self, returns: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Factor model risk estimation"""

        # Use PCA for factor extraction
        pca = PCA(n_components=min(5, len(returns.columns)))
        factors = pca.fit_transform(returns.dropna())

        # Factor loadings
        loadings = pca.components_.T

        # Factor covariance
        factor_cov = np.cov(factors.T)

        # Specific risk (diagonal)
        explained_var = np.sum(
            (loadings @ factor_cov @ loadings.T) * np.eye(len(returns.columns)), axis=1
        )
        total_var = np.diag(returns.cov())
        specific_var = np.maximum(total_var - explained_var, 0.01 * total_var)

        # Reconstruct covariance
        cov_matrix = loadings @ factor_cov @ loadings.T + np.diag(specific_var)

        return cov_matrix, {
            "method": "factor_model",
            "factors": pca.n_components_,
            "explained_variance": np.sum(pca.explained_variance_ratio_),
        }

    def _sample_covariance(self, returns: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Simple sample covariance"""

        return returns.cov().values, {"method": "sample"}


class PortfolioOptimizer:
    """Advanced portfolio optimization engine"""

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.risk_model = RiskModel(config)

    def optimize_portfolio(
        self,
        asset_signals: List[AssetSignal],
        current_prices: Dict[str, float],
        current_allocation: Dict[str, float] = None,
    ) -> PortfolioAllocation:
        """Optimize portfolio allocation"""

        try:
            # Prepare optimization inputs
            symbols, expected_returns, risk_matrix = self._prepare_optimization_inputs(
                asset_signals
            )

            if len(symbols) == 0:
                return self._empty_allocation()

            # Choose optimization method
            if HAS_CVXPY:
                allocation_weights = self._cvxpy_optimization(
                    expected_returns, risk_matrix, symbols
                )
            else:
                allocation_weights = self._scipy_optimization(
                    expected_returns, risk_matrix, symbols
                )

            # Create allocation dictionary
            allocations = dict(zip(symbols, allocation_weights))

            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                allocations, expected_returns, risk_matrix
            )

            # Calculate rebalancing cost
            rebalance_cost = self._calculate_rebalance_cost(allocations, current_allocation or {})

            return PortfolioAllocation(
                allocations=allocations,
                expected_return=portfolio_metrics["expected_return"],
                expected_volatility=portfolio_metrics["expected_volatility"],
                sharpe_ratio=portfolio_metrics["sharpe_ratio"],
                max_drawdown=portfolio_metrics["max_drawdown"],
                optimization_method="cvxpy" if HAS_CVXPY else "scipy",
                constraints_active=portfolio_metrics["constraints_active"],
                rebalance_cost=rebalance_cost,
                confidence_score=portfolio_metrics["confidence_score"],
                creation_timestamp=datetime.now(),
            )

        except Exception as e:
            logging.error(f"Portfolio optimization failed: {e}")
            return self._empty_allocation()

    def _prepare_optimization_inputs(
        self, asset_signals: List[AssetSignal]
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Prepare inputs for optimization"""

        # Filter high-confidence signals
        high_conf_signals = [signal for signal in asset_signals if signal.confidence >= 0.5]

        if not high_conf_signals:
            return [], np.array([]), np.array([[]])

        # Sort by confidence and limit number
        high_conf_signals.sort(key=lambda x: x.confidence, reverse=True)
        selected_signals = high_conf_signals[: self.config.max_assets]

        # Extract data
        symbols = [signal.symbol for signal in selected_signals]
        expected_returns = np.array([signal.expected_return for signal in selected_signals])

        # Create simple covariance matrix (in real implementation, use historical data)
        volatilities = np.array([signal.expected_volatility for signal in selected_signals])
        correlation_matrix = np.eye(len(symbols))  # Simplified

        # Add some realistic correlations
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                correlation_matrix[i, j] = correlation_matrix[j, i] = np.random.uniform(0.3, 0.7)

        risk_matrix = np.outer(volatilities, volatilities) * correlation_matrix

        return symbols, expected_returns, risk_matrix

    def _cvxpy_optimization(
        self, expected_returns: np.ndarray, risk_matrix: np.ndarray, symbols: List[str]
    ) -> np.ndarray:
        """Portfolio optimization using CVXPY"""

        n_assets = len(symbols)
        weights = cp.Variable(n_assets)

        # Objective: maximize return - risk penalty
        risk_penalty = 0.5  # Risk aversion parameter
        portfolio_return = expected_returns.T @ weights
        portfolio_risk = cp.quad_form(weights, risk_matrix)

        objective = cp.Maximize(portfolio_return - risk_penalty * portfolio_risk)

        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Fully invested
            weights >= self.config.min_position_size,  # Minimum position
            weights <= self.config.max_position_size,  # Maximum position
        ]

        # Portfolio volatility constraint
        if self.config.risk_tolerance:
            constraints.append(
                cp.sqrt(cp.quad_form(weights, risk_matrix)) <= self.config.risk_tolerance
            )

        # Solve optimization
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status == cp.OPTIMAL:
            return weights.value
        else:
            logging.warning(f"Optimization failed with status: {problem.status}")
            return self._equal_weight_fallback(n_assets)

    def _scipy_optimization(
        self, expected_returns: np.ndarray, risk_matrix: np.ndarray, symbols: List[str]
    ) -> np.ndarray:
        """Portfolio optimization using SciPy"""

        n_assets = len(symbols)

        def objective_function(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(risk_matrix, weights)))
            return -(portfolio_return - 0.5 * portfolio_risk)  # Negative for minimization

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Fully invested
        ]

        # Bounds
        bounds = [
            (self.config.min_position_size, self.config.max_position_size) for _ in range(n_assets)
        ]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = optimize.minimize(
            objective_function, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if result.success:
            return result.x
        else:
            logging.warning(f"SciPy optimization failed: {result.message}")
            return self._equal_weight_fallback(n_assets)

    def _equal_weight_fallback(self, n_assets: int) -> np.ndarray:
        """Fallback to equal weights"""
        return np.ones(n_assets) / n_assets

    def _calculate_portfolio_metrics(
        self, allocations: Dict[str, float], expected_returns: np.ndarray, risk_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate portfolio performance metrics"""

        weights = np.array(list(allocations.values()))

        # Expected return
        portfolio_return = np.dot(weights, expected_returns)

        # Expected volatility
        portfolio_variance = np.dot(weights.T, np.dot(risk_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

        # Max drawdown (simplified estimate)
        max_drawdown = portfolio_volatility * 2  # Heuristic

        # Check active constraints
        constraints_active = []
        if np.any(weights <= self.config.min_position_size + 1e-6):
            constraints_active.append("min_position")
        if np.any(weights >= self.config.max_position_size - 1e-6):
            constraints_active.append("max_position")
        if portfolio_volatility >= self.config.risk_tolerance - 1e-6:
            constraints_active.append("risk_tolerance")

        # Confidence score based on concentration and diversification
        concentration = np.sum(weights**2)  # Herfindahl index
        diversification_score = 1 - concentration
        confidence_score = min(diversification_score * 2, 1.0)

        return {
            "expected_return": float(portfolio_return),
            "expected_volatility": float(portfolio_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "constraints_active": constraints_active,
            "confidence_score": float(confidence_score),
        }

    def _calculate_rebalance_cost(
        self, new_allocation: Dict[str, float], current_allocation: Dict[str, float]
    ) -> float:
        """Calculate estimated rebalancing cost"""

        if not current_allocation:
            return 0.0

        total_turnover = 0.0

        for symbol, new_weight in new_allocation.items():
            current_weight = current_allocation.get(symbol, 0.0)
            total_turnover += abs(new_weight - current_weight)

        # Add turnover from assets no longer in portfolio
        for symbol, current_weight in current_allocation.items():
            if symbol not in new_allocation:
                total_turnover += current_weight

        # Assume 0.1% transaction cost
        return total_turnover * 0.001

    def _empty_allocation(self) -> PortfolioAllocation:
        """Return empty allocation for error cases"""
        return PortfolioAllocation(
            allocations={},
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            optimization_method="none",
            constraints_active=[],
            rebalance_cost=0.0,
            confidence_score=0.0,
            creation_timestamp=datetime.now(),
        )


class PerformanceTracker:
    """Track portfolio performance over time"""

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.performance_history = []
        self.benchmark_history = []

    def record_performance(
        self,
        allocation: PortfolioAllocation,
        actual_returns: Dict[str, float],
        benchmark_return: float = None,
    ):
        """Record actual performance"""

        # Calculate portfolio return
        portfolio_return = sum(
            weight * actual_returns.get(symbol, 0)
            for symbol, weight in allocation.allocations.items()
        )

        performance_record = {
            "timestamp": datetime.now(),
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return or 0,
            "allocation": allocation.allocations.copy(),
            "rebalance_cost": allocation.rebalance_cost,
        }

        self.performance_history.append(performance_record)

        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    def get_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Calculate performance metrics"""

        if len(self.performance_history) < 2:
            return {"error": "Insufficient performance history"}

        recent_history = self.performance_history[-days:]

        # Calculate metrics
        portfolio_returns = [r["portfolio_return"] for r in recent_history]
        benchmark_returns = [r["benchmark_return"] for r in recent_history]

        portfolio_cumulative = np.cumprod([1 + r for r in portfolio_returns])[-1] - 1
        benchmark_cumulative = np.cumprod([1 + r for r in benchmark_returns])[-1] - 1

        portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        benchmark_vol = np.std(benchmark_returns) * np.sqrt(252)

        return {
            "portfolio_return": portfolio_cumulative,
            "benchmark_return": benchmark_cumulative,
            "excess_return": portfolio_cumulative - benchmark_cumulative,
            "portfolio_volatility": portfolio_vol,
            "benchmark_volatility": benchmark_vol,
            "sharpe_ratio": np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252),
            "tracking_error": np.std([p - b for p, b in zip(portfolio_returns, benchmark_returns)])
            * np.sqrt(252),
            "information_ratio": (np.mean(portfolio_returns) - np.mean(benchmark_returns))
            / np.std([p - b for p, b in zip(portfolio_returns, benchmark_returns)]),
            "max_drawdown": self._calculate_max_drawdown(portfolio_returns),
            "win_rate": len([r for r in portfolio_returns if r > 0]) / len(portfolio_returns),
        }

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))


class AIPortfolioOptimizerCoordinator:
    """Main coordinator for AI-driven portfolio optimization"""

    def __init__(self, config: PortfolioConfig = None):
        self.config = config or PortfolioConfig()

        # Initialize components
        self.ml_predictor = MLReturnPredictor(self.config)
        self.optimizer = PortfolioOptimizer(self.config)
        self.performance_tracker = PerformanceTracker(self.config)

        # State
        self.current_allocation = {}
        self.last_optimization = None

        self._lock = threading.Lock()

        logging.info("AI Portfolio Optimizer Coordinator initialized")

    def optimize_portfolio_allocation(
        self,
        asset_signals: List[AssetSignal],
        current_prices: Dict[str, float],
        historical_data: pd.DataFrame = None,
    ) -> Dict[str, Any]:
        """Main portfolio optimization function"""

        try:
            with self._lock:
                # Train ML models if historical data available
                if historical_data is not None and not self.ml_predictor.is_trained:
                    # Create dummy features for demo
                    features = historical_data.rolling(20).mean()  # Simple moving average
                    training_result = self.ml_predictor.train_models(historical_data, features)
                    logging.info(f"ML training result: {training_result}")

                # Optimize allocation
                allocation = self.optimizer.optimize_portfolio(
                    asset_signals, current_prices, self.current_allocation
                )

                # Update current allocation
                self.current_allocation = allocation.allocations.copy()
                self.last_optimization = datetime.now()

                return {
                    "success": True,
                    "allocation": allocation.allocations,
                    "expected_return": allocation.expected_return,
                    "expected_volatility": allocation.expected_volatility,
                    "sharpe_ratio": allocation.sharpe_ratio,
                    "rebalance_cost": allocation.rebalance_cost,
                    "confidence_score": allocation.confidence_score,
                    "optimization_method": allocation.optimization_method,
                    "constraints_active": allocation.constraints_active,
                    "optimization_timestamp": allocation.creation_timestamp.isoformat(),
                    "total_assets": len(allocation.allocations),
                }

        except Exception as e:
            logging.error(f"Portfolio optimization failed: {e}")
            return {"error": str(e)}

    def update_performance(
        self, actual_returns: Dict[str, float], benchmark_return: float = None
    ) -> Dict[str, Any]:
        """Update performance tracking"""

        if not self.current_allocation:
            return {"error": "No current allocation to track"}

        try:
            # Create allocation object for tracking
            allocation = PortfolioAllocation(
                allocations=self.current_allocation,
                expected_return=0,
                expected_volatility=0,
                sharpe_ratio=0,
                max_drawdown=0,
                optimization_method="",
                constraints_active=[],
                rebalance_cost=0,
                confidence_score=0,
                creation_timestamp=datetime.now(),
            )

            self.performance_tracker.record_performance(
                allocation, actual_returns, benchmark_return
            )

            # Get recent performance metrics
            performance_metrics = self.performance_tracker.get_performance_metrics()

            return {
                "success": True,
                "performance_recorded": True,
                "recent_metrics": performance_metrics,
            }

        except Exception as e:
            logging.error(f"Performance update failed: {e}")
            return {"error": str(e)}

    def get_portfolio_analysis(self) -> Dict[str, Any]:
        """Get comprehensive portfolio analysis"""

        with self._lock:
            analysis = {
                "current_allocation": self.current_allocation,
                "last_optimization": self.last_optimization.isoformat()
                if self.last_optimization
                else None,
                "ml_models_trained": self.ml_predictor.is_trained,
                "configuration": asdict(self.config),
            }

            # Add performance metrics if available
            if self.performance_tracker.performance_history:
                try:
                    performance_metrics = self.performance_tracker.get_performance_metrics()
                    analysis["performance_metrics"] = performance_metrics
                except Exception:
                    analysis["performance_metrics"] = {"error": "Failed to calculate metrics"}

            return analysis

    def should_rebalance(self) -> bool:
        """Check if portfolio should be rebalanced"""

        if not self.last_optimization:
            return True

        # Time-based rebalancing
        hours_since_last = (datetime.now() - self.last_optimization).total_seconds() / 3600

        if self.config.rebalance_frequency == "daily" and hours_since_last >= 24:
            return True
        elif self.config.rebalance_frequency == "weekly" and hours_since_last >= 168:
            return True
        elif self.config.rebalance_frequency == "monthly" and hours_since_last >= 720:
            return True

        return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""

        return {
            "ml_predictor_trained": self.ml_predictor.is_trained,
            "models_count": len(self.ml_predictor.models),
            "current_allocation_assets": len(self.current_allocation),
            "last_optimization": self.last_optimization.isoformat()
            if self.last_optimization
            else None,
            "performance_history_length": len(self.performance_tracker.performance_history),
            "should_rebalance": self.should_rebalance(),
            "dependencies": {"sklearn": HAS_SKLEARN, "cvxpy": HAS_CVXPY},
            "config": asdict(self.config),
        }


# Singleton coordinator instance
_portfolio_optimizer_coordinator = None
_coordinator_lock = threading.Lock()


def get_ai_portfolio_optimizer_coordinator(
    config: PortfolioConfig = None,
) -> AIPortfolioOptimizerCoordinator:
    """Get the singleton AI portfolio optimizer coordinator"""
    global _portfolio_optimizer_coordinator

    with _coordinator_lock:
        if _portfolio_optimizer_coordinator is None:
            _portfolio_optimizer_coordinator = AIPortfolioOptimizerCoordinator(config)

        return _portfolio_optimizer_coordinator


# Test function
def test_ai_portfolio_optimizer():
    """Test the AI portfolio optimizer"""
    print("Testing AI Portfolio Optimizer...")

    config = PortfolioConfig(max_position_size=0.3, risk_tolerance=0.2)
    coordinator = get_ai_portfolio_optimizer_coordinator(config)

    # Create test signals
    test_signals = [
        AssetSignal(
            symbol="BTC",
            expected_return=0.15,
            expected_volatility=0.4,
            confidence=0.8,
            ml_prediction=0.12,
            sentiment_score=0.6,
            technical_score=0.7,
            fundamental_score=0.8,
            risk_score=0.3,
            timestamp=datetime.now(),
        ),
        AssetSignal(
            symbol="ETH",
            expected_return=0.12,
            expected_volatility=0.35,
            confidence=0.75,
            ml_prediction=0.1,
            sentiment_score=0.5,
            technical_score=0.6,
            fundamental_score=0.7,
            risk_score=0.4,
            timestamp=datetime.now(),
        ),
    ]

    current_prices = {"BTC": 45000, "ETH": 3000}

    # Test optimization
    result = coordinator.optimize_portfolio_allocation(test_signals, current_prices)
    print(f"Optimization result: {result}")

    # Test status
    status = coordinator.get_system_status()
    print(f"System status: {status}")

    print("AI Portfolio Optimizer test completed!")


if __name__ == "__main__":
    test_ai_portfolio_optimizer()
