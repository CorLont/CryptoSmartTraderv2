#!/usr/bin/env python3
"""
Demo: Enterprise Alpha & Portfolio Management System
Comprehensive demonstration of regime detection, Kelly sizing, cluster management, and return attribution.
"""

import asyncio
import numpy as np
import pandas as pd
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptosmarttrader.alpha import (
    create_regime_detector, create_kelly_optimizer, 
    create_cluster_manager, create_return_attributor,
    MarketRegime, AssetCluster, ReturnComponent
)


def generate_synthetic_data():
    """Generate synthetic market data for demonstration."""
    # Create synthetic price data for multiple assets
    symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'UNI-USD', 'AAVE-USD', 'SOL-USD']
    
    # Generate correlated returns
    np.random.seed(42)
    n_days = 200
    n_assets = len(symbols)
    
    # Create correlation structure
    correlation_matrix = np.random.uniform(0.3, 0.8, (n_assets, n_assets))
    np.fill_diagonal(correlation_matrix, 1.0)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
    
    # Generate returns
    returns = np.random.multivariate_normal(
        mean=[0.001] * n_assets,  # 0.1% daily return
        cov=correlation_matrix * 0.04,  # 4% daily volatility base
        size=n_days
    )
    
    # Convert to prices
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    price_data = {}
    
    base_prices = [50000, 3000, 0.5, 25, 15, 10, 300, 100]  # Starting prices
    
    for i, symbol in enumerate(symbols):
        prices = [base_prices[i]]
        for j in range(n_days - 1):
            prices.append(prices[-1] * (1 + returns[j, i]))
        
        # Add volume data
        volumes = np.random.uniform(1000000, 10000000, n_days)
        
        price_data[symbol] = pd.DataFrame({
            'close': prices,
            'volume': volumes
        }, index=dates)
    
    # Asset metadata
    asset_metadata = {
        'BTC-USD': {'market_cap': 1000000000000, 'category': 'store of value', 'age_days': 5000},
        'ETH-USD': {'market_cap': 400000000000, 'category': 'platform layer 1', 'age_days': 3000},
        'ADA-USD': {'market_cap': 15000000000, 'category': 'platform layer 1', 'age_days': 2000},
        'DOT-USD': {'market_cap': 8000000000, 'category': 'platform layer 1', 'age_days': 1500},
        'LINK-USD': {'market_cap': 7000000000, 'category': 'infrastructure oracle', 'age_days': 2500},
        'UNI-USD': {'market_cap': 5000000000, 'category': 'defi exchange', 'age_days': 1200},
        'AAVE-USD': {'market_cap': 2000000000, 'category': 'defi lending', 'age_days': 1800},
        'SOL-USD': {'market_cap': 50000000000, 'category': 'platform layer 1', 'age_days': 1000}
    }
    
    return price_data, asset_metadata, pd.DataFrame(returns, columns=symbols, index=dates)


async def demonstrate_alpha_portfolio_system():
    """Comprehensive demonstration of alpha and portfolio management."""
    print("🎯 ENTERPRISE ALPHA & PORTFOLIO MANAGEMENT DEMONSTRATION")
    print("=" * 70)
    
    # Generate synthetic data
    print("📊 Generating synthetic market data...")
    price_data, asset_metadata, returns_data = generate_synthetic_data()
    print(f"   Generated data for {len(price_data)} assets over {len(returns_data)} days")
    
    # Initialize systems
    regime_detector = create_regime_detector(lookback_periods=50, min_confidence=0.6)
    kelly_optimizer = create_kelly_optimizer(kelly_fraction=0.25, volatility_target=0.15)
    cluster_manager = create_cluster_manager(max_cluster_allocation=0.30)
    return_attributor = create_return_attributor(benchmark_return=0.08)  # 8% annual benchmark
    
    print("✅ Alpha & Portfolio systems initialized")
    
    # Demo 1: Regime Detection
    print("\n🔍 DEMO 1: Market Regime Detection")
    print("-" * 50)
    
    regime_signals = {}
    
    for symbol in ['BTC-USD', 'ETH-USD', 'ADA-USD']:
        print(f"\n   Analyzing {symbol} regime...")
        
        price_series = price_data[symbol]['close']
        volume_series = price_data[symbol]['volume']
        
        regime_signal = regime_detector.detect_regime(price_series, volume_series)
        regime_signals[symbol] = regime_signal
        
        print(f"      Regime: {regime_signal.regime.value}")
        print(f"      Confidence: {regime_signal.confidence:.2f}")
        print(f"      Strength: {regime_signal.strength:.2f}")
        print(f"      Hurst Exponent: {regime_signal.indicators.get('hurst_exponent', 0):.3f}")
        print(f"      ADX: {regime_signal.indicators.get('adx', 0):.1f}")
        print(f"      Volatility: {regime_signal.indicators.get('realized_volatility', 0):.3f}")
        
        # Get strategy configuration for regime
        strategy_config = regime_detector.get_regime_strategy_config(regime_signal.regime)
        print(f"      Strategy: {strategy_config['strategy_type']}")
        print(f"      Vol Target: {strategy_config['volatility_target']:.1%}")
        print(f"      Throttling: {strategy_config['throttling_factor']:.1f}x")
    
    # Demo 2: Asset Classification & Clustering
    print("\n🏷️ DEMO 2: Asset Classification & Clustering")
    print("-" * 50)
    
    # Classify assets into clusters
    asset_classifications = cluster_manager.classify_assets(asset_metadata, returns_data)
    
    print("   Asset Classifications:")
    for symbol, classification in asset_classifications.items():
        print(f"      {symbol}:")
        print(f"         Primary Cluster: {classification.primary_cluster.value}")
        print(f"         Market Cap Tier: {classification.market_cap_tier}")
        print(f"         Risk Score: {classification.risk_score:.2f}")
        print(f"         Correlation Group: {classification.correlation_group}")
    
    # Demo 3: Kelly Sizing & Portfolio Optimization
    print("\n💰 DEMO 3: Kelly Sizing & Portfolio Optimization")
    print("-" * 50)
    
    # Calculate expected returns and volatilities
    expected_returns = {}
    volatilities = {}
    confidence_scores = {}
    
    for symbol in asset_metadata.keys():
        # Use last 30 days for estimation
        recent_returns = returns_data[symbol].iloc[-30:]
        
        # Expected return (annualized)
        expected_returns[symbol] = recent_returns.mean() * 365
        
        # Volatility (annualized)
        volatilities[symbol] = recent_returns.std() * np.sqrt(365)
        
        # Mock confidence score based on regime detection
        regime_signal = regime_signals.get(symbol)
        if regime_signal:
            confidence_scores[symbol] = regime_signal.confidence
        else:
            confidence_scores[symbol] = 0.7
    
    print("   Expected Returns & Volatilities:")
    for symbol in list(expected_returns.keys())[:5]:
        print(f"      {symbol}: Return {expected_returns[symbol]:.1%}, Vol {volatilities[symbol]:.1%}, Confidence {confidence_scores[symbol]:.2f}")
    
    # Calculate optimal portfolio allocation
    correlations = returns_data.corr()
    
    portfolio_allocation = kelly_optimizer.calculate_kelly_sizing(
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlations=correlations,
        confidence_scores=confidence_scores
    )
    
    print(f"\n   Portfolio Allocation Results:")
    print(f"      Total Allocation: {portfolio_allocation.total_allocation:.1%}")
    print(f"      Portfolio Volatility: {portfolio_allocation.portfolio_volatility:.1%}")
    print(f"      Portfolio Sharpe: {portfolio_allocation.portfolio_sharpe:.2f}")
    print(f"      Diversification Ratio: {portfolio_allocation.diversification_ratio:.2f}")
    print(f"      Risk Budget Used: {portfolio_allocation.risk_budget_used:.1%}")
    
    print("\n   Top Position Allocations:")
    sorted_positions = sorted(
        portfolio_allocation.allocations.items(),
        key=lambda x: x[1].final_allocation,
        reverse=True
    )
    
    for symbol, sizing in sorted_positions[:6]:
        print(f"      {symbol}: {sizing.final_allocation:.1%} "
              f"(Kelly: {sizing.kelly_fraction:.1%}, "
              f"Sharpe: {sizing.sharpe_ratio:.2f})")
    
    if portfolio_allocation.constraints_applied:
        print(f"\n   Constraints Applied: {portfolio_allocation.constraints_applied}")
    
    # Demo 4: Cluster Risk Management
    print("\n🎛️ DEMO 4: Cluster Risk Management")
    print("-" * 50)
    
    # Extract final allocations
    final_allocations = {
        symbol: sizing.final_allocation 
        for symbol, sizing in portfolio_allocation.allocations.items()
    }
    
    # Check cluster limits
    is_valid, violations = cluster_manager.check_cluster_limits(final_allocations)
    
    print(f"   Cluster Limits Check: {'✅ PASS' if is_valid else '❌ VIOLATIONS'}")
    if violations:
        print("   Violations:")
        for violation in violations:
            print(f"      ⚠️ {violation}")
    
    # Calculate cluster exposures
    cluster_exposures = cluster_manager.calculate_cluster_exposures(final_allocations)
    
    print("\n   Cluster Exposures:")
    for cluster, exposure in cluster_exposures.items():
        if exposure.asset_count > 0:
            print(f"      {cluster.value}:")
            print(f"         Allocation: {exposure.total_allocation:.1%}")
            print(f"         Assets: {exposure.asset_count}")
            print(f"         Avg Correlation: {exposure.avg_correlation:.2f}")
            print(f"         Limit Utilization: {exposure.limit_utilization:.1%}")
    
    # Apply cluster constraints if needed
    if not is_valid:
        print("\n   Applying cluster constraints...")
        constrained_allocations = cluster_manager.apply_cluster_constraints(final_allocations)
        
        print("   Adjusted allocations:")
        for symbol in list(constrained_allocations.keys())[:5]:
            original = final_allocations.get(symbol, 0.0)
            adjusted = constrained_allocations[symbol]
            change = adjusted - original
            print(f"      {symbol}: {original:.1%} → {adjusted:.1%} ({change:+.1%})")
    
    # Demo 5: Dynamic Leverage & Regime Adaptation
    print("\n📊 DEMO 5: Dynamic Leverage & Regime Adaptation")
    print("-" * 50)
    
    # Test different regime scenarios
    test_regimes = [
        ('trending_up', 0.12, 1.8),
        ('high_volatility', 0.25, 0.8),
        ('mean_reverting', 0.08, 1.2),
        ('choppy', 0.15, 0.6)
    ]
    
    print("   Dynamic Leverage Calculations:")
    for regime, market_vol, portfolio_sharpe in test_regimes:
        dynamic_leverage = kelly_optimizer.calculate_dynamic_leverage(
            regime, portfolio_sharpe, market_vol
        )
        print(f"      {regime}: {dynamic_leverage:.2f}x leverage "
              f"(Sharpe: {portfolio_sharpe:.1f}, Vol: {market_vol:.1%})")
    
    # Demo 6: Return Attribution Analysis
    print("\n📈 DEMO 6: Return Attribution Analysis")
    print("-" * 50)
    
    # Generate mock execution data
    print("   Generating mock execution and performance data...")
    
    period_start = datetime.now() - timedelta(days=30)
    period_end = datetime.now()
    
    # Mock portfolio data
    mock_portfolio_data = {}
    mock_execution_data = {}
    
    for symbol in list(expected_returns.keys())[:5]:
        # Portfolio position data
        mock_portfolio_data[symbol] = {
            'position_size': final_allocations.get(symbol, 0.0) * 100000,  # $100k portfolio
            'optimal_size': final_allocations.get(symbol, 0.0) * 100000,
            'actual_size': final_allocations.get(symbol, 0.0) * 100000 * random.uniform(0.8, 1.2),
            'leverage': 1.0,
            'funding_rate': 0.0001 if symbol in ['BTC-USD', 'ETH-USD'] else 0.0
        }
        
        # Mock execution data
        executions = []
        for i in range(random.randint(2, 8)):
            execution_time = period_start + timedelta(days=random.uniform(0, 30))
            executions.append({
                'timestamp': execution_time,
                'price': price_data[symbol]['close'].iloc[-random.randint(1, 30)],
                'quantity': random.uniform(0.01, 1.0),
                'side': random.choice(['buy', 'sell']),
                'fee_rate': 0.001,
                'slippage_bps': random.uniform(2, 20)
            })
        
        mock_execution_data[symbol] = executions
    
    # Perform return attribution
    portfolio_attribution = return_attributor.attribute_returns(
        portfolio_data=mock_portfolio_data,
        execution_data=mock_execution_data,
        market_data=price_data,
        period_start=period_start,
        period_end=period_end
    )
    
    print(f"\n   Portfolio Attribution Results:")
    print(f"      Total Return: {portfolio_attribution.total_portfolio_return:.2%}")
    print(f"      Benchmark Return: {portfolio_attribution.benchmark_return:.2%}")
    print(f"      Excess Return: {portfolio_attribution.excess_return:.2%}")
    print(f"      Performance Quality: {portfolio_attribution.performance_quality:.2f}")
    
    print(f"\n   Return Component Breakdown:")
    for component, value in portfolio_attribution.component_summary.items():
        if abs(value) > 0.001:  # Only show significant components
            print(f"      {component.value}: {value:.2%}")
    
    print(f"\n   Risk-Adjusted Metrics:")
    for metric, value in portfolio_attribution.risk_adjusted_metrics.items():
        print(f"      {metric}: {value:.3f}")
    
    # Show individual asset attributions
    print(f"\n   Top Asset Attributions:")
    asset_items = list(portfolio_attribution.asset_attributions.items())[:3]
    for symbol, attribution in asset_items:
        print(f"      {symbol}:")
        print(f"         Total Return: {attribution.total_return:.2%}")
        print(f"         Alpha: {attribution.components[ReturnComponent.ALPHA]:.2%}")
        print(f"         Fees: {attribution.components[ReturnComponent.FEES]:.2%}")
        print(f"         Slippage: {attribution.components[ReturnComponent.SLIPPAGE]:.2%}")
        print(f"         Quality: {attribution.attribution_quality:.2f}")
    
    # Demo 7: Rebalancing & Transaction Cost Optimization
    print("\n🔄 DEMO 7: Rebalancing & Transaction Cost Optimization")
    print("-" * 50)
    
    # Simulate changed market conditions
    print("   Simulating changed market conditions...")
    
    # Adjust expected returns (simulate regime change)
    new_expected_returns = expected_returns.copy()
    for symbol in new_expected_returns:
        new_expected_returns[symbol] *= random.uniform(0.7, 1.3)
    
    # Calculate new optimal allocation
    new_allocation = kelly_optimizer.calculate_kelly_sizing(
        expected_returns=new_expected_returns,
        volatilities=volatilities,
        correlations=correlations,
        confidence_scores=confidence_scores,
        current_positions=final_allocations
    )
    
    # Calculate rebalancing with transaction costs
    rebalanced_allocation = kelly_optimizer.rebalance_portfolio(
        current_allocation=portfolio_allocation,
        new_expected_returns=new_expected_returns,
        new_volatilities=volatilities,
        transaction_cost=0.001  # 0.1% transaction cost
    )
    
    print(f"   Rebalancing Analysis:")
    print(f"      New Total Allocation: {rebalanced_allocation.total_allocation:.1%}")
    print(f"      Portfolio Sharpe Change: {portfolio_allocation.portfolio_sharpe:.2f} → {rebalanced_allocation.portfolio_sharpe:.2f}")
    
    print(f"\n   Rebalancing Changes:")
    for symbol in list(final_allocations.keys())[:5]:
        original = final_allocations.get(symbol, 0.0)
        new = rebalanced_allocation.allocations.get(symbol)
        new_size = new.final_allocation if new else 0.0
        change = new_size - original
        
        if abs(change) > 0.005:  # Show changes > 0.5%
            transaction_cost = new.metadata.get('transaction_cost', 0.0) if new else 0.0
            print(f"      {symbol}: {original:.1%} → {new_size:.1%} "
                  f"({change:+.1%}, cost: {transaction_cost:.3%})")
    
    # Demo 8: Performance Summary & Recommendations
    print("\n🎯 DEMO 8: Performance Summary & Recommendations")
    print("-" * 50)
    
    # Generate comprehensive summary
    cluster_summary = cluster_manager.get_cluster_summary(final_allocations)
    
    print(f"   Portfolio Summary:")
    print(f"      Total Assets: {cluster_summary['diversification_metrics']['total_assets']}")
    print(f"      Active Clusters: {cluster_summary['diversification_metrics']['active_clusters']}")
    print(f"      Concentration Score: {cluster_summary['diversification_metrics']['concentration_score']:.2f}")
    print(f"      Max Single Allocation: {cluster_summary['diversification_metrics']['max_single_allocation']:.1%}")
    
    print(f"\n   Risk Management Status:")
    if cluster_summary['constraint_violations']:
        print(f"      ⚠️ {len(cluster_summary['constraint_violations'])} constraint violations")
    else:
        print(f"      ✅ All constraints satisfied")
    
    print(f"      Risk Budget Used: {portfolio_allocation.risk_budget_used:.1%}")
    print(f"      Diversification Ratio: {portfolio_allocation.diversification_ratio:.2f}")
    
    # Performance quality assessment
    avg_quality = np.mean([
        portfolio_allocation.portfolio_sharpe / 2.0,  # Normalize Sharpe
        portfolio_allocation.diversification_ratio,
        1.0 - cluster_summary['diversification_metrics']['concentration_score'],
        portfolio_attribution.performance_quality
    ])
    
    print(f"\n   Overall System Quality: {avg_quality:.1%}")
    
    # Recommendations based on analysis
    print(f"\n   Recommendations:")
    
    if portfolio_allocation.portfolio_sharpe < 1.0:
        print(f"      📊 Consider increasing alpha generation - Sharpe ratio is {portfolio_allocation.portfolio_sharpe:.2f}")
    
    if cluster_summary['diversification_metrics']['concentration_score'] > 0.7:
        print(f"      🎯 Portfolio is concentrated - consider diversification")
    
    if portfolio_attribution.component_summary[ReturnComponent.FEES] < -0.02:
        print(f"      💰 High fee impact - optimize execution strategy")
    
    if any(regime_signal.confidence < 0.7 for regime_signal in regime_signals.values()):
        print(f"      🔍 Low regime detection confidence - gather more market data")
    
    if portfolio_allocation.risk_budget_used < 0.5:
        print(f"      📈 Low risk budget utilization - consider increasing position sizes")
    
    print("\n✅ ALPHA & PORTFOLIO MANAGEMENT DEMONSTRATION COMPLETED")
    print("=" * 70)
    
    # Final statistics
    print(f"🎯 ENTERPRISE ALPHA SYSTEM ACHIEVEMENTS:")
    print(f"   ✅ Market regime detection with {np.mean([s.confidence for s in regime_signals.values()]):.1%} avg confidence")
    print(f"   ✅ Portfolio optimization with {portfolio_allocation.portfolio_sharpe:.2f} Sharpe ratio")
    print(f"   ✅ Cluster risk management across {cluster_summary['diversification_metrics']['active_clusters']} clusters")
    print(f"   ✅ Return attribution with {portfolio_attribution.performance_quality:.1%} quality score")
    print(f"   ✅ Dynamic leverage adjustment and regime adaptation")
    print(f"   ✅ Transaction cost optimization and rebalancing analysis")


if __name__ == "__main__":
    print("🎯 CRYPTOSMARTTRADER V2 - ALPHA & PORTFOLIO MANAGEMENT DEMO")
    print("=" * 70)
    
    try:
        asyncio.run(demonstrate_alpha_portfolio_system())
        print("\n🏆 Alpha & portfolio management demonstration completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)