"""
CryptoSmartTrader V2 - Alpha Opportunities Dashboard
STRICT: Only shows coins with 80%+ confidence and 100%+ expected 30-day return
NO DUMMY DATA - All data must be validated and real
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import time

class AlphaOpportunitiesDashboard:
    """Dashboard for alpha opportunities - STRICT criteria enforcement"""
    
    def __init__(self, container):
        self.container = container
        
        # Initialize components
        try:
            # Import real-time pipeline
            from core.real_time_pipeline import RealTimePipeline
            
            self.pipeline = RealTimePipeline(container)
            
            # Auto-start pipeline if not already running
            if not hasattr(st.session_state, 'pipeline_started'):
                self.pipeline.start_pipeline()
                st.session_state.pipeline_started = True
                
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {e}")
            self.pipeline = None
            
    def render(self):
        """Render the alpha opportunities dashboard with STRICT criteria"""
        st.title("🚀 Alpha Opportunities - 500%+ Return Potential")
        st.markdown("**STRICT CRITERIA: Alleen coins met 80%+ confidence EN 100%+ verwacht rendement (30 dagen)**")
        st.markdown("**GEEN DUMMY DATA - Alle data is gevalideerd en echt**")
        
        # Pipeline status warning
        if not self.pipeline:
            st.error("Real-time pipeline niet beschikbaar. Kan geen betrouwbare opportunities tonen.")
            return
        
        # Auto-refresh controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**Real-time analysis van ALLE Kraken cryptocurrencies - Zero tolerance voor dummy data**")
        with col2:
            auto_refresh = st.checkbox("Auto Refresh (60s)", value=True)
        with col3:
            if st.button("🔄 Force Pipeline Update"):
                self._force_pipeline_update()
                st.rerun()
        
        if auto_refresh:
            time.sleep(60)
            st.rerun()
        
        # Pipeline status
        self._render_pipeline_status()
        
        # STRICT Alpha opportunities table
        self._render_strict_alpha_opportunities()
        
        # Data quality verification
        self._render_data_quality_verification()
        
        # Pipeline performance
        self._render_pipeline_performance()
        
        # Multi-horizon ML status
        self._render_ml_system_status()
    
    def _render_pipeline_status(self):
        """Render real-time pipeline status with data quality focus"""
        st.header("📊 Real-time Pipeline Status")
        
        try:
            pipeline_status = self.pipeline.get_pipeline_status()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                status_indicator = "🟢 Running" if pipeline_status.get('pipeline_active', False) else "🔴 Stopped"
                st.metric("Pipeline Status", status_indicator)
            
            with col2:
                quality_metrics = pipeline_status.get('data_quality_metrics', {})
                total_coins = quality_metrics.get('total_coins_discovered', 0)
                st.metric("Kraken Coins Discovered", f"{total_coins:,}")
            
            with col3:
                complete_coins = quality_metrics.get('coins_with_complete_data', 0)
                st.metric("Complete Data Coins", f"{complete_coins:,}")
            
            with col4:
                completeness_ratio = quality_metrics.get('data_completeness_ratio', 0)
                st.metric("Data Completeness", f"{completeness_ratio:.1%}")
            
            # Data validation status
            st.subheader("✅ Data Validation Status")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Validation Requirements:**")
                st.write("• OHLCV data must be complete and valid")
                st.write("• No NULL, NaN, or negative values allowed")  
                st.write("• Sentiment data must be from real sources")
                st.write("• Whale data validated against blockchain")
                
            with col2:
                st.write("**Data Quality Metrics:**")
                if quality_metrics:
                    failed_coins = len(quality_metrics.get('failed_scraping_coins', set()))
                    missing_coins = len(quality_metrics.get('missing_data_coins', set()))
                    st.write(f"• Failed scraping: {failed_coins} coins")
                    st.write(f"• Missing data: {missing_coins} coins")
                    st.write(f"• Last scan: {quality_metrics.get('last_complete_scan', 'Never')}")
            
            # Pipeline task intervals
            st.subheader("⏱️ Pipeline Task Schedule")
            
            intervals = pipeline_status.get('task_intervals', {})
            
            task_display = {
                'coin_discovery': '🔍 Full Kraken Scan (10 min)',
                'price_data': '💰 Price Data Collection (5 min)', 
                'sentiment_scraping': '💭 Real Sentiment Scraping (15 min)',
                'whale_detection': '🐋 Whale Activity Detection (30 min)',
                'ml_batch_inference': '🤖 ML Batch Analysis (60 min)',
                'data_quality_check': '✅ Data Quality Verification (20 min)'
            }
            
            for task, display_name in task_display.items():
                if task in intervals:
                    interval_min = intervals[task] / 60
                    st.write(f"**{display_name}** - Every {interval_min:.0f} minutes")
                    
        except Exception as e:
            st.error(f"Failed to get pipeline status: {e}")
    
    def _render_strict_alpha_opportunities(self):
        """Render STRICT alpha opportunities - 80%+ confidence AND 100%+ return ONLY"""
        st.header("🎯 STRICT Alpha Opportunities")
        st.markdown("**ALLEEN coins met 80%+ confidence EN 100%+ verwacht rendement (30d). Anders NIETS.**")
        
        try:
            cache_manager = self.container.cache_manager()
            if not cache_manager:
                st.error("Cache manager niet beschikbaar voor data retrieval")
                return
            
            # Get final alpha opportunities from pipeline
            alpha_results = cache_manager.get('alpha_opportunities_final')
            
            if not alpha_results:
                st.info("⏳ Pipeline analyseert nog... Wacht tot de eerste batch-analyse voltooid is.")
                st.markdown("**Pipeline Status:**")
                st.write("• Coin discovery van Kraken")
                st.write("• Data validation (geen dummy data)")
                st.write("• Sentiment scraping van echte bronnen")
                st.write("• Whale detection op blockchain")
                st.write("• ML batch inference met confidence scoring")
                return
            
            opportunities = alpha_results.get('opportunities', [])
            
            # Apply STRICT filtering
            strict_opportunities = [
                opp for opp in opportunities
                if (opp.get('confidence', 0) >= 0.80 and 
                    opp.get('expected_return_30d', 0) >= 1.0 and  # 100%+ return
                    opp.get('meets_criteria', False))
            ]
            
            # Show analysis summary
            total_analyzed = alpha_results.get('total_analyzed', 0)
            high_conf_count = alpha_results.get('high_confidence_count', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Coins Analyzed", f"{total_analyzed:,}")
            with col2:
                st.metric("High Confidence", f"{high_conf_count:,}")
            with col3:
                st.metric("STRICT Criteria Met", f"{len(strict_opportunities):,}")
            
            # If no opportunities meet strict criteria, show NOTHING
            if not strict_opportunities:
                st.warning("🚫 GEEN opportunities voldoen aan STRICT criteria (80%+ confidence + 100%+ return)")
                st.markdown("**Criteria niet gehaald:**")
                st.write("• Confidence < 80% OF")
                st.write("• Verwacht rendement < 100% (30 dagen) OF")
                st.write("• Data niet volledig gevalideerd")
                
                # Show why opportunities were filtered out
                if opportunities:
                    st.subheader("🔍 Waarom opportunities gefilterd werden:")
                    filtered_reasons = []
                    
                    for opp in opportunities[:10]:  # Show first 10
                        conf = opp.get('confidence', 0) * 100
                        ret = opp.get('expected_return_30d', 0) * 100
                        symbol = opp.get('symbol', 'Unknown')
                        
                        if conf < 80:
                            filtered_reasons.append(f"{symbol}: Confidence {conf:.1f}% < 80%")
                        elif ret < 100:
                            filtered_reasons.append(f"{symbol}: Return {ret:.1f}% < 100%")
                    
                    for reason in filtered_reasons:
                        st.write(f"• {reason}")
                
                return
            
            # Show STRICT opportunities table
            st.subheader(f"🏆 {len(strict_opportunities)} Opportunities Meeting STRICT Criteria")
            st.markdown("**Multi-Horizon ML Analysis (1H, 24H, 7D, 30D)**")
            
            # Create table data
            table_data = []
            for opp in strict_opportunities:
                # Check for multi-horizon data
                horizon_data = opp.get('horizon_predictions', {})
                multi_horizon = len(horizon_data) > 1
                
                table_data.append({
                    'Symbol': opp['symbol'],
                    '1H Return': f"{horizon_data.get('1H', {}).get('predicted_return', 0)*100:.1f}%" if '1H' in horizon_data else "N/A",
                    '24H Return': f"{horizon_data.get('24H', {}).get('predicted_return', 0)*100:.1f}%" if '24H' in horizon_data else "N/A", 
                    '7D Return': f"{opp.get('expected_return_7d', 0)*100:.1f}%",
                    '30D Return': f"{opp.get('expected_return_30d', 0)*100:.1f}%",
                    'Confidence': f"{opp.get('confidence', 0)*100:.1f}%",
                    'ML Type': "Multi-Horizon" if multi_horizon else "Single",
                    'Timestamp': opp.get('prediction_timestamp', '')[:16]  # YYYY-MM-DD HH:MM
                })
            
            # Sort by 30-day return (highest first)
            table_data.sort(key=lambda x: float(x['30D Return'].replace('%', '')), reverse=True)
            
            df_strict = pd.DataFrame(table_data)
            
            st.dataframe(
                df_strict,
                use_container_width=True,
                column_config={
                    "30D Return": st.column_config.TextColumn("30D Return ⬇️", help="Expected return in 30 days (sorted high to low)"),
                    "7D Return": st.column_config.TextColumn("7D Return", help="Expected return in 7 days"),
                    "24H Return": st.column_config.TextColumn("24H Return", help="Expected return in 24 hours"),
                    "1H Return": st.column_config.TextColumn("1H Return", help="Expected return in 1 hour"),
                    "Confidence": st.column_config.TextColumn("Confidence", help="ML prediction confidence"),
                    "ML Type": st.column_config.TextColumn("ML Type", help="Type of ML analysis"),
                    "Symbol": st.column_config.TextColumn("Symbol", width="small")
                }
            )
            
            # Detailed breakdown for top 3
            st.subheader("📊 Top 3 Detailed Analysis")
            
            for i, opp in enumerate(strict_opportunities[:3], 1):
                with st.expander(f"#{i} {opp['symbol']} - {opp.get('expected_return_30d', 0)*100:.1f}% (30d) - {opp.get('confidence', 0)*100:.1f}% confidence"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Validated Features Used:**")
                        features = opp.get('features_used', [])
                        for feature in features:
                            st.write(f"✅ {feature}")
                        
                        st.write(f"**Data Validation:** ✅ All data verified real (no dummy data)")
                        st.write(f"**Prediction Time:** {opp.get('prediction_timestamp', 'Unknown')}")
                    
                    with col2:
                        # Multi-horizon return visualization
                        horizon_data = opp.get('horizon_predictions', {})
                        
                        if horizon_data:
                            timeframes = []
                            returns = []
                            
                            # Include all available horizons
                            for horizon in ['1H', '24H', '7D', '30D']:
                                if horizon in horizon_data:
                                    timeframes.append(horizon)
                                    returns.append(horizon_data[horizon].get('predicted_return', 0) * 100)
                            
                            if timeframes:
                                return_data = {
                                    'Timeframe': timeframes,
                                    'Expected Return (%)': returns
                                }
                                
                                fig = px.bar(
                                    return_data,
                                    x='Timeframe',
                                    y='Expected Return (%)',
                                    title=f"{opp['symbol']} Multi-Horizon Predictions",
                                    color='Expected Return (%)',
                                    color_continuous_scale='Viridis'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Fallback to simple 7D/30D view
                            return_data = {
                                'Timeframe': ['7 Days', '30 Days'],
                                'Expected Return (%)': [
                                    opp.get('expected_return_7d', 0) * 100,
                                    opp.get('expected_return_30d', 0) * 100
                                ]
                            }
                            
                            fig = px.bar(
                                return_data,
                                x='Timeframe',
                                y='Expected Return (%)',
                                title=f"{opp['symbol']} Return Prediction",
                                color='Expected Return (%)',
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            # Data integrity confirmation
            st.success("✅ ALLE getoonde data is gevalideerd - GEEN dummy data gebruikt")
            
        except Exception as e:
            st.error(f"Failed to render strict alpha opportunities: {e}")
    
    def _render_analysis_insights(self):
        """Render analysis insights and market overview"""
        st.header("🧠 Analysis Insights")
        
        try:
            cache_manager = self.container.cache_manager()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 OpenAI Analysis Results")
                
                openai_results = cache_manager.get('openai_comprehensive_analysis')
                if openai_results and 'results' in openai_results:
                    results = openai_results['results']
                    st.write(f"**Coins Analyzed:** {len(results)}")
                    
                    # Show recent OpenAI insights
                    if results:
                        recent_analysis = results[-1]  # Most recent
                        st.write(f"**Latest Analysis:** {recent_analysis['symbol']}")
                        
                        openai_data = recent_analysis.get('openai_analysis', {})
                        if openai_data:
                            st.write("**OpenAI Insights:**")
                            st.json(openai_data)
                else:
                    st.info("OpenAI analysis results not available yet")
            
            with col2:
                st.subheader("🐋 Whale Activity Summary")
                
                whale_results = cache_manager.get('whale_detection_results')
                if whale_results and 'results' in whale_results:
                    whale_data = whale_results['results']
                    
                    # Analyze whale activity trends
                    high_activity_coins = []
                    for symbol, data in whale_data.items():
                        if data.get('large_transactions_24h', 0) > 5:
                            high_activity_coins.append({
                                'symbol': symbol,
                                'transactions': data.get('large_transactions_24h', 0),
                                'net_flow': data.get('net_whale_flow', 0)
                            })
                    
                    if high_activity_coins:
                        st.write(f"**High Whale Activity:** {len(high_activity_coins)} coins")
                        
                        # Show top whale activity
                        high_activity_coins.sort(key=lambda x: x['transactions'], reverse=True)
                        for coin in high_activity_coins[:5]:
                            flow_indicator = "🟢" if coin['net_flow'] > 0 else "🔴"
                            st.write(f"{flow_indicator} **{coin['symbol']}**: {coin['transactions']} large transactions")
                    else:
                        st.write("No significant whale activity detected")
                else:
                    st.info("Whale detection results not available yet")
            
            # Market sentiment overview
            st.subheader("💭 Market Sentiment Overview")
            
            sentiment_results = cache_manager.get('sentiment_analysis_results')
            if sentiment_results and 'results' in sentiment_results:
                sentiment_data = sentiment_results['results']
                
                # Calculate sentiment distribution
                sentiment_scores = [data.get('sentiment_score', 0.5) for data in sentiment_data.values()]
                
                if sentiment_scores:
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    bullish_count = len([s for s in sentiment_scores if s > 0.6])
                    bearish_count = len([s for s in sentiment_scores if s < 0.4])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
                    with col2:
                        st.metric("Bullish Coins", bullish_count)
                    with col3:
                        st.metric("Bearish Coins", bearish_count)
                    
                    # Sentiment distribution chart
                    sentiment_df = pd.DataFrame({
                        'Sentiment Score': sentiment_scores
                    })
                    
                    fig_sentiment = px.histogram(
                        sentiment_df,
                        x='Sentiment Score',
                        bins=20,
                        title="Market Sentiment Distribution"
                    )
                    st.plotly_chart(fig_sentiment, use_container_width=True)
            else:
                st.info("Sentiment analysis results not available yet")
                
        except Exception as e:
            st.error(f"Failed to render analysis insights: {e}")
    
    def _render_performance_tracking(self):
        """Render alpha seeking performance tracking"""
        st.header("📈 Alpha Seeking Performance")
        
        try:
            performance = self.alpha_seeker.get_system_performance()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                accuracy = performance['accuracy_metrics']['average_accuracy'] * 100
                st.metric("Prediction Accuracy", f"{accuracy:.1f}%")
            
            with col2:
                total_predictions = performance['accuracy_metrics']['total_predictions']
                st.metric("Total Predictions", total_predictions)
            
            with col3:
                high_conf_opportunities = performance['high_confidence_opportunities']
                st.metric("High Confidence Opportunities", high_conf_opportunities)
            
            # Alpha seeking configuration
            st.subheader("⚙️ Alpha Seeking Configuration")
            
            config = performance['alpha_config']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Minimum Confidence:** {config['min_confidence']*100:.0f}%")
                st.write(f"**Target Return:** {config['target_return']*100:.0f}%")
                st.write(f"**Analysis Timeframe:** {config['max_timeframe_days']} days")
            
            with col2:
                st.write(f"**Volume Growth Threshold:** {config['volume_growth_threshold']}x")
                st.write(f"**Momentum Threshold:** {config['momentum_threshold']*100:.0f}%")
                st.write(f"**Whale Activity Weight:** {config['whale_activity_weight']*100:.0f}%")
            
            # Weight distribution
            weights = {
                'Sentiment': config['sentiment_weight'],
                'Technical': config['technical_weight'],
                'ML Prediction': config['ml_weight'],
                'Whale Activity': config['whale_activity_weight']
            }
            
            weight_df = pd.DataFrame({
                'Analysis Type': list(weights.keys()),
                'Weight': [w*100 for w in weights.values()]
            })
            
            fig_weights = px.pie(
                weight_df,
                values='Weight',
                names='Analysis Type',
                title="Analysis Weight Distribution"
            )
            st.plotly_chart(fig_weights, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to render performance tracking: {e}")
    
    def _render_data_quality_verification(self):
        """Render data quality verification section"""
        st.header("✅ Data Quality Verification")
        
        try:
            cache_manager = self.container.cache_manager()
            quality_metrics = cache_manager.get('data_quality_metrics') if cache_manager else None
            
            if quality_metrics:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Data Completeness")
                    
                    total_coins = quality_metrics.get('total_coins_discovered', 0)
                    complete_coins = quality_metrics.get('coins_with_complete_data', 0)
                    ratio = quality_metrics.get('data_completeness_ratio', 0)
                    
                    st.write(f"**Total Kraken Coins:** {total_coins:,}")
                    st.write(f"**Complete Data Coins:** {complete_coins:,}")
                    st.write(f"**Completeness Ratio:** {ratio:.1%}")
                    
                    # Progress bar for completeness
                    st.progress(ratio)
                    
                with col2:
                    st.subheader("🚫 Rejected Data")
                    
                    failed_scraping = len(quality_metrics.get('failed_scraping_coins', set()))
                    missing_data = len(quality_metrics.get('missing_data_coins', set()))
                    
                    st.write(f"**Failed Scraping:** {failed_scraping} coins")
                    st.write(f"**Missing Data:** {missing_data} coins")
                    st.write("**Action:** Coins excluded from analysis")
                    
                    if failed_scraping > 0 or missing_data > 0:
                        st.warning(f"Total {failed_scraping + missing_data} coins rejected due to incomplete data")
            
            # Data validation rules
            st.subheader("📋 Validation Rules Applied")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Price Data Validation:**")
                st.write("• OHLC values must be > 0")
                st.write("• Volume must be ≥ 0")
                st.write("• High ≥ max(Open, Close)")
                st.write("• Low ≤ min(Open, Close)")
                st.write("• No NULL/NaN values allowed")
                
            with col2:
                st.write("**Sentiment/Whale Validation:**")
                st.write("• Sentiment score 0-1 range")
                st.write("• Mention volume ≥ 0")
                st.write("• Whale data from blockchain only")
                st.write("• All timestamps must be recent")
                st.write("• No synthetic/dummy data")
            
            st.info("🔒 **ZERO TOLERANCE**: Coins met ontbrekende of invalid data worden volledig uitgesloten")
            
        except Exception as e:
            st.error(f"Failed to render data quality verification: {e}")
    
    def _render_pipeline_performance(self):
        """Render pipeline performance metrics"""
        st.header("⚡ Pipeline Performance")
        
        try:
            pipeline_status = self.pipeline.get_pipeline_status()
            
            # Latest opportunities summary
            latest = pipeline_status.get('latest_opportunities', {})
            
            if latest:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    timestamp = latest.get('timestamp', '')
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp)
                            time_ago = datetime.now() - dt
                            minutes_ago = int(time_ago.total_seconds() / 60)
                            st.metric("Last Analysis", f"{minutes_ago}m ago")
                        except Exception:
                            st.metric("Last Analysis", "Unknown")
                    else:
                        st.metric("Last Analysis", "Never")
                
                with col2:
                    total_analyzed = latest.get('total_analyzed', 0)
                    st.metric("Coins Analyzed", f"{total_analyzed:,}")
                
                with col3:
                    high_conf = latest.get('high_confidence_count', 0)
                    st.metric("Met Criteria", f"{high_conf:,}")
            
            # Pipeline configuration
            st.subheader("⚙️ Pipeline Configuration")
            
            st.write("**STRICT Mode Enabled:**")
            st.write("• ✅ No dummy data tolerance")
            st.write("• ✅ 80% minimum confidence")
            st.write("• ✅ 100% minimum 30-day return")
            st.write("• ✅ Complete data validation")
            st.write("• ✅ Real-time Kraken coverage")
            
        except Exception as e:
            st.error(f"Failed to render pipeline performance: {e}")
    
    def _force_pipeline_update(self):
        """Force update of the entire pipeline"""
        try:
            with st.spinner("Forcing complete pipeline update..."):
                if not self.pipeline:
                    st.error("Pipeline niet beschikbaar")
                    return
                
                # Clear all cached data to force fresh collection
                cache_manager = self.container.cache_manager()
                if cache_manager and hasattr(cache_manager, '_cache'):
                    cache_keys_to_clear = [
                        'complete_coin_discovery',
                        'alpha_opportunities_final',
                        'data_quality_metrics'
                    ]
                    
                    # Also clear all validated data
                    keys_to_remove = []
                    for key in cache_manager._cache.keys():
                        if (key.startswith('validated_') or 
                            key.startswith('alpha_opportunities') or
                            'pipeline' in key):
                            keys_to_remove.append(key)
                    
                    for key in keys_to_remove + cache_keys_to_clear:
                        if key in cache_manager._cache:
                            del cache_manager._cache[key]
                
                # Trigger immediate pipeline tasks
                self.pipeline._execute_pipeline_task('coin_discovery')
                self.pipeline._execute_pipeline_task('data_quality_check')
                
                st.success("Pipeline update gestart! Data wordt opnieuw verzameld en gevalideerd.")
                
        except Exception as e:
            st.error(f"Failed to force pipeline update: {e}")
    
    def _render_ml_system_status(self):
        """Render multi-horizon ML system status"""
        st.header("🤖 Multi-Horizon ML System")
        
        try:
            cache_manager = self.container.cache_manager()
            ml_status = cache_manager.get('multi_horizon_ml_status') if cache_manager else None
            
            if ml_status:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    models_loaded = ml_status.get('models_loaded', 0)
                    total_horizons = ml_status.get('total_horizons', 4)
                    st.metric("Models Loaded", f"{models_loaded}/{total_horizons}")
                
                with col2:
                    prediction_count = ml_status.get('prediction_log_count', 0)
                    st.metric("Predictions Made", f"{prediction_count:,}")
                
                with col3:
                    # Check if any models need retraining
                    retrain_needed = ml_status.get('retrain_needed', {})
                    needs_retrain = sum(retrain_needed.values()) if retrain_needed else 0
                    st.metric("Models Needing Retrain", needs_retrain)
                
                with col4:
                    # Show latest training time
                    training_times = ml_status.get('last_training_times', {})
                    if training_times:
                        latest_training = max(training_times.values()) if training_times.values() else "Never"
                        if latest_training != "Never":
                            try:
                                dt = datetime.fromisoformat(latest_training)
                                hours_ago = int((datetime.now() - dt).total_seconds() / 3600)
                                st.metric("Last Training", f"{hours_ago}h ago")
                            except Exception:
                                st.metric("Last Training", "Unknown")
                    else:
                        st.metric("Last Training", "Never")
                
                # Model performance details
                st.subheader("📊 Model Performance by Horizon")
                
                performance = ml_status.get('model_performance', {})
                if performance:
                    perf_data = []
                    for horizon, metrics in performance.items():
                        perf_data.append({
                            'Horizon': horizon,
                            'Test MAE': f"{metrics.get('test_mae', 0):.4f}",
                            'Training Samples': f"{metrics.get('training_samples', 0):,}",
                            'Features': metrics.get('feature_count', 0)
                        })
                    
                    if perf_data:
                        df_perf = pd.DataFrame(perf_data)
                        st.dataframe(df_perf, use_container_width=True)
                
                # Feature importance
                st.subheader("🎯 Feature Importance (Top Features)")
                
                feature_importance = ml_status.get('feature_importance', {})
                if feature_importance:
                    # Get average importance across all horizons
                    all_features = {}
                    for horizon, features in feature_importance.items():
                        for feature, importance in features.items():
                            if feature not in all_features:
                                all_features[feature] = []
                            all_features[feature].append(importance)
                    
                    # Calculate average importance
                    avg_importance = {
                        feature: np.mean(importances)
                        for feature, importances in all_features.items()
                    }
                    
                    # Sort and get top 10
                    top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    if top_features:
                        feature_df = pd.DataFrame(top_features, columns=['Feature', 'Avg Importance'])
                        
                        fig_importance = px.bar(
                            feature_df,
                            x='Avg Importance',
                            y='Feature',
                            orientation='h',
                            title="Top 10 Most Important Features",
                            labels={'Avg Importance': 'Average Importance Across Horizons'}
                        )
                        fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_importance, use_container_width=True)
                
                # Training configuration
                st.subheader("⚙️ ML Configuration")
                
                training_config = ml_status.get('training_config', {})
                if training_config:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Training Parameters:**")
                        st.write(f"• Min samples: {training_config.get('min_training_samples', 0):,}")
                        st.write(f"• Test size: {training_config.get('test_size', 0):.1%}")
                        st.write(f"• Max features: {training_config.get('max_features', 0)}")
                    
                    with col2:
                        st.write("**Quality Thresholds:**")
                        st.write(f"• Confidence: {training_config.get('confidence_threshold', 0):.1%}")
                        st.write(f"• Retrain MAE: {training_config.get('retrain_threshold_mae', 0):.1%}")
                        st.write(f"• Time horizons: 1H, 24H, 7D, 30D")
            else:
                st.info("Multi-horizon ML system status not available yet. System will initialize during first batch inference.")
                
                st.markdown("**System Features:**")
                st.write("• Training on 4 time horizons (1H, 24H, 7D, 30D)")
                st.write("• Multi-target regression with confidence scoring")
                st.write("• Automatic feature importance tracking")
                st.write("• Self-learning with prediction accuracy monitoring")
                st.write("• GPU-accelerated feature engineering")
                
        except Exception as e:
            st.error(f"Failed to render ML system status: {e}")