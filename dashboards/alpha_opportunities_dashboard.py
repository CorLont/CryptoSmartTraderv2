"""
CryptoSmartTrader V2 - Alpha Opportunities Dashboard
Displays high-growth potential cryptocurrencies with 500%+ return potential
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import time

class AlphaOpportunitiesDashboard:
    """Dashboard for alpha opportunities with high return potential"""
    
    def __init__(self, container):
        self.container = container
        
        # Initialize components
        try:
            # Import alpha seeker
            from core.alpha_seeker import AlphaSeeker
            from core.comprehensive_analyzer import ComprehensiveAnalyzer
            
            self.alpha_seeker = AlphaSeeker(
                config_manager=container.config(),
                cache_manager=container.cache_manager(),
                openai_analyzer=container.openai_analyzer()
            )
            
            self.comprehensive_analyzer = ComprehensiveAnalyzer(container)
            
            # Auto-start background analysis
            if not hasattr(st.session_state, 'alpha_analysis_started'):
                self.comprehensive_analyzer.start_continuous_analysis()
                st.session_state.alpha_analysis_started = True
                
        except Exception as e:
            st.error(f"Failed to initialize alpha components: {e}")
            
    def render(self):
        """Render the alpha opportunities dashboard"""
        st.title("🚀 Alpha Opportunities - 500%+ Return Potential")
        st.markdown("**Target: 500% rendement binnen 6 maanden door snelle groeiers vroeg te spotten**")
        
        # Auto-refresh controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**Real-time analysis van ALLE Kraken cryptocurrencies met OpenAI intelligence**")
        with col2:
            auto_refresh = st.checkbox("Auto Refresh (60s)", value=True)
        with col3:
            if st.button("🔄 Force Analysis Update"):
                self._force_analysis_update()
                st.rerun()
        
        if auto_refresh:
            time.sleep(60)
            st.rerun()
        
        # System status
        self._render_system_status()
        
        # Alpha opportunities table
        self._render_alpha_opportunities()
        
        # Analysis insights
        self._render_analysis_insights()
        
        # Performance tracking
        self._render_performance_tracking()
    
    def _render_system_status(self):
        """Render system analysis status"""
        st.header("📊 System Analysis Status")
        
        # Get analysis status
        try:
            analysis_status = self.comprehensive_analyzer.get_analysis_status()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                status_indicator = "🟢 Active" if analysis_status.get('analysis_active', False) else "🔴 Stopped"
                st.metric("Analysis System", status_indicator)
            
            with col2:
                market_data = analysis_status.get('data_availability', {})
                market_coins = market_data.get('market_scan', 0)
                st.metric("Coins Scanned", f"{market_coins:,}")
            
            with col3:
                sentiment_coins = market_data.get('sentiment_analysis', 0)
                st.metric("Sentiment Analysis", f"{sentiment_coins:,}")
            
            with col4:
                openai_coins = market_data.get('openai_comprehensive_analysis', 0)
                st.metric("OpenAI Analyzed", f"{openai_coins:,}")
            
            # Last analysis times
            st.subheader("📅 Last Analysis Times")
            last_analyses = analysis_status.get('last_analyses', {})
            
            analysis_types = {
                'market_scan': '📈 Market Scan',
                'sentiment_analysis': '💭 Sentiment Analysis',
                'whale_detection': '🐋 Whale Detection',
                'ml_prediction': '🤖 ML Predictions',
                'openai_comprehensive_analysis': '🧠 OpenAI Analysis'
            }
            
            for analysis_type, display_name in analysis_types.items():
                if analysis_type in last_analyses:
                    try:
                        last_time = datetime.fromisoformat(last_analyses[analysis_type])
                        time_ago = datetime.now() - last_time
                        minutes_ago = int(time_ago.total_seconds() / 60)
                        st.write(f"**{display_name}:** {minutes_ago} minutes ago")
                    except:
                        st.write(f"**{display_name}:** Unknown")
                else:
                    st.write(f"**{display_name}:** Not run yet")
                    
        except Exception as e:
            st.error(f"Failed to get analysis status: {e}")
    
    def _render_alpha_opportunities(self):
        """Render alpha opportunities table with 80% confidence minimum"""
        st.header("🎯 Alpha Opportunities (80%+ Confidence)")
        st.markdown("**Alleen cryptocurrencies met 80%+ confidence en verwacht rendement >100% worden getoond**")
        
        try:
            # Get alpha opportunities
            opportunities = self.alpha_seeker.get_top_alpha_opportunities(min_confidence=0.80)
            
            if not opportunities:
                st.info("Geen alpha opportunities gevonden met 80%+ confidence. Het systeem analyseert continu alle beschikbare coins.")
                st.markdown("**Huidige criteria:**")
                st.write("• Minimaal 80% confidence level")
                st.write("• Verwacht rendement >100% binnen 30 dagen")
                st.write("• Gebaseerd op technische analyse, sentiment, whale activity en ML predictions")
                return
            
            # Filter for significant return potential
            high_return_opportunities = [
                opp for opp in opportunities
                if opp.get('expected_returns', {}).get('30_day', 0) >= 1.0  # 100%+ return
            ]
            
            if not high_return_opportunities:
                st.warning("Geen opportunities gevonden met >100% verwacht rendement binnen 30 dagen.")
                st.write(f"Totaal {len(opportunities)} opportunities met lagere returns beschikbaar.")
                return
            
            # Create opportunities table
            table_data = []
            for opp in high_return_opportunities:
                returns = opp.get('expected_returns', {})
                
                table_data.append({
                    'Symbol': opp['symbol'],
                    '7 Days': f"{returns.get('7_day', 0)*100:.1f}%",
                    '30 Days': f"{returns.get('30_day', 0)*100:.1f}%",
                    '180 Days': f"{returns.get('180_day', 0)*100:.1f}%",
                    'Confidence': f"{opp.get('confidence', 0)*100:.1f}%",
                    'Alpha Score': f"{opp.get('alpha_score', 0):.2f}",
                    'Key Factors': len(opp.get('confidence_factors', [])),
                    'Risk Factors': len(opp.get('risk_factors', []))
                })
            
            if table_data:
                df_opportunities = pd.DataFrame(table_data)
                
                st.subheader(f"🏆 Top {len(table_data)} Alpha Opportunities")
                st.dataframe(
                    df_opportunities,
                    use_container_width=True,
                    column_config={
                        "30 Days": st.column_config.TextColumn("30 Days", help="Expected return in 30 days"),
                        "Confidence": st.column_config.TextColumn("Confidence", help="Prediction confidence level"),
                        "Alpha Score": st.column_config.NumberColumn("Alpha Score", min_value=0, max_value=1)
                    }
                )
                
                # Detailed analysis for top opportunities
                st.subheader("📋 Detailed Analysis")
                
                for i, opp in enumerate(high_return_opportunities[:5], 1):
                    with st.expander(f"#{i} {opp['symbol']} - {opp.get('expected_returns', {}).get('30_day', 0)*100:.1f}% (30d)"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Confidence Factors:**")
                            for factor in opp.get('confidence_factors', []):
                                st.write(f"✅ {factor}")
                        
                        with col2:
                            st.write("**Risk Factors:**")
                            for risk in opp.get('risk_factors', []):
                                st.write(f"⚠️ {risk}")
                            
                            if not opp.get('risk_factors'):
                                st.write("✅ No significant risk factors identified")
                        
                        # Return breakdown
                        returns = opp.get('expected_returns', {})
                        return_data = {
                            'Timeframe': ['7 Days', '30 Days', '180 Days'],
                            'Expected Return': [
                                returns.get('7_day', 0) * 100,
                                returns.get('30_day', 0) * 100,
                                returns.get('180_day', 0) * 100
                            ]
                        }
                        
                        fig_returns = px.bar(
                            return_data,
                            x='Timeframe',
                            y='Expected Return',
                            title=f"{opp['symbol']} Expected Returns",
                            labels={'Expected Return': 'Return %'}
                        )
                        st.plotly_chart(fig_returns, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to render alpha opportunities: {e}")
    
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
    
    def _force_analysis_update(self):
        """Force update of all analysis components"""
        try:
            with st.spinner("Forcing analysis update..."):
                # Trigger market scan
                market_scanner = self.container.market_scanner()
                market_scanner.force_full_scan()
                
                # Clear cache to force fresh analysis
                cache_manager = self.container.cache_manager()
                keys_to_clear = [
                    'sentiment_analysis_results',
                    'whale_detection_results',
                    'ml_prediction_results',
                    'openai_comprehensive_analysis'
                ]
                
                for key in keys_to_clear:
                    if hasattr(cache_manager, '_cache') and key in cache_manager._cache:
                        del cache_manager._cache[key]
                
                st.success("Analysis update completed!")
                
        except Exception as e:
            st.error(f"Failed to force analysis update: {e}")