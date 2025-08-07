"""
CryptoSmartTrader V2 - Analysis Control Dashboard
Dashboard for controlling ML analysis and social media scraping
"""

import streamlit as st
import asyncio
import subprocess
import sys
import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from containers import ApplicationContainer
from core.daily_analysis_scheduler import DailyAnalysisScheduler


class AnalysisControlDashboard:
    """Dashboard for controlling analysis services"""
    
    def __init__(self, container: ApplicationContainer):
        self.container = container
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.orchestrator = container.orchestrator()
        self.config_manager = container.config()
        self.health_monitor = container.health_monitor()
        self.cache_manager = container.cache_manager()
        
        # Initialize daily scheduler
        self.daily_scheduler = DailyAnalysisScheduler(
            self.config_manager,
            self.cache_manager,
            self.health_monitor
        )
        
        # Service status tracking
        if 'analysis_services' not in st.session_state:
            st.session_state.analysis_services = {
                'ml_analysis': {'status': 'stopped', 'last_run': None, 'results': None},
                'social_scraper': {'status': 'stopped', 'last_run': None, 'results': None},
                'technical_analysis': {'status': 'stopped', 'last_run': None, 'results': None},
                'sentiment_analysis': {'status': 'stopped', 'last_run': None, 'results': None}
            }
    
    def render(self):
        """Render the analysis control dashboard"""
        st.title("🎛️ Analysis Control Center")
        st.markdown("---")
        
        # Daily Analysis Overview
        self._render_daily_analysis_overview()
        
        # Service status overview
        self._render_service_status()
        
        # Analysis controls
        col1, col2 = st.columns(2)
        
    def _render_daily_analysis_overview(self):
        """Render daily analysis overview"""
        st.subheader("📈 Daily Analysis Overview")
        
        # Get daily analysis status
        daily_status = self.daily_scheduler.get_daily_analysis_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            scheduler_status = "🟢 Active" if daily_status["scheduler_running"] else "🔴 Stopped"
            st.metric("Daily Scheduler", scheduler_status)
            
            if st.button("Start Daily Analysis" if not daily_status["scheduler_running"] else "Stop Daily Analysis"):
                if daily_status["scheduler_running"]:
                    self.daily_scheduler.stop_daily_analysis()
                    st.success("Daily analysis stopped")
                else:
                    self.daily_scheduler.start_daily_analysis()
                    st.success("Daily analysis started")
                st.rerun()
        
        with col2:
            ml_status = daily_status["services"]["ml_analysis"]
            ml_indicator = "🟢 Running" if ml_status == "running" else "🔴 Stopped"
            st.metric("ML Service", ml_indicator)
        
        with col3:
            social_status = daily_status["services"]["social_scraper"]
            social_indicator = "🟢 Running" if social_status == "running" else "🔴 Stopped"
            st.metric("Social Scraper", social_indicator)
        
        with col4:
            # Show latest report status
            latest_report = self.daily_scheduler.get_latest_daily_report()
            if latest_report and latest_report.get("daily_summary", {}).get("generated_at"):
                st.metric("Latest Report", "📊 Available")
            else:
                st.metric("Latest Report", "⏳ Pending")
        
        # Daily progress
        if daily_status["analysis_data"]:
            self._render_daily_progress(daily_status["analysis_data"])
        
        st.markdown("---")
    
    def _render_daily_progress(self, daily_data: Dict):
        """Render daily analysis progress"""
        st.subheader("📊 Today's Progress")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**ML Analysis**")
            ml_data = daily_data.get("ml_analysis", {})
            ml_status = ml_data.get("status", "pending")
            
            if ml_status == "active":
                predictions_count = len(ml_data.get("predictions", []))
                st.success(f"✅ {predictions_count} predictions generated")
                if ml_data.get("last_update"):
                    st.caption(f"Last update: {ml_data['last_update'][:16]}")
            elif ml_status == "running":
                st.info("🔄 Analysis in progress...")
            else:
                st.warning("⏳ Waiting to start...")
        
        with col2:
            st.write("**Social Sentiment**")
            social_data = daily_data.get("social_sentiment", {})
            social_status = social_data.get("status", "pending")
            
            if social_status == "active":
                reddit_count = len(social_data.get("reddit_data", []))
                twitter_count = len(social_data.get("twitter_data", []))
                st.success(f"✅ {reddit_count + twitter_count} posts analyzed")
                if social_data.get("last_update"):
                    st.caption(f"Last update: {social_data['last_update'][:16]}")
            elif social_status == "running":
                st.info("🔄 Scraping in progress...")
            else:
                st.warning("⏳ Waiting to start...")
        
        with col3:
            st.write("**Technical Analysis**")
            tech_data = daily_data.get("technical_analysis", {})
            tech_status = tech_data.get("status", "pending")
            
            if tech_status == "active":
                indicators_count = len(tech_data.get("indicators", {}))
                signals_count = len(tech_data.get("signals", []))
                st.success(f"✅ {indicators_count} indicators, {signals_count} signals")
                if tech_data.get("last_update"):
                    st.caption(f"Last update: {tech_data['last_update'][:16]}")
            elif tech_status == "running":
                st.info("🔄 Analysis in progress...")
            else:
                st.warning("⏳ Waiting to start...")
        
        # Show daily summary if available
        daily_summary = daily_data.get("daily_summary", {})
        if daily_summary.get("generated_at"):
            st.subheader("📋 Daily Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend = daily_summary.get("market_trend", "neutral")
                trend_emoji = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(trend, "➡️")
                st.metric("Market Trend", f"{trend_emoji} {trend.title()}")
            
            with col2:
                sentiment = daily_summary.get("sentiment_overview", "neutral")
                sentiment_emoji = {"positive": "😊", "negative": "😔", "neutral": "😐"}.get(sentiment, "😐")
                st.metric("Sentiment", f"{sentiment_emoji} {sentiment.title()}")
            
            with col3:
                confidence = daily_summary.get("confidence_score", 0.5)
                st.metric("Confidence", f"{confidence:.1%}")
            
            # Show recommendations
            recommendations = daily_summary.get("recommendations", [])
            if recommendations:
                st.write("**Recommendations:**")
                for rec in recommendations:
                    st.write(f"• {rec}")
        
        # Quick actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 View Detailed Report"):
                self._show_detailed_daily_report(daily_data)
        
        with col2:
            if st.button("💾 Export Report"):
                self._export_daily_report(daily_data)
        
        with col3:
            if st.button("🔄 Force Update"):
                self.daily_scheduler._collect_analysis_results()
                st.success("Analysis results updated")
                st.rerun()
    
    def _show_detailed_daily_report(self, daily_data: Dict):
        """Show detailed daily report in expandable section"""
        with st.expander("📊 Detailed Daily Report", expanded=True):
            st.json(daily_data)
    
    def _export_daily_report(self, daily_data: Dict):
        """Export daily report as JSON"""
        import json
        
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"daily_report_{today}.json"
        
        json_data = json.dumps(daily_data, indent=2, default=str)
        
        st.download_button(
            label="💾 Download Report",
            data=json_data,
            file_name=filename,
            mime="application/json"
        )
        
        with col1:
            self._render_ml_analysis_controls()
            self._render_technical_analysis_controls()
        
        with col2:
            self._render_social_scraping_controls()
            self._render_sentiment_analysis_controls()
        
        # Analysis results
        st.markdown("---")
        self._render_analysis_results()
        
        # Background services management
        st.markdown("---")
        self._render_background_services()
    
    def _render_service_status(self):
        """Render service status overview"""
        st.subheader("📊 Service Status Overview")
        
        # Create status metrics
        col1, col2, col3, col4 = st.columns(4)
        
        services = st.session_state.analysis_services
        
        with col1:
            ml_status = services['ml_analysis']['status']
            status_color = "🟢" if ml_status == "running" else "🔴" if ml_status == "error" else "🟡"
            st.metric("ML Analysis", f"{status_color} {ml_status.title()}")
        
        with col2:
            social_status = services['social_scraper']['status']
            status_color = "🟢" if social_status == "running" else "🔴" if social_status == "error" else "🟡"
            st.metric("Social Scraper", f"{status_color} {social_status.title()}")
        
        with col3:
            tech_status = services['technical_analysis']['status']
            status_color = "🟢" if tech_status == "running" else "🔴" if tech_status == "error" else "🟡"
            st.metric("Technical Analysis", f"{status_color} {tech_status.title()}")
        
        with col4:
            sentiment_status = services['sentiment_analysis']['status']
            status_color = "🟢" if sentiment_status == "running" else "🔴" if sentiment_status == "error" else "🟡"
            st.metric("Sentiment Analysis", f"{status_color} {sentiment_status.title()}")
    
    def _render_ml_analysis_controls(self):
        """Render ML analysis control panel"""
        st.subheader("🤖 ML Analysis")
        
        with st.container():
            st.markdown("**Machine Learning Predictions**")
            
            # Configuration options
            prediction_horizons = st.multiselect(
                "Prediction Horizons",
                options=["1h", "4h", "1d", "7d", "30d"],
                default=["1h", "1d", "7d"],
                key="ml_horizons"
            )
            
            num_coins = st.slider(
                "Number of coins to analyze",
                min_value=10,
                max_value=100,
                value=30,
                key="ml_num_coins"
            )
            
            # Control buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Start ML Analysis", key="start_ml"):
                    self._start_ml_analysis(prediction_horizons, num_coins)
            
            with col2:
                if st.button("⏹️ Stop ML Analysis", key="stop_ml"):
                    self._stop_ml_analysis()
            
            with col3:
                if st.button("🔄 Quick Analysis", key="quick_ml"):
                    self._run_quick_ml_analysis(prediction_horizons[:2], min(num_coins, 10))
            
            # Show last run info
            last_run = st.session_state.analysis_services['ml_analysis']['last_run']
            if last_run:
                st.info(f"Last run: {last_run}")
    
    def _render_social_scraping_controls(self):
        """Render social media scraping control panel"""
        st.subheader("📱 Social Media Scraping")
        
        with st.container():
            st.markdown("**Reddit & Twitter Monitoring**")
            
            # Configuration options
            platforms = st.multiselect(
                "Platforms",
                options=["Reddit", "Twitter"],
                default=["Reddit", "Twitter"],
                key="social_platforms"
            )
            
            update_interval = st.selectbox(
                "Update Interval",
                options=["5 minutes", "15 minutes", "30 minutes", "1 hour"],
                index=0,
                key="social_interval"
            )
            
            # Control buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Start Scraping", key="start_social"):
                    self._start_social_scraping(platforms, update_interval)
            
            with col2:
                if st.button("⏹️ Stop Scraping", key="stop_social"):
                    self._stop_social_scraping()
            
            with col3:
                if st.button("📊 One-time Scrape", key="onetime_social"):
                    self._run_onetime_scraping(platforms)
            
            # Show last run info
            last_run = st.session_state.analysis_services['social_scraper']['last_run']
            if last_run:
                st.info(f"Last run: {last_run}")
    
    def _render_technical_analysis_controls(self):
        """Render technical analysis control panel"""
        st.subheader("📈 Technical Analysis")
        
        with st.container():
            st.markdown("**Technical Indicators & Signals**")
            
            # Configuration options
            indicators = st.multiselect(
                "Indicators",
                options=["RSI", "MACD", "Bollinger Bands", "Moving Averages", "Stochastic", "ADX"],
                default=["RSI", "MACD", "Bollinger Bands"],
                key="tech_indicators"
            )
            
            timeframes = st.multiselect(
                "Timeframes",
                options=["1h", "4h", "1d", "1w"],
                default=["1h", "1d"],
                key="tech_timeframes"
            )
            
            # Control buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Start Technical Analysis", key="start_tech"):
                    self._start_technical_analysis(indicators, timeframes)
            
            with col2:
                if st.button("⏹️ Stop Technical Analysis", key="stop_tech"):
                    self._stop_technical_analysis()
            
            # Show last run info
            last_run = st.session_state.analysis_services['technical_analysis']['last_run']
            if last_run:
                st.info(f"Last run: {last_run}")
    
    def _render_sentiment_analysis_controls(self):
        """Render sentiment analysis control panel"""
        st.subheader("💭 Sentiment Analysis")
        
        with st.container():
            st.markdown("**Market Sentiment & News Analysis**")
            
            # Configuration options
            sources = st.multiselect(
                "Data Sources",
                options=["Social Media", "News Articles", "Forum Posts", "Expert Analysis"],
                default=["Social Media", "News Articles"],
                key="sentiment_sources"
            )
            
            analysis_depth = st.selectbox(
                "Analysis Depth",
                options=["Basic (TextBlob)", "Advanced (OpenAI)", "Ensemble"],
                index=2,
                key="sentiment_depth"
            )
            
            # Control buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Start Sentiment Analysis", key="start_sentiment"):
                    self._start_sentiment_analysis(sources, analysis_depth)
            
            with col2:
                if st.button("⏹️ Stop Sentiment Analysis", key="stop_sentiment"):
                    self._stop_sentiment_analysis()
            
            # Show last run info
            last_run = st.session_state.analysis_services['sentiment_analysis']['last_run']
            if last_run:
                st.info(f"Last run: {last_run}")
    
    def _render_analysis_results(self):
        """Render analysis results section"""
        st.subheader("📋 Recent Analysis Results")
        
        # Tabs for different analysis types
        tab1, tab2, tab3, tab4 = st.tabs(["ML Predictions", "Social Sentiment", "Technical Signals", "Market Sentiment"])
        
        with tab1:
            self._show_ml_results()
        
        with tab2:
            self._show_social_results()
        
        with tab3:
            self._show_technical_results()
        
        with tab4:
            self._show_sentiment_results()
    
    def _render_background_services(self):
        """Render background services management"""
        st.subheader("⚙️ Background Services")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Service Management**")
            if st.button("🚀 Start All Services", key="start_all"):
                self._start_all_background_services()
            
            if st.button("⏹️ Stop All Services", key="stop_all"):
                self._stop_all_background_services()
        
        with col2:
            st.markdown("**System Information**")
            if st.button("📊 View System Status", key="system_status"):
                self._show_system_status()
            
            if st.button("📁 Open Data Folder", key="open_data"):
                self._open_data_folder()
        
        with col3:
            st.markdown("**Logs & Monitoring**")
            if st.button("📄 View Logs", key="view_logs"):
                self._show_logs()
            
            if st.button("🔄 Refresh Status", key="refresh_status"):
                self._refresh_service_status()
    
    def _start_ml_analysis(self, horizons, num_coins):
        """Start ML analysis with specified parameters"""
        try:
            # Update status
            st.session_state.analysis_services['ml_analysis']['status'] = 'starting'
            
            # Show progress
            with st.spinner("Starting ML Analysis..."):
                # Create analysis task
                ml_agent = self.container.ml_predictor_agent()
                
                # Run analysis in background thread
                def run_ml_analysis():
                    try:
                        # Run ML analysis simulation
                        results = {
                            "symbol": "BTC/USD",
                            "horizons": horizons,
                            "num_coins": num_coins,
                            "predictions": {
                                "1h": {"price": 45250, "confidence": 0.75},
                                "1d": {"price": 46100, "confidence": 0.68},
                                "7d": {"price": 48500, "confidence": 0.62}
                            },
                            "status": "simulated_analysis"
                        }
                        
                        # Update session state
                        st.session_state.analysis_services['ml_analysis'].update({
                            'status': 'completed',
                            'last_run': datetime.now().strftime("%H:%M:%S"),
                            'results': results
                        })
                        
                    except Exception as e:
                        self.logger.error(f"ML analysis failed: {e}")
                        st.session_state.analysis_services['ml_analysis'].update({
                            'status': 'error',
                            'last_run': datetime.now().strftime("%H:%M:%S"),
                            'error': str(e)
                        })
                
                # Start in background
                threading.Thread(target=run_ml_analysis, daemon=True).start()
                
                st.session_state.analysis_services['ml_analysis']['status'] = 'running'
                st.success(f"Started ML analysis for {num_coins} coins with {len(horizons)} prediction horizons")
        
        except Exception as e:
            st.error(f"Failed to start ML analysis: {e}")
            self.logger.error(f"ML analysis start failed: {e}")
    
    def _start_social_scraping(self, platforms, interval):
        """Start social media scraping"""
        try:
            st.session_state.analysis_services['social_scraper']['status'] = 'starting'
            
            with st.spinner("Starting Social Media Scraping..."):
                # Convert interval to seconds
                interval_seconds = {
                    "5 minutes": 300,
                    "15 minutes": 900,
                    "30 minutes": 1800,
                    "1 hour": 3600
                }.get(interval, 300)
                
                def run_social_scraping():
                    try:
                        sentiment_agent = self.container.sentiment_agent()
                        
                        # Run sentiment analysis simulation
                        results = {
                            "platforms": platforms,
                            "interval": interval_seconds,
                            "status": "simulated_scraping",
                            "sentiment_score": 0.65,
                            "posts_analyzed": 150
                        }
                        
                        st.session_state.analysis_services['social_scraper'].update({
                            'status': 'completed',
                            'last_run': datetime.now().strftime("%H:%M:%S"),
                            'results': results
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Social scraping failed: {e}")
                        st.session_state.analysis_services['social_scraper'].update({
                            'status': 'error',
                            'last_run': datetime.now().strftime("%H:%M:%S"),
                            'error': str(e)
                        })
                
                threading.Thread(target=run_social_scraping, daemon=True).start()
                
                st.session_state.analysis_services['social_scraper']['status'] = 'running'
                st.success(f"Started social media scraping for {', '.join(platforms)} with {interval} updates")
        
        except Exception as e:
            st.error(f"Failed to start social scraping: {e}")
            self.logger.error(f"Social scraping start failed: {e}")
    
    def _run_quick_ml_analysis(self, horizons, num_coins):
        """Run a quick ML analysis"""
        try:
            with st.spinner(f"Running quick analysis for {num_coins} coins..."):
                ml_agent = self.container.ml_predictor_agent()
                
                # Run quick ML prediction simulation
                results = {
                    "symbol": "BTC/USD",
                    "quick_prediction": {"price": 45180, "confidence": 0.72},
                    "timestamp": datetime.now().isoformat(),
                    "status": "simulated_quick_analysis"
                }
                
                st.session_state.analysis_services['ml_analysis'].update({
                    'status': 'completed',
                    'last_run': datetime.now().strftime("%H:%M:%S"),
                    'results': results
                })
                
                st.success("Quick analysis completed!")
                
                # Show quick results
                if results:
                    if isinstance(results, dict) and 'predictions' in results:
                        st.json(results['predictions'])
                    else:
                        st.json(results)
        
        except Exception as e:
            st.error(f"Quick analysis failed: {e}")
            self.logger.error(f"Quick ML analysis failed: {e}")
    
    def _run_onetime_scraping(self, platforms):
        """Run one-time social media scraping"""
        try:
            with st.spinner(f"Scraping {', '.join(platforms)}..."):
                sentiment_agent = self.container.sentiment_agent()
                
                # Run one-time sentiment analysis simulation
                results = {
                    "platforms": platforms,
                    "sentiment_analysis": {
                        "overall_sentiment": 0.68,
                        "positive_mentions": 85,
                        "negative_mentions": 23,
                        "neutral_mentions": 42
                    },
                    "status": "simulated_onetime_scraping"
                }
                
                st.session_state.analysis_services['social_scraper'].update({
                    'status': 'completed',
                    'last_run': datetime.now().strftime("%H:%M:%S"),
                    'results': results
                })
                
                st.success("One-time scraping completed!")
                
                # Show quick results
                if results:
                    st.json(results)
        
        except Exception as e:
            st.error(f"One-time scraping failed: {e}")
            self.logger.error(f"One-time scraping failed: {e}")
    
    def _stop_ml_analysis(self):
        """Stop ML analysis"""
        st.session_state.analysis_services['ml_analysis']['status'] = 'stopped'
        st.info("ML analysis stopped")
    
    def _stop_social_scraping(self):
        """Stop social media scraping"""
        st.session_state.analysis_services['social_scraper']['status'] = 'stopped'
        st.info("Social media scraping stopped")
    
    def _start_technical_analysis(self, indicators, timeframes):
        """Start technical analysis"""
        try:
            with st.spinner("Starting technical analysis..."):
                technical_agent = self.container.technical_agent()
                
                def run_technical():
                    try:
                        # Run technical analysis simulation
                        results = {
                            "symbol": "BTC/USD",
                            "indicators": indicators,
                            "timeframes": timeframes,
                            "technical_signals": {
                                "RSI": {"value": 65.4, "signal": "neutral"},
                                "MACD": {"value": 1.23, "signal": "bullish"},
                                "BB": {"position": "upper", "signal": "overbought"}
                            },
                            "overall_signal": "neutral_bullish",
                            "status": "simulated_technical_analysis"
                        }
                        
                        st.session_state.analysis_services['technical_analysis'].update({
                            'status': 'completed',
                            'last_run': datetime.now().strftime("%H:%M:%S"),
                            'results': results
                        })
                    except Exception as e:
                        st.session_state.analysis_services['technical_analysis'].update({
                            'status': 'error',
                            'error': str(e)
                        })
                
                threading.Thread(target=run_technical, daemon=True).start()
                st.session_state.analysis_services['technical_analysis']['status'] = 'running'
                st.success("Technical analysis started")
        
        except Exception as e:
            st.error(f"Failed to start technical analysis: {e}")
    
    def _stop_technical_analysis(self):
        """Stop technical analysis"""
        st.session_state.analysis_services['technical_analysis']['status'] = 'stopped'
        st.info("Technical analysis stopped")
    
    def _start_sentiment_analysis(self, sources, depth):
        """Start sentiment analysis"""
        try:
            with st.spinner("Starting sentiment analysis..."):
                sentiment_agent = self.container.sentiment_agent()
                
                def run_sentiment():
                    try:
                        # Run sentiment analysis simulation
                        results = {
                            "sources": sources,
                            "analysis_depth": depth,
                            "market_sentiment": {
                                "bitcoin": {"score": 0.72, "confidence": 0.85},
                                "ethereum": {"score": 0.68, "confidence": 0.82},
                                "overall_market": {"score": 0.65, "confidence": 0.78}
                            },
                            "news_analysis": {
                                "bullish_articles": 12,
                                "bearish_articles": 5,
                                "neutral_articles": 8
                            },
                            "status": "simulated_sentiment_analysis"
                        }
                        
                        st.session_state.analysis_services['sentiment_analysis'].update({
                            'status': 'completed',
                            'last_run': datetime.now().strftime("%H:%M:%S"),
                            'results': results
                        })
                    except Exception as e:
                        st.session_state.analysis_services['sentiment_analysis'].update({
                            'status': 'error',
                            'error': str(e)
                        })
                
                threading.Thread(target=run_sentiment, daemon=True).start()
                st.session_state.analysis_services['sentiment_analysis']['status'] = 'running'
                st.success("Sentiment analysis started")
        
        except Exception as e:
            st.error(f"Failed to start sentiment analysis: {e}")
    
    def _stop_sentiment_analysis(self):
        """Stop sentiment analysis"""
        st.session_state.analysis_services['sentiment_analysis']['status'] = 'stopped'
        st.info("Sentiment analysis stopped")
    
    def _show_ml_results(self):
        """Show ML analysis results"""
        results = st.session_state.analysis_services['ml_analysis'].get('results')
        
        if results:
            st.json(results)
        else:
            st.info("No ML analysis results available. Run an analysis to see results here.")
    
    def _show_social_results(self):
        """Show social media scraping results"""
        results = st.session_state.analysis_services['social_scraper'].get('results')
        
        if results:
            st.json(results)
        else:
            st.info("No social media data available. Run scraping to see results here.")
    
    def _show_technical_results(self):
        """Show technical analysis results"""
        results = st.session_state.analysis_services['technical_analysis'].get('results')
        
        if results:
            st.json(results)
        else:
            st.info("No technical analysis results available. Run analysis to see results here.")
    
    def _show_sentiment_results(self):
        """Show sentiment analysis results"""
        results = st.session_state.analysis_services['sentiment_analysis'].get('results')
        
        if results:
            st.json(results)
        else:
            st.info("No sentiment analysis results available. Run analysis to see results here.")
    
    def _start_all_background_services(self):
        """Start all background services"""
        try:
            # Start Windows batch files if on Windows
            if os.name == 'nt':
                scripts_dir = Path("scripts")
                
                # Start ML analysis service
                subprocess.Popen([str(scripts_dir / "start_ml_analysis.bat")], shell=True)
                
                # Start social scraper service
                subprocess.Popen([str(scripts_dir / "start_social_scraper.bat")], shell=True)
                
                st.success("Background services started! Check the console windows.")
            else:
                st.warning("Background service scripts are designed for Windows. Use the Python scripts directly on other platforms.")
        
        except Exception as e:
            st.error(f"Failed to start background services: {e}")
    
    def _stop_all_background_services(self):
        """Stop all background services"""
        try:
            if os.name == 'nt':
                # Kill processes by window title (Windows specific)
                subprocess.run(['taskkill', '/FI', 'WINDOWTITLE eq ML Analysis*', '/F'], shell=True)
                subprocess.run(['taskkill', '/FI', 'WINDOWTITLE eq Social Scraper*', '/F'], shell=True)
                
                st.success("Background services stopped")
            else:
                st.info("Please stop Python services manually on non-Windows systems")
        
        except Exception as e:
            st.error(f"Failed to stop services: {e}")
    
    def _show_system_status(self):
        """Show system status information"""
        try:
            health_status = self.health_monitor.get_system_health()
            st.json(health_status)
        except Exception as e:
            st.error(f"Failed to get system status: {e}")
    
    def _open_data_folder(self):
        """Open data folder in file explorer"""
        try:
            data_path = Path("data").resolve()
            if os.name == 'nt':
                os.startfile(data_path)
            else:
                subprocess.run(['open', data_path] if sys.platform == 'darwin' else ['xdg-open', data_path])
            
            st.success("Data folder opened in file explorer")
        except Exception as e:
            st.error(f"Failed to open data folder: {e}")
    
    def _show_logs(self):
        """Show recent log entries"""
        try:
            logs_dir = Path("logs")
            
            if logs_dir.exists():
                log_files = list(logs_dir.glob("*.log"))
                
                if log_files:
                    selected_log = st.selectbox("Select log file:", [f.name for f in log_files])
                    
                    if selected_log:
                        log_path = logs_dir / selected_log
                        
                        # Read last 50 lines
                        with open(log_path, 'r') as f:
                            lines = f.readlines()
                            recent_lines = lines[-50:] if len(lines) > 50 else lines
                        
                        st.text_area("Recent log entries:", value=''.join(recent_lines), height=300)
                else:
                    st.info("No log files found")
            else:
                st.info("Logs directory not found")
        
        except Exception as e:
            st.error(f"Failed to show logs: {e}")
    
    def _refresh_service_status(self):
        """Refresh service status"""
        st.rerun()