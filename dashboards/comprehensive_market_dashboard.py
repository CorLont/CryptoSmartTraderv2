"""
CryptoSmartTrader V2 - Comprehensive Market Dashboard
Complete market coverage dashboard with dynamic cryptocurrency discovery and opportunities
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import time


class ComprehensiveMarketDashboard:
    """Dashboard for comprehensive market analysis and opportunity discovery"""

    def __init__(self, container):
        self.container = container
        self.market_scanner = container.market_scanner()
        self.cache_manager = container.cache_manager()

    def render(self):
        """Render the comprehensive market dashboard"""
        st.title("🌍 Comprehensive Market Scanner")
        st.markdown(
            "Complete cryptocurrency market coverage with dynamic discovery and multi-timeframe analysis"
        )

        # Auto-refresh controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**Real-time market scanning across all available cryptocurrencies**")
        with col2:
            auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
        with col3:
            if st.button("🔄 Force Full Scan"):
                with st.spinner("Running comprehensive market scan..."):
                    self.market_scanner.force_full_scan()
                st.success("Market scan completed!")
                st.rerun()

        if auto_refresh:
            time.sleep(30)
            st.rerun()

        # Market overview
        self._render_market_overview()

        # Trading opportunities
        self._render_trading_opportunities()

        # Discovered coins
        self._render_discovered_coins()

        # Multi-timeframe analysis
        self._render_timeframe_analysis()

        # Scanner controls
        self._render_scanner_controls()

    def _render_market_overview(self):
        """Render market overview section"""
        st.header("📊 Market Overview")

        # Get scanning status
        scan_status = self.market_scanner.get_scan_status()

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            status_indicator = "🟢 Active" if scan_status["scanning_active"] else "🔴 Stopped"
            st.metric("Scanner Status", status_indicator)

        with col2:
            total_coins = scan_status["discovered_coins_count"]
            st.metric("Total Coins", f"{total_coins:,}")

        with col3:
            active_coins = scan_status["active_coins_count"]
            st.metric("Active Pairs", f"{active_coins:,}")

        with col4:
            opportunities = scan_status["statistics"]["opportunities_found"]
            st.metric("Opportunities", f"{opportunities:,}")

        with col5:
            timeframes = len(scan_status["timeframes"])
            st.metric("Timeframes", timeframes)

        # Scanner statistics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Scanning Statistics")
            stats = scan_status["statistics"]

            if stats.get("last_full_scan"):
                last_scan = datetime.fromisoformat(stats["last_full_scan"])
                time_ago = datetime.now() - last_scan
                st.write(f"**Last Full Scan:** {time_ago.seconds // 60} minutes ago")

            if stats.get("scan_duration_seconds"):
                st.write(f"**Scan Duration:** {stats['scan_duration_seconds']:.1f} seconds")

            st.write(f"**Analysis Threads:** {scan_status['configuration']['analysis_threads']}")
            st.write(f"**Batch Size:** {scan_status['configuration']['batch_size']}")

        with col2:
            st.subheader("🔗 Exchange Coverage")
            exchanges = scan_status["exchanges"]

            exchange_info = {
                "kraken": "🇺🇸 Kraken - Primary",
                "binance": "🌍 Binance - Global",
                "coinbasepro": "🇺🇸 Coinbase Pro - US",
            }

            for exchange in exchanges:
                status_emoji = "🟢" if exchange in exchanges else "🔴"
                description = exchange_info.get(exchange, exchange.title())
                st.write(f"{status_emoji} {description}")

    def _render_trading_opportunities(self):
        """Render trading opportunities section"""
        st.header("🎯 Trading Opportunities")

        # Get opportunities
        opportunities = self.market_scanner.get_trading_opportunities(min_score=2)

        if not opportunities:
            st.info(
                "No trading opportunities found. The scanner continuously monitors all markets for potential opportunities."
            )
            return

        # Opportunity filters
        col1, col2, col3 = st.columns(3)

        with col1:
            min_score = st.selectbox("Minimum Score", options=[1, 2, 3, 4, 5], index=1)

        with col2:
            timeframe_filter = st.selectbox(
                "Timeframe", options=["All"] + list(self.market_scanner.timeframes.keys()), index=0
            )

        with col3:
            max_opportunities = st.selectbox("Show Top", options=[10, 25, 50, 100], index=1)

        # Filter opportunities
        filtered_opportunities = [opp for opp in opportunities if opp.get("score", 0) >= min_score]

        if timeframe_filter != "All":
            filtered_opportunities = [
                opp for opp in filtered_opportunities if opp.get("timeframe") == timeframe_filter
            ]

        filtered_opportunities = filtered_opportunities[:max_opportunities]

        if filtered_opportunities:
            # Display opportunities
            st.subheader(f"🔍 Found {len(filtered_opportunities)} Opportunities")

            for i, opp in enumerate(filtered_opportunities, 1):
                with st.expander(
                    f"#{i} {opp['symbol']} ({opp['timeframe']}) - Score: {opp['score']}"
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Trading Signals:**")
                        for signal in opp.get("signals", []):
                            st.write(f"• {signal}")

                        analysis = opp.get("analysis", {})
                        if analysis:
                            st.write(f"**Price:** ${analysis.get('last_price', 0):.4f}")
                            st.write(
                                f"**24h Change:** {analysis.get('price_change_24h_pct', 0):+.2f}%"
                            )
                            st.write(f"**RSI:** {analysis.get('rsi', 0):.1f}")

                    with col2:
                        st.write("**Technical Analysis:**")
                        if analysis:
                            trend = analysis.get("trend_direction", "Unknown")
                            trend_emoji = {"bullish": "📈", "bearish": "📉", "sideways": "➡️"}.get(
                                trend, "❓"
                            )
                            st.write(f"**Trend:** {trend_emoji} {trend.title()}")

                            if "volume_ratio" in analysis:
                                st.write(f"**Volume Ratio:** {analysis['volume_ratio']:.1f}x")

                            if "volatility" in analysis:
                                st.write(f"**Volatility:** {analysis['volatility']:.2f}%")

                        detected_time = datetime.fromisoformat(opp["detected_at"])
                        st.write(f"**Detected:** {detected_time.strftime('%H:%M:%S')}")

                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"📊 Detailed Analysis", key=f"detail_{i}"):
                            self._show_detailed_analysis(opp["symbol"])
                    with col2:
                        if st.button(f"⭐ Add to Watchlist", key=f"watch_{i}"):
                            st.success(f"Added {opp['symbol']} to watchlist")
                    with col3:
                        if st.button(f"🔔 Set Alert", key=f"alert_{i}"):
                            st.success(f"Alert set for {opp['symbol']}")
        else:
            st.warning(f"No opportunities found with score >= {min_score}")

    def _render_discovered_coins(self):
        """Render discovered coins section"""
        st.header("🪙 Discovered Cryptocurrencies")

        discovered_data = self.market_scanner.get_all_discovered_coins()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Discovery Statistics")

            total_coins = discovered_data["total_coins"]
            active_coins = len(
                [
                    coin
                    for coin, metadata in discovered_data["metadata"].items()
                    if metadata.get("active", False)
                ]
            )

            st.write(f"**Total Discovered:** {total_coins:,}")
            st.write(f"**Active Trading Pairs:** {active_coins:,}")
            st.write(f"**Coverage Percentage:** {(active_coins / total_coins * 100):.1f}%")

            # Exchange distribution
            exchange_counts = {}
            for metadata in discovered_data["metadata"].values():
                exchange = metadata.get("exchange", "unknown")
                exchange_counts[exchange] = exchange_counts.get(exchange, 0) + 1

            if exchange_counts:
                st.write("**Exchange Distribution:**")
                for exchange, count in exchange_counts.items():
                    st.write(f"• {exchange.title()}: {count:,} pairs")

        with col2:
            st.subheader("🔍 Search & Filter")

            search_term = st.text_input(
                "Search cryptocurrencies", placeholder="e.g., BTC, ETH, ADA"
            )

            show_only_active = st.checkbox("Show only active pairs", value=True)

            # Filter coins
            filtered_coins = []
            for symbol, metadata in discovered_data["metadata"].items():
                if show_only_active and not metadata.get("active", False):
                    continue

                if search_term:
                    if search_term.upper() not in symbol.upper():
                        continue

                filtered_coins.append((symbol, metadata))

            # Sort by symbol
            filtered_coins.sort(key=lambda x: x[0])

            if filtered_coins:
                st.write(f"**Showing {len(filtered_coins)} coins:**")

                # Display in paginated format
                items_per_page = 20
                total_pages = (len(filtered_coins) + items_per_page - 1) // items_per_page

                if total_pages > 1:
                    page = (
                        st.selectbox("Page", options=list(range(1, total_pages + 1)), index=0) - 1
                    )
                    start_idx = page * items_per_page
                    end_idx = min(start_idx + items_per_page, len(filtered_coins))
                    page_coins = filtered_coins[start_idx:end_idx]
                else:
                    page_coins = filtered_coins

                for symbol, metadata in page_coins:
                    base = metadata.get("base", "")
                    quote = metadata.get("quote", "")
                    exchange = metadata.get("exchange", "")

                    col_symbol, col_exchange, col_action = st.columns([2, 1, 1])

                    with col_symbol:
                        st.write(f"**{symbol}** ({base}/{quote})")

                    with col_exchange:
                        st.write(exchange.title())

                    with col_action:
                        if st.button("📈", key=f"analyze_{symbol}", help="Analyze"):
                            self._show_detailed_analysis(symbol)

    def _render_timeframe_analysis(self):
        """Render multi-timeframe analysis section"""
        st.header("⏰ Multi-Timeframe Analysis")

        # Timeframe selector
        col1, col2 = st.columns(2)

        with col1:
            selected_symbol = st.text_input(
                "Enter cryptocurrency symbol",
                placeholder="e.g., BTC/USD, ETH/USDT",
                help="Use exact symbol format from discovered coins",
            )

        with col2:
            timeframes = list(self.market_scanner.timeframes.keys())
            selected_timeframes = st.multiselect(
                "Select timeframes", options=timeframes, default=timeframes[:4]
            )

        if selected_symbol and selected_timeframes:
            if st.button("🔍 Analyze Across Timeframes"):
                self._show_multi_timeframe_analysis(selected_symbol, selected_timeframes)

    def _render_scanner_controls(self):
        """Render scanner control section"""
        st.header("⚙️ Scanner Controls")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎛️ Configuration")

            scan_status = self.market_scanner.get_scan_status()

            if scan_status["scanning_active"]:
                if st.button("⏹️ Stop Scanner"):
                    self.market_scanner.stop_comprehensive_scanning()
                    st.success("Scanner stopped")
                    st.rerun()
            else:
                if st.button("▶️ Start Scanner"):
                    self.market_scanner.start_comprehensive_scanning()
                    st.success("Scanner started")
                    st.rerun()

            if st.button("🔄 Discover New Coins"):
                with st.spinner("Discovering new cryptocurrencies..."):
                    self.market_scanner._discover_all_cryptocurrencies()
                st.success("Coin discovery completed")
                st.rerun()

        with col2:
            st.subheader("📊 Performance Metrics")

            config = scan_status["configuration"]
            st.write(f"**Discovery Interval:** {config['discovery_interval']} seconds")
            st.write(f"**Analysis Threads:** {config['analysis_threads']}")
            st.write(f"**Batch Size:** {config['batch_size']}")
            st.write(f"**Min Volume:** ${config['min_volume_usd']:,}")

            # Real-time performance
            stats = scan_status["statistics"]
            if stats.get("scan_duration_seconds"):
                coins_per_second = stats["active_coins_analyzed"] / max(
                    1, stats["scan_duration_seconds"]
                )
                st.write(f"**Analysis Speed:** {coins_per_second:.1f} coins/second")

    def _show_detailed_analysis(self, symbol: str):
        """Show detailed analysis for a specific cryptocurrency"""
        st.subheader(f"📊 Detailed Analysis: {symbol}")

        comprehensive_analysis = self.market_scanner.get_comprehensive_analysis(symbol)

        if not comprehensive_analysis.get("timeframe_analysis"):
            st.warning(f"No analysis data available for {symbol}. Try running a full scan first.")
            return

        # Metadata
        metadata = comprehensive_analysis.get("metadata", {})
        if metadata:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Exchange:** {metadata.get('exchange', 'Unknown')}")
            with col2:
                st.write(f"**Type:** {metadata.get('type', 'spot')}")
            with col3:
                st.write(f"**Active:** {'✅' if metadata.get('active') else '❌'}")

        # Timeframe analysis
        timeframe_data = comprehensive_analysis["timeframe_analysis"]

        # Create comparison table
        comparison_data = []
        for timeframe, analysis in timeframe_data.items():
            comparison_data.append(
                {
                    "Timeframe": timeframe,
                    "Price": f"${analysis.get('last_price', 0):.4f}",
                    "Change %": f"{analysis.get('price_change_pct', 0):+.2f}%",
                    "RSI": f"{analysis.get('rsi', 0):.1f}",
                    "Trend": analysis.get("trend_direction", "Unknown"),
                    "Volume Ratio": f"{analysis.get('volume_ratio', 0):.1f}x",
                }
            )

        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)

        # Aggregated signals
        if comprehensive_analysis.get("aggregated_signals"):
            st.subheader("🎯 Aggregated Signals")

            signals = comprehensive_analysis["aggregated_signals"]

            col1, col2, col3 = st.columns(3)
            with col1:
                trend = signals.get("overall_trend", "neutral")
                trend_emoji = {"bullish": "📈", "bearish": "📉", "sideways": "➡️"}.get(trend, "❓")
                st.metric("Overall Trend", f"{trend_emoji} {trend.title()}")

            with col2:
                strength = signals.get("strength_score", 0)
                st.metric("Consensus Strength", f"{strength:.1%}")

            with col3:
                risk = signals.get("risk_level", "medium")
                risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "🟡")
                st.metric("Risk Level", f"{risk_emoji} {risk.title()}")

        # Related opportunities
        opportunities = comprehensive_analysis.get("opportunities", [])
        if opportunities:
            st.subheader("🎯 Related Opportunities")
            for opp in opportunities:
                st.write(
                    f"• **{opp['timeframe']}:** Score {opp['score']} - {', '.join(opp['signals'])}"
                )

    def _show_multi_timeframe_analysis(self, symbol: str, timeframes: List[str]):
        """Show multi-timeframe analysis visualization"""
        st.subheader(f"📊 Multi-Timeframe Analysis: {symbol}")

        # Collect data from cache
        analysis_data = {}
        for timeframe in timeframes:
            cache_key = f"analysis_{symbol}_{timeframe}"
            if self.cache_manager:
                data = self.cache_manager.get(cache_key)
                if data:
                    analysis_data[timeframe] = data

        if not analysis_data:
            st.warning(
                f"No analysis data found for {symbol}. The symbol may not be available or analyzed yet."
            )
            return

        # Create visualization
        metrics_to_plot = ["rsi", "price_change_pct", "volume_ratio", "volatility"]

        for metric in metrics_to_plot:
            values = []
            timeframe_labels = []

            for tf in timeframes:
                if tf in analysis_data and metric in analysis_data[tf]:
                    values.append(analysis_data[tf][metric])
                    timeframe_labels.append(tf)

            if values:
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=timeframe_labels,
                        y=values,
                        name=metric.upper(),
                        text=[f"{v:.2f}" for v in values],
                        textposition="auto",
                    )
                )

                fig.update_layout(
                    title=f"{metric.upper()} Across Timeframes",
                    xaxis_title="Timeframe",
                    yaxis_title=metric.upper(),
                    height=400,
                )

                st.plotly_chart(fig, use_container_width=True)
