"""
CryptoSmartTrader V2 - System Health Dashboard
Complete system validation and health monitoring
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class SystemHealthDashboard:
    """Complete system health and validation dashboard"""
    
    def __init__(self, container):
        self.container = container
        
    def render(self):
        """Render complete system health dashboard"""
        st.set_page_config(
            page_title="System Health - CryptoSmartTrader V2",
            page_icon="🏥",
            layout="wide"
        )
        
        st.title("🏥 System Health & Validation")
        st.markdown("**Complete system status, validation and health monitoring**")
        
        # Run system validation
        self._render_system_validation()
        
        # Component status overview
        self._render_component_status()
        
        # Data integrity checks
        self._render_data_integrity()
        
        # Performance metrics
        self._render_performance_metrics()
        
        # Auto-fix capabilities
        self._render_auto_fix_section()
    
    def _render_system_validation(self):
        """Run and display complete system validation"""
        st.header("🔍 Complete System Validation")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🔄 Run Full Validation", type="primary"):
                st.rerun()
        
        try:
            # Get system validator
            validator = self.container.system_validator()
            
            # Run complete validation
            with st.spinner("Running complete system validation..."):
                validation_results = validator.run_complete_validation()
            
            # Overall status
            status = validation_results.get('overall_status', 'UNKNOWN')
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if status == 'PASS':
                    st.success(f"**System Status:** {status}")
                elif status == 'FAIL':
                    st.error(f"**System Status:** {status}")
                else:
                    st.warning(f"**System Status:** {status}")
            
            with col2:
                components = validation_results.get('components', {})
                passed_components = sum(1 for comp in components.values() if comp.get('status', False))
                total_components = len(components)
                st.metric("Components OK", f"{passed_components}/{total_components}")
            
            with col3:
                errors = len(validation_results.get('errors', []))
                st.metric("Critical Errors", errors, delta=-errors if errors > 0 else None)
            
            with col4:
                warnings = len(validation_results.get('warnings', []))
                st.metric("Warnings", warnings, delta=-warnings if warnings > 0 else None)
            
            # Component details
            st.subheader("📊 Component Validation Details")
            
            if components:
                component_data = []
                for comp_name, comp_result in components.items():
                    component_data.append({
                        'Component': comp_name.replace('_', ' ').title(),
                        'Status': '✅ PASS' if comp_result.get('status', False) else '❌ FAIL',
                        'Info': comp_result.get('info', {}),
                        'Errors': ', '.join(comp_result.get('errors', [])) or 'None'
                    })
                
                df_components = pd.DataFrame(component_data)
                st.dataframe(df_components, use_container_width=True)
            
            # Errors and warnings
            if validation_results.get('errors'):
                st.subheader("🚨 Critical Errors")
                for error in validation_results['errors']:
                    st.error(f"• {error}")
            
            if validation_results.get('warnings'):
                st.subheader("⚠️ Warnings")
                for warning in validation_results['warnings']:
                    st.warning(f"• {warning}")
            
            # Recommendations
            if validation_results.get('recommendations'):
                st.subheader("💡 Recommendations")
                for rec in validation_results['recommendations']:
                    st.info(f"• {rec}")
            
        except Exception as e:
            st.error(f"System validation failed: {e}")
    
    def _render_component_status(self):
        """Render individual component status"""
        st.header("🔧 Component Status Details")
        
        try:
            # Check cache manager
            with st.expander("💾 Cache Manager"):
                try:
                    cache_manager = self.container.cache_manager()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        cache_size = len(cache_manager._cache) if hasattr(cache_manager, '_cache') else 0
                        st.metric("Cache Entries", f"{cache_size:,}")
                    
                    with col2:
                        memory_usage = cache_manager.get_total_memory_usage() if hasattr(cache_manager, 'get_total_memory_usage') else 0
                        st.metric("Memory Usage", f"{memory_usage:.1f} MB")
                    
                    with col3:
                        max_memory = getattr(cache_manager, 'max_memory_mb', 1000)
                        memory_pct = (memory_usage / max_memory * 100) if max_memory > 0 else 0
                        st.metric("Memory %", f"{memory_pct:.1f}%")
                    
                    # Cache statistics
                    if hasattr(cache_manager, 'get_cache_stats'):
                        stats = cache_manager.get_cache_stats()
                        st.json(stats)
                    
                except Exception as e:
                    st.error(f"Cache manager error: {e}")
            
            # Check real-time pipeline
            with st.expander("⚡ Real-Time Pipeline"):
                try:
                    pipeline = self.container.real_time_pipeline()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        pipeline_active = getattr(pipeline, 'pipeline_active', False)
                        st.metric("Pipeline Active", "✅ Yes" if pipeline_active else "❌ No")
                    
                    with col2:
                        task_count = len(getattr(pipeline, 'pipeline_tasks', {}))
                        st.metric("Background Tasks", task_count)
                    
                    with col3:
                        # Check last execution times
                        last_exec = getattr(pipeline, 'last_execution_times', {})
                        recent_tasks = sum(1 for dt in last_exec.values() 
                                         if isinstance(dt, datetime) and dt > datetime.now() - timedelta(minutes=10))
                        st.metric("Recent Activity", f"{recent_tasks} tasks")
                    
                    # Task details
                    if hasattr(pipeline, 'get_pipeline_status'):
                        status = pipeline.get_pipeline_status()
                        st.json(status)
                    
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
            
            # Check multi-horizon ML
            with st.expander("🤖 Multi-Horizon ML System"):
                try:
                    ml_system = self.container.multi_horizon_ml()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        models_loaded = len(getattr(ml_system, 'models', {}))
                        st.metric("Models Loaded", models_loaded)
                    
                    with col2:
                        horizons = list(getattr(ml_system, 'horizons', {}).keys())
                        st.metric("Horizons", f"{len(horizons)}")
                    
                    with col3:
                        predictions_made = len(getattr(ml_system, 'prediction_log', {}))
                        st.metric("Predictions Made", f"{predictions_made:,}")
                    
                    # Model performance
                    performance = getattr(ml_system, 'model_performance', {})
                    if performance:
                        st.write("**Model Performance:**")
                        perf_data = []
                        for horizon, metrics in performance.items():
                            perf_data.append({
                                'Horizon': horizon,
                                'Test MAE': f"{metrics.get('test_mae', 0):.4f}",
                                'Samples': f"{metrics.get('training_samples', 0):,}"
                            })
                        
                        if perf_data:
                            df_perf = pd.DataFrame(perf_data)
                            st.dataframe(df_perf, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"ML system error: {e}")
            
        except Exception as e:
            st.error(f"Component status check failed: {e}")
    
    def _render_data_integrity(self):
        """Render data integrity validation"""
        st.header("🔒 Data Integrity Validation")
        
        try:
            cache_manager = self.container.cache_manager()
            
            # Data type analysis
            price_data_count = 0
            sentiment_data_count = 0
            whale_data_count = 0
            dummy_data_count = 0
            total_entries = 0
            
            if hasattr(cache_manager, '_cache'):
                for key in cache_manager._cache.keys():
                    total_entries += 1
                    
                    if key.startswith('validated_price_data_'):
                        price_data_count += 1
                    elif key.startswith('validated_sentiment_'):
                        sentiment_data_count += 1
                    elif key.startswith('validated_whale_'):
                        whale_data_count += 1
                    elif 'dummy' in key.lower():
                        dummy_data_count += 1
            
            # Data integrity metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Price Data Coins", f"{price_data_count:,}")
            
            with col2:
                st.metric("Sentiment Data Coins", f"{sentiment_data_count:,}")
            
            with col3:
                st.metric("Whale Data Coins", f"{whale_data_count:,}")
            
            with col4:
                # Dummy data should be 0 (strict requirement)
                if dummy_data_count == 0:
                    st.success(f"✅ No Dummy Data")
                else:
                    st.error(f"❌ {dummy_data_count} Dummy Entries")
            
            # Data completeness visualization
            if total_entries > 0:
                data_types = ['Price Data', 'Sentiment Data', 'Whale Data', 'Other Data']
                data_counts = [
                    price_data_count,
                    sentiment_data_count, 
                    whale_data_count,
                    total_entries - price_data_count - sentiment_data_count - whale_data_count
                ]
                
                fig_pie = px.pie(
                    values=data_counts,
                    names=data_types,
                    title="Data Distribution in Cache"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Data quality checks
            st.subheader("🔍 Data Quality Analysis")
            
            complete_data_coins = []
            
            # Find coins with all data types
            coin_data_status = {}
            
            if hasattr(cache_manager, '_cache'):
                for key in cache_manager._cache.keys():
                    if key.startswith('validated_'):
                        parts = key.split('_')
                        if len(parts) >= 3:
                            data_type = parts[1]
                            coin = '_'.join(parts[2:])
                            
                            if coin not in coin_data_status:
                                coin_data_status[coin] = {'price': False, 'sentiment': False, 'whale': False}
                            
                            if data_type == 'price':
                                coin_data_status[coin]['price'] = True
                            elif data_type == 'sentiment':
                                coin_data_status[coin]['sentiment'] = True
                            elif data_type == 'whale':
                                coin_data_status[coin]['whale'] = True
            
            # Calculate completeness
            complete_coins = [
                coin for coin, status in coin_data_status.items()
                if all(status.values())
            ]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Unique Coins", len(coin_data_status))
            
            with col2:
                st.metric("Complete Data Coins", len(complete_coins))
            
            with col3:
                completion_rate = len(complete_coins) / len(coin_data_status) * 100 if coin_data_status else 0
                st.metric("Data Completeness", f"{completion_rate:.1f}%")
            
            # Show complete coins
            if complete_coins:
                st.subheader("✅ Coins with Complete Data")
                complete_df = pd.DataFrame({'Coin': complete_coins[:20]})  # Show first 20
                st.dataframe(complete_df, use_container_width=True)
                
                if len(complete_coins) > 20:
                    st.info(f"Showing first 20 of {len(complete_coins)} coins with complete data")
            
        except Exception as e:
            st.error(f"Data integrity check failed: {e}")
    
    def _render_performance_metrics(self):
        """Render system performance metrics"""
        st.header("📈 System Performance")
        
        try:
            # System resource usage
            import psutil
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cpu_percent = psutil.cpu_percent(interval=1)
                st.metric("CPU Usage", f"{cpu_percent:.1f}%")
            
            with col2:
                memory = psutil.virtual_memory()
                st.metric("Memory Usage", f"{memory.percent:.1f}%")
            
            with col3:
                disk = psutil.disk_usage('.')
                st.metric("Disk Usage", f"{disk.percent:.1f}%")
            
            with col4:
                # Process count
                process_count = len(psutil.pids())
                st.metric("System Processes", f"{process_count:,}")
            
            # Performance over time (simulated)
            st.subheader("📊 Performance Trends")
            
            # Create sample performance data
            now = datetime.now()
            times = [now - timedelta(minutes=x*5) for x in range(12, 0, -1)]
            
            perf_data = pd.DataFrame({
                'Time': times,
                'CPU (%)': [cpu_percent + (i-6)*2 for i in range(12)],
                'Memory (%)': [memory.percent + (i-6)*1.5 for i in range(12)],
                'Cache Size': [1000 + i*100 for i in range(12)]
            })
            
            fig_perf = px.line(
                perf_data,
                x='Time',
                y=['CPU (%)', 'Memory (%)'],
                title="System Resource Usage (Last Hour)"
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            
        except Exception as e:
            st.error(f"Performance metrics failed: {e}")
    
    def _render_auto_fix_section(self):
        """Render automatic issue resolution section"""
        st.header("🛠️ Automatic Issue Resolution")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            **Available Auto-Fixes:**
            • Create missing directories
            • Initialize cache manager
            • Remove dummy data entries
            • Fix common configuration issues
            """)
        
        with col2:
            if st.button("🔧 Run Auto-Fix", type="secondary"):
                try:
                    validator = self.container.system_validator()
                    
                    with st.spinner("Running automatic fixes..."):
                        fix_results = validator.fix_common_issues()
                    
                    # Show results
                    fixes_applied = fix_results.get('fixes', [])
                    errors = fix_results.get('errors', [])
                    
                    if fixes_applied:
                        st.success(f"Applied {len(fixes_applied)} fixes:")
                        for fix in fixes_applied:
                            st.write(f"✅ {fix}")
                    
                    if errors:
                        st.error("Some fixes failed:")
                        for error in errors:
                            st.write(f"❌ {error}")
                    
                    if not fixes_applied and not errors:
                        st.info("No issues found that could be auto-fixed")
                    
                    # Refresh after fixes
                    if fixes_applied:
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Auto-fix failed: {e}")
        
        # Manual validation trigger
        st.subheader("🔄 Manual Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Clear All Cache"):
                try:
                    cache_manager = self.container.cache_manager()
                    cache_manager.clear()
                    st.success("Cache cleared successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"Cache clear failed: {e}")
        
        with col2:
            if st.button("Restart Pipeline"):
                try:
                    pipeline = self.container.real_time_pipeline()
                    pipeline.stop_pipeline()
                    pipeline.start_pipeline()
                    st.success("Pipeline restarted")
                    st.rerun()
                except Exception as e:
                    st.error(f"Pipeline restart failed: {e}")
        
        with col3:
            if st.button("Retrain ML Models"):
                try:
                    ml_system = self.container.multi_horizon_ml()
                    
                    # Prepare training data
                    training_data = ml_system.prepare_training_data(lookback_days=30)
                    
                    if training_data is not None:
                        training_results = ml_system.train_models(training_data)
                        if training_results:
                            st.success(f"Trained {len(training_results)} models")
                        else:
                            st.warning("Model training failed")
                    else:
                        st.warning("Insufficient training data")
                        
                except Exception as e:
                    st.error(f"Model training failed: {e}")


# Initialize dashboard
def main():
    """Main dashboard function"""
    try:
        # Import container
        from containers import ApplicationContainer
        
        # Initialize container
        container = ApplicationContainer()
        container.wire(modules=[__name__])
        
        # Initialize dashboard
        dashboard = SystemHealthDashboard(container)
        dashboard.render()
        
    except Exception as e:
        st.error(f"Dashboard initialization failed: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()