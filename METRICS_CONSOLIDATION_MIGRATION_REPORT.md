# Metrics Consolidation Migration Report

**Generated:** 2025-08-14 15:24:14

## Migration Summary

- **Files Scanned:** 46209
- **Metrics Found:** 368
- **Alerts Found:** 56
- **Metrics Migrated:** 0
- **Alerts Migrated:** 0
- **Compatibility Aliases:** 368
- **Migration Time:** 69.1s
- **Backup Created:** ✅

## Discovered Metrics

| Name | Type | File | Line | Labels |
|------|------|------|------|--------|
| `orders_sent_total` | counter | centralize_observability.py | 107 | None |
| `orders_filled_total` | counter | centralize_observability.py | 114 | None |
| `order_errors_total` | counter | centralize_observability.py | 121 | None |
| `signals_received_total` | counter | centralize_observability.py | 160 | None |
| `signals_processed_total` | counter | centralize_observability.py | 167 | None |
| `api_calls_total` | counter | centralize_observability.py | 182 | None |
| `cache_hits_total` | counter | centralize_observability.py | 189 | None |
| `risk_violations_total` | counter | centralize_observability.py | 204 | None |
| `data_points_received_total` | counter | centralize_observability.py | 219 | None |
| `equity_usd` | gauge | centralize_observability.py | 145 | None |
| `drawdown_pct` | gauge | centralize_observability.py | 152 | None |
| `signal_accuracy_pct` | gauge | centralize_observability.py | 174 | None |
| `memory_usage_bytes` | gauge | centralize_observability.py | 196 | None |
| `position_size_usd` | gauge | centralize_observability.py | 211 | None |
| `data_quality_score` | gauge | centralize_observability.py | 226 | None |
| `latency_ms` | histogram | centralize_observability.py | 129 | None |
| `slippage_bps` | histogram | centralize_observability.py | 137 | None |
| `cc` | counter | .cache/uv/archive-v0/wKOcRBJ3b5R8ncmzUkUqL/prometheus_client/__init__.py | 56 | None |
| `gg` | gauge | .cache/uv/archive-v0/wKOcRBJ3b5R8ncmzUkUqL/prometheus_client/__init__.py | 59 | None |
| `hh` | histogram | .cache/uv/archive-v0/wKOcRBJ3b5R8ncmzUkUqL/prometheus_client/__init__.py | 65 | None |
| `ss` | summary | .cache/uv/archive-v0/wKOcRBJ3b5R8ncmzUkUqL/prometheus_client/__init__.py | 62 | None |
| `my_requests_total` | counter | .cache/uv/archive-v0/wKOcRBJ3b5R8ncmzUkUqL/prometheus_client/metrics.py | 142 | None |
| `my_requests_total` | counter | .cache/uv/archive-v0/wKOcRBJ3b5R8ncmzUkUqL/prometheus_client/metrics.py | 150 | None |
| `my_failures_total` | counter | .cache/uv/archive-v0/wKOcRBJ3b5R8ncmzUkUqL/prometheus_client/metrics.py | 255 | None |
| `my_inprogress_requests` | gauge | .cache/uv/archive-v0/wKOcRBJ3b5R8ncmzUkUqL/prometheus_client/metrics.py | 333 | None |
| `data_objects` | gauge | .cache/uv/archive-v0/wKOcRBJ3b5R8ncmzUkUqL/prometheus_client/metrics.py | 352 | None |
| `request_size_bytes` | histogram | .cache/uv/archive-v0/wKOcRBJ3b5R8ncmzUkUqL/prometheus_client/metrics.py | 537 | None |
| `response_latency_seconds` | histogram | .cache/uv/archive-v0/wKOcRBJ3b5R8ncmzUkUqL/prometheus_client/metrics.py | 544 | None |
| `request_size_bytes` | summary | .cache/uv/archive-v0/wKOcRBJ3b5R8ncmzUkUqL/prometheus_client/metrics.py | 465 | None |
| `response_latency_seconds` | summary | .cache/uv/archive-v0/wKOcRBJ3b5R8ncmzUkUqL/prometheus_client/metrics.py | 472 | None |
| `crypto_requests_total` | counter | utils/metrics.py | 13 | None |
| `cache_hits_total` | counter | utils/metrics.py | 23 | None |
| `cache_misses_total` | counter | utils/metrics.py | 24 | None |
| `errors_total` | counter | utils/metrics.py | 25 | None |
| `system_health_score` | gauge | utils/metrics.py | 16 | None |
| `agent_performance_score` | gauge | utils/metrics.py | 17 | None |
| `data_freshness_seconds` | gauge | utils/metrics.py | 18 | None |
| `prediction_accuracy` | gauge | utils/metrics.py | 19 | None |
| `exchange_latency_seconds` | histogram | utils/metrics.py | 22 | None |
| `cryptotrader_requests_total` | counter | exports/unified_technical_review/source_code/metrics/metrics_server.py | 20 | None |
| `cryptotrader_api_calls_total` | counter | exports/unified_technical_review/source_code/metrics/metrics_server.py | 58 | None |
| `cryptotrader_active_trades` | gauge | exports/unified_technical_review/source_code/metrics/metrics_server.py | 34 | None |
| `cryptotrader_portfolio_value_usd` | gauge | exports/unified_technical_review/source_code/metrics/metrics_server.py | 40 | None |
| `cryptotrader_confidence_score` | gauge | exports/unified_technical_review/source_code/metrics/metrics_server.py | 46 | None |
| `cryptotrader_system_health_score` | gauge | exports/unified_technical_review/source_code/metrics/metrics_server.py | 52 | None |
| `cryptotrader_prediction_accuracy` | gauge | exports/unified_technical_review/source_code/metrics/metrics_server.py | 65 | None |
| `cryptotrader_request_duration_seconds` | histogram | exports/unified_technical_review/source_code/metrics/metrics_server.py | 27 | None |
| `cryptotrader_operations_total` | counter | exports/unified_technical_review/source_code/src/cryptosmarttrader/core/improved_logging_manager.py | 80 | None |
| `cryptotrader_active_agents` | gauge | exports/unified_technical_review/source_code/src/cryptosmarttrader/core/improved_logging_manager.py | 95 | None |
| `cryptotrader_prediction_accuracy` | gauge | exports/unified_technical_review/source_code/src/cryptosmarttrader/core/improved_logging_manager.py | 102 | None |
| `cryptotrader_operation_duration_seconds` | histogram | exports/unified_technical_review/source_code/src/cryptosmarttrader/core/improved_logging_manager.py | 87 | None |
| `crypto_requests_total` | counter | exports/unified_technical_review/source_code/src/cryptosmarttrader/utils/metrics.py | 13 | None |
| `cache_hits_total` | counter | exports/unified_technical_review/source_code/src/cryptosmarttrader/utils/metrics.py | 19 | None |
| `cache_misses_total` | counter | exports/unified_technical_review/source_code/src/cryptosmarttrader/utils/metrics.py | 20 | None |
| `errors_total` | counter | exports/unified_technical_review/source_code/src/cryptosmarttrader/utils/metrics.py | 21 | None |
| `system_health_score` | gauge | exports/unified_technical_review/source_code/src/cryptosmarttrader/utils/metrics.py | 14 | None |
| `agent_performance_score` | gauge | exports/unified_technical_review/source_code/src/cryptosmarttrader/utils/metrics.py | 15 | None |
| `data_freshness_seconds` | gauge | exports/unified_technical_review/source_code/src/cryptosmarttrader/utils/metrics.py | 16 | None |
| `prediction_accuracy` | gauge | exports/unified_technical_review/source_code/src/cryptosmarttrader/utils/metrics.py | 17 | None |
| `exchange_latency_seconds` | histogram | exports/unified_technical_review/source_code/src/cryptosmarttrader/utils/metrics.py | 18 | None |
| `crypto_requests_total` | counter | src/cryptosmarttrader/utils/metrics.py | 13 | None |
| `cache_hits_total` | counter | src/cryptosmarttrader/utils/metrics.py | 23 | None |
| `cache_misses_total` | counter | src/cryptosmarttrader/utils/metrics.py | 24 | None |
| `errors_total` | counter | src/cryptosmarttrader/utils/metrics.py | 25 | None |
| `system_health_score` | gauge | src/cryptosmarttrader/utils/metrics.py | 16 | None |
| `agent_performance_score` | gauge | src/cryptosmarttrader/utils/metrics.py | 17 | None |
| `data_freshness_seconds` | gauge | src/cryptosmarttrader/utils/metrics.py | 18 | None |
| `prediction_accuracy` | gauge | src/cryptosmarttrader/utils/metrics.py | 19 | None |
| `exchange_latency_seconds` | histogram | src/cryptosmarttrader/utils/metrics.py | 22 | None |
| `slo_violations_total` | counter | src/cryptosmarttrader/deployment/go_live_system.py | 144 | None |
| `slo_compliance_percentage` | gauge | src/cryptosmarttrader/deployment/go_live_system.py | 140 | None |
| `cryptotrader_orders_total` | counter | src/cryptosmarttrader/observability/metrics_collector.py | 150 | None |
| `cryptotrader_order_errors_total` | counter | src/cryptosmarttrader/observability/metrics_collector.py | 167 | None |
| `cryptotrader_signals_received_total` | counter | src/cryptosmarttrader/observability/metrics_collector.py | 220 | None |
| `cryptotrader_equity_usd` | gauge | src/cryptosmarttrader/observability/metrics_collector.py | 186 | None |
| `cryptotrader_drawdown_percent` | gauge | src/cryptosmarttrader/observability/metrics_collector.py | 191 | None |
| `cryptotrader_daily_pnl_usd` | gauge | src/cryptosmarttrader/observability/metrics_collector.py | 198 | None |
| `cryptotrader_position_size` | gauge | src/cryptosmarttrader/observability/metrics_collector.py | 203 | None |
| `cryptotrader_risk_score` | gauge | src/cryptosmarttrader/observability/metrics_collector.py | 211 | None |
| `cryptotrader_last_signal_timestamp` | gauge | src/cryptosmarttrader/observability/metrics_collector.py | 228 | None |
| `cryptotrader_uptime_seconds` | gauge | src/cryptosmarttrader/observability/metrics_collector.py | 236 | None |
| `cryptotrader_memory_usage_bytes` | gauge | src/cryptosmarttrader/observability/metrics_collector.py | 241 | None |
| `cryptotrader_active_connections` | gauge | src/cryptosmarttrader/observability/metrics_collector.py | 249 | None |
| `cryptotrader_api_rate_limit_usage_percent` | gauge | src/cryptosmarttrader/observability/metrics_collector.py | 268 | None |
| `cryptotrader_exchange_connectivity` | gauge | src/cryptosmarttrader/observability/metrics_collector.py | 276 | None |
| `cryptotrader_order_latency_seconds` | histogram | src/cryptosmarttrader/observability/metrics_collector.py | 158 | None |
| `cryptotrader_slippage_bps` | histogram | src/cryptosmarttrader/observability/metrics_collector.py | 175 | None |
| `cryptotrader_api_request_duration_seconds` | histogram | src/cryptosmarttrader/observability/metrics_collector.py | 259 | None |
| `trading_orders_total` | counter | src/cryptosmarttrader/observability/unified_metrics.py | 104 | None |
| `trading_order_errors_total` | counter | src/cryptosmarttrader/observability/unified_metrics.py | 111 | None |
| `ml_signals_received_total` | counter | src/cryptosmarttrader/observability/unified_metrics.py | 137 | None |
| `api_requests_total` | counter | src/cryptosmarttrader/observability/unified_metrics.py | 160 | None |
| `alerts_fired_total` | counter | src/cryptosmarttrader/observability/unified_metrics.py | 209 | None |
| `portfolio_drawdown_percent` | gauge | src/cryptosmarttrader/observability/unified_metrics.py | 126 | None |
| `portfolio_equity_usd` | gauge | src/cryptosmarttrader/observability/unified_metrics.py | 132 | None |
| `ml_signal_age_minutes` | gauge | src/cryptosmarttrader/observability/unified_metrics.py | 144 | None |
| `exchange_connectivity_status` | gauge | src/cryptosmarttrader/observability/unified_metrics.py | 175 | None |
| `data_quality_score` | gauge | src/cryptosmarttrader/observability/unified_metrics.py | 182 | None |
| `portfolio_position_count` | gauge | src/cryptosmarttrader/observability/unified_metrics.py | 198 | None |
| `portfolio_largest_position_percent` | gauge | src/cryptosmarttrader/observability/unified_metrics.py | 202 | None |
| `alerts_active_count` | gauge | src/cryptosmarttrader/observability/unified_metrics.py | 216 | None |
| `trading_slippage_bps` | histogram | src/cryptosmarttrader/observability/unified_metrics.py | 118 | None |
| `ml_prediction_confidence` | histogram | src/cryptosmarttrader/observability/unified_metrics.py | 151 | None |
| `api_duration_seconds` | histogram | src/cryptosmarttrader/observability/unified_metrics.py | 167 | None |
| `trading_orders_total` | counter | src/cryptosmarttrader/observability/centralized_prometheus.py | 115 | None |
| `trading_order_errors_total` | counter | src/cryptosmarttrader/observability/centralized_prometheus.py | 122 | None |
| `trading_fills_total` | counter | src/cryptosmarttrader/observability/centralized_prometheus.py | 138 | None |
| `trading_signals_generated_total` | counter | src/cryptosmarttrader/observability/centralized_prometheus.py | 170 | None |
| `system_uptime_seconds_total` | counter | src/cryptosmarttrader/observability/centralized_prometheus.py | 202 | None |
| `data_updates_total` | counter | src/cryptosmarttrader/observability/centralized_prometheus.py | 209 | None |
| `data_gaps_total` | counter | src/cryptosmarttrader/observability/centralized_prometheus.py | 224 | None |
| `exchange_errors_total` | counter | src/cryptosmarttrader/observability/centralized_prometheus.py | 239 | None |
| `execution_fees_usd_total` | counter | src/cryptosmarttrader/observability/centralized_prometheus.py | 281 | None |
| `risk_limit_violations_total` | counter | src/cryptosmarttrader/observability/centralized_prometheus.py | 319 | None |
| `kill_switch_triggers_total` | counter | src/cryptosmarttrader/observability/centralized_prometheus.py | 347 | None |
| `trading_slippage_p95_bps` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 162 | None |
| `trading_signals_last_generated_timestamp` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 177 | None |
| `system_health_score` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 196 | None |
| `exchange_connections_active` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 232 | None |
| `execution_partial_fills_ratio` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 266 | None |
| `execution_rejection_rate` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 273 | None |
| `portfolio_value_usd` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 300 | None |
| `portfolio_drawdown_percent` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 306 | None |
| `portfolio_daily_pnl_usd` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 312 | None |
| `position_sizes_usd` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 326 | None |
| `exposure_utilization_percent` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 333 | None |
| `kill_switch_active` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 341 | None |
| `strategy_returns_percent` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 358 | None |
| `strategy_sharpe_ratio` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 365 | None |
| `strategy_win_rate` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 372 | None |
| `backtest_live_tracking_error_bps` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 380 | None |
| `backtest_live_parity_status` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 386 | None |
| `system_cpu_usage_percent` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 393 | None |
| `system_memory_usage_mb` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 399 | None |
| `system_disk_usage_percent` | gauge | src/cryptosmarttrader/observability/centralized_prometheus.py | 405 | None |
| `trading_order_execution_duration_seconds` | histogram | src/cryptosmarttrader/observability/centralized_prometheus.py | 129 | None |
| `trading_fill_size_usd` | histogram | src/cryptosmarttrader/observability/centralized_prometheus.py | 145 | None |
| `trading_slippage_bps` | histogram | src/cryptosmarttrader/observability/centralized_prometheus.py | 154 | None |
| `trading_signal_strength` | histogram | src/cryptosmarttrader/observability/centralized_prometheus.py | 184 | None |
| `data_update_latency_seconds` | histogram | src/cryptosmarttrader/observability/centralized_prometheus.py | 216 | None |
| `exchange_latency_seconds` | histogram | src/cryptosmarttrader/observability/centralized_prometheus.py | 246 | None |
| `execution_quality_score` | histogram | src/cryptosmarttrader/observability/centralized_prometheus.py | 258 | None |
| `execution_cost_bps` | histogram | src/cryptosmarttrader/observability/centralized_prometheus.py | 288 | None |
| `orders_sent_total` | counter | src/cryptosmarttrader/observability/metrics.py | 68 | None |
| `orders_filled_total` | counter | src/cryptosmarttrader/observability/metrics.py | 75 | None |
| `order_errors_total` | counter | src/cryptosmarttrader/observability/metrics.py | 82 | None |
| `signals_received_total` | counter | src/cryptosmarttrader/observability/metrics.py | 121 | None |
| `signals_processed_total` | counter | src/cryptosmarttrader/observability/metrics.py | 128 | None |
| `api_calls_total` | counter | src/cryptosmarttrader/observability/metrics.py | 143 | None |
| `cache_operations_total` | counter | src/cryptosmarttrader/observability/metrics.py | 150 | None |
| `execution_decisions_total` | counter | src/cryptosmarttrader/observability/metrics.py | 167 | None |
| `execution_gates_total` | counter | src/cryptosmarttrader/observability/metrics.py | 174 | None |
| `risk_violations_total` | counter | src/cryptosmarttrader/observability/metrics.py | 198 | None |
| `kill_switch_triggers_total` | counter | src/cryptosmarttrader/observability/metrics.py | 235 | None |
| `api_calls_total` | counter | src/cryptosmarttrader/observability/metrics.py | 466 | None |
| `cache_hits_total` | counter | src/cryptosmarttrader/observability/metrics.py | 473 | None |
| `risk_violations_total` | counter | src/cryptosmarttrader/observability/metrics.py | 488 | None |
| `data_points_received_total` | counter | src/cryptosmarttrader/observability/metrics.py | 503 | None |
| `equity_usd` | gauge | src/cryptosmarttrader/observability/metrics.py | 106 | None |
| `drawdown_pct` | gauge | src/cryptosmarttrader/observability/metrics.py | 113 | None |
| `signal_accuracy_pct` | gauge | src/cryptosmarttrader/observability/metrics.py | 135 | None |
| `memory_usage_bytes` | gauge | src/cryptosmarttrader/observability/metrics.py | 157 | None |
| `portfolio_risk_score` | gauge | src/cryptosmarttrader/observability/metrics.py | 205 | None |
| `portfolio_equity_usd` | gauge | src/cryptosmarttrader/observability/metrics.py | 211 | None |
| `portfolio_drawdown_pct` | gauge | src/cryptosmarttrader/observability/metrics.py | 217 | None |
| `portfolio_exposure_usd` | gauge | src/cryptosmarttrader/observability/metrics.py | 223 | None |
| `portfolio_positions_count` | gauge | src/cryptosmarttrader/observability/metrics.py | 229 | None |
| `high_order_error_rate` | gauge | src/cryptosmarttrader/observability/metrics.py | 242 | None |
| `drawdown_too_high` | gauge | src/cryptosmarttrader/observability/metrics.py | 248 | None |
| `no_signals_timeout` | gauge | src/cryptosmarttrader/observability/metrics.py | 254 | None |
| `last_signal_timestamp_seconds` | gauge | src/cryptosmarttrader/observability/metrics.py | 260 | None |
| `memory_usage_bytes` | gauge | src/cryptosmarttrader/observability/metrics.py | 480 | None |
| `position_size_usd` | gauge | src/cryptosmarttrader/observability/metrics.py | 495 | None |
| `latency_ms` | histogram | src/cryptosmarttrader/observability/metrics.py | 90 | None |
| `slippage_bps` | histogram | src/cryptosmarttrader/observability/metrics.py | 98 | None |
| `execution_latency_ms` | histogram | src/cryptosmarttrader/observability/metrics.py | 181 | None |
| `estimated_slippage_bps` | histogram | src/cryptosmarttrader/observability/metrics.py | 189 | None |
| `cst_orders_total` | counter | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 46 | None |
| `cst_api_requests_total` | counter | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 114 | None |
| `cst_signals_generated_total` | counter | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 130 | None |
| `cst_alerts_total` | counter | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 169 | None |
| `cst_position_pnl_usd` | gauge | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 69 | None |
| `cst_portfolio_value_usd` | gauge | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 77 | None |
| `cst_daily_pnl_percent` | gauge | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 81 | None |
| `cst_max_drawdown_percent` | gauge | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 85 | None |
| `cst_kill_switch_active` | gauge | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 96 | None |
| `cst_data_source_last_update_timestamp` | gauge | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 103 | None |
| `cst_data_quality_score` | gauge | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 110 | None |
| `cst_predictions_accuracy_percent` | gauge | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 145 | None |
| `cst_agent_status` | gauge | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 153 | None |
| `cst_memory_usage_bytes` | gauge | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 160 | None |
| `cst_cpu_usage_percent` | gauge | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 164 | None |
| `cst_error_rate_per_minute` | gauge | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 173 | None |
| `cst_order_execution_seconds` | histogram | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 53 | None |
| `cst_slippage_percent` | histogram | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 61 | None |
| `cst_api_request_duration_seconds` | histogram | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 121 | None |
| `cst_signal_confidence` | histogram | src/cryptosmarttrader/monitoring/prometheus_metrics.py | 137 | None |
| `cryptotrader_operations_total` | counter | experiments/quarantined_modules/cryptosmarttrader/core/improved_logging_manager.py | 80 | None |
| `cryptotrader_active_agents` | gauge | experiments/quarantined_modules/cryptosmarttrader/core/improved_logging_manager.py | 95 | None |
| `cryptotrader_prediction_accuracy` | gauge | experiments/quarantined_modules/cryptosmarttrader/core/improved_logging_manager.py | 102 | None |
| `cryptotrader_operation_duration_seconds` | histogram | experiments/quarantined_modules/cryptosmarttrader/core/improved_logging_manager.py | 87 | None |
| `cc` | counter | .pythonlibs/lib/python3.11/site-packages/prometheus_client/__init__.py | 56 | None |
| `gg` | gauge | .pythonlibs/lib/python3.11/site-packages/prometheus_client/__init__.py | 59 | None |
| `hh` | histogram | .pythonlibs/lib/python3.11/site-packages/prometheus_client/__init__.py | 65 | None |
| `ss` | summary | .pythonlibs/lib/python3.11/site-packages/prometheus_client/__init__.py | 62 | None |
| `my_requests_total` | counter | .pythonlibs/lib/python3.11/site-packages/prometheus_client/metrics.py | 142 | None |
| `my_requests_total` | counter | .pythonlibs/lib/python3.11/site-packages/prometheus_client/metrics.py | 150 | None |
| `my_failures_total` | counter | .pythonlibs/lib/python3.11/site-packages/prometheus_client/metrics.py | 256 | None |
| `my_inprogress_requests` | gauge | .pythonlibs/lib/python3.11/site-packages/prometheus_client/metrics.py | 335 | None |
| `data_objects` | gauge | .pythonlibs/lib/python3.11/site-packages/prometheus_client/metrics.py | 354 | None |
| `request_size_bytes` | histogram | .pythonlibs/lib/python3.11/site-packages/prometheus_client/metrics.py | 539 | None |
| `response_latency_seconds` | histogram | .pythonlibs/lib/python3.11/site-packages/prometheus_client/metrics.py | 546 | None |
| `request_size_bytes` | summary | .pythonlibs/lib/python3.11/site-packages/prometheus_client/metrics.py | 467 | None |
| `response_latency_seconds` | summary | .pythonlibs/lib/python3.11/site-packages/prometheus_client/metrics.py | 474 | None |
| `orders_sent_total` | counter | technical_review_package/centralize_observability.py | 107 | None |
| `orders_filled_total` | counter | technical_review_package/centralize_observability.py | 114 | None |
| `order_errors_total` | counter | technical_review_package/centralize_observability.py | 121 | None |
| `signals_received_total` | counter | technical_review_package/centralize_observability.py | 160 | None |
| `signals_processed_total` | counter | technical_review_package/centralize_observability.py | 167 | None |
| `api_calls_total` | counter | technical_review_package/centralize_observability.py | 182 | None |
| `cache_hits_total` | counter | technical_review_package/centralize_observability.py | 189 | None |
| `risk_violations_total` | counter | technical_review_package/centralize_observability.py | 204 | None |
| `data_points_received_total` | counter | technical_review_package/centralize_observability.py | 219 | None |
| `equity_usd` | gauge | technical_review_package/centralize_observability.py | 145 | None |
| `drawdown_pct` | gauge | technical_review_package/centralize_observability.py | 152 | None |
| `signal_accuracy_pct` | gauge | technical_review_package/centralize_observability.py | 174 | None |
| `memory_usage_bytes` | gauge | technical_review_package/centralize_observability.py | 196 | None |
| `position_size_usd` | gauge | technical_review_package/centralize_observability.py | 211 | None |
| `data_quality_score` | gauge | technical_review_package/centralize_observability.py | 226 | None |
| `latency_ms` | histogram | technical_review_package/centralize_observability.py | 129 | None |
| `slippage_bps` | histogram | technical_review_package/centralize_observability.py | 137 | None |
| `crypto_requests_total` | counter | technical_review_package/src/cryptosmarttrader/utils/metrics.py | 13 | None |
| `cache_hits_total` | counter | technical_review_package/src/cryptosmarttrader/utils/metrics.py | 23 | None |
| `cache_misses_total` | counter | technical_review_package/src/cryptosmarttrader/utils/metrics.py | 24 | None |
| `errors_total` | counter | technical_review_package/src/cryptosmarttrader/utils/metrics.py | 25 | None |
| `system_health_score` | gauge | technical_review_package/src/cryptosmarttrader/utils/metrics.py | 16 | None |
| `agent_performance_score` | gauge | technical_review_package/src/cryptosmarttrader/utils/metrics.py | 17 | None |
| `data_freshness_seconds` | gauge | technical_review_package/src/cryptosmarttrader/utils/metrics.py | 18 | None |
| `prediction_accuracy` | gauge | technical_review_package/src/cryptosmarttrader/utils/metrics.py | 19 | None |
| `exchange_latency_seconds` | histogram | technical_review_package/src/cryptosmarttrader/utils/metrics.py | 22 | None |
| `slo_violations_total` | counter | technical_review_package/src/cryptosmarttrader/deployment/go_live_system.py | 144 | None |
| `slo_compliance_percentage` | gauge | technical_review_package/src/cryptosmarttrader/deployment/go_live_system.py | 140 | None |
| `cryptotrader_orders_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics_collector.py | 150 | None |
| `cryptotrader_order_errors_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics_collector.py | 167 | None |
| `cryptotrader_signals_received_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics_collector.py | 220 | None |
| `cryptotrader_equity_usd` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics_collector.py | 186 | None |
| `cryptotrader_drawdown_percent` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics_collector.py | 191 | None |
| `cryptotrader_daily_pnl_usd` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics_collector.py | 198 | None |
| `cryptotrader_position_size` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics_collector.py | 203 | None |
| `cryptotrader_risk_score` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics_collector.py | 211 | None |
| `cryptotrader_last_signal_timestamp` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics_collector.py | 228 | None |
| `cryptotrader_uptime_seconds` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics_collector.py | 236 | None |
| `cryptotrader_memory_usage_bytes` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics_collector.py | 241 | None |
| `cryptotrader_active_connections` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics_collector.py | 249 | None |
| `cryptotrader_api_rate_limit_usage_percent` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics_collector.py | 268 | None |
| `cryptotrader_exchange_connectivity` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics_collector.py | 276 | None |
| `cryptotrader_order_latency_seconds` | histogram | technical_review_package/src/cryptosmarttrader/observability/metrics_collector.py | 158 | None |
| `cryptotrader_slippage_bps` | histogram | technical_review_package/src/cryptosmarttrader/observability/metrics_collector.py | 175 | None |
| `cryptotrader_api_request_duration_seconds` | histogram | technical_review_package/src/cryptosmarttrader/observability/metrics_collector.py | 259 | None |
| `trading_orders_total` | counter | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 104 | None |
| `trading_order_errors_total` | counter | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 111 | None |
| `ml_signals_received_total` | counter | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 137 | None |
| `api_requests_total` | counter | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 160 | None |
| `alerts_fired_total` | counter | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 209 | None |
| `portfolio_drawdown_percent` | gauge | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 126 | None |
| `portfolio_equity_usd` | gauge | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 132 | None |
| `ml_signal_age_minutes` | gauge | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 144 | None |
| `exchange_connectivity_status` | gauge | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 175 | None |
| `data_quality_score` | gauge | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 182 | None |
| `portfolio_position_count` | gauge | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 198 | None |
| `portfolio_largest_position_percent` | gauge | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 202 | None |
| `alerts_active_count` | gauge | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 216 | None |
| `trading_slippage_bps` | histogram | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 118 | None |
| `ml_prediction_confidence` | histogram | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 151 | None |
| `api_duration_seconds` | histogram | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 167 | None |
| `orders_sent_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 51 | None |
| `orders_filled_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 58 | None |
| `order_errors_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 65 | None |
| `signals_received_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 104 | None |
| `signals_processed_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 111 | None |
| `api_calls_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 126 | None |
| `cache_operations_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 133 | None |
| `execution_decisions_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 150 | None |
| `execution_gates_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 157 | None |
| `risk_violations_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 181 | None |
| `kill_switch_triggers_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 218 | None |
| `api_calls_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 424 | None |
| `cache_hits_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 431 | None |
| `risk_violations_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 446 | None |
| `data_points_received_total` | counter | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 461 | None |
| `equity_usd` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 89 | None |
| `drawdown_pct` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 96 | None |
| `signal_accuracy_pct` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 118 | None |
| `memory_usage_bytes` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 140 | None |
| `portfolio_risk_score` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 188 | None |
| `portfolio_equity_usd` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 194 | None |
| `portfolio_drawdown_pct` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 200 | None |
| `portfolio_exposure_usd` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 206 | None |
| `portfolio_positions_count` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 212 | None |
| `high_order_error_rate` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 225 | None |
| `drawdown_too_high` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 231 | None |
| `no_signals_timeout` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 237 | None |
| `last_signal_timestamp_seconds` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 243 | None |
| `memory_usage_bytes` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 438 | None |
| `position_size_usd` | gauge | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 453 | None |
| `latency_ms` | histogram | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 73 | None |
| `slippage_bps` | histogram | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 81 | None |
| `execution_latency_ms` | histogram | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 164 | None |
| `estimated_slippage_bps` | histogram | technical_review_package/src/cryptosmarttrader/observability/metrics.py | 172 | None |
| `trading_orders_total` | counter | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 115 | None |
| `trading_order_errors_total` | counter | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 122 | None |
| `trading_fills_total` | counter | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 138 | None |
| `trading_signals_generated_total` | counter | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 170 | None |
| `system_uptime_seconds_total` | counter | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 202 | None |
| `data_updates_total` | counter | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 209 | None |
| `data_gaps_total` | counter | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 224 | None |
| `exchange_errors_total` | counter | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 239 | None |
| `execution_fees_usd_total` | counter | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 281 | None |
| `risk_limit_violations_total` | counter | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 319 | None |
| `kill_switch_triggers_total` | counter | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 347 | None |
| `trading_slippage_p95_bps` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 162 | None |
| `trading_signals_last_generated_timestamp` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 177 | None |
| `system_health_score` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 196 | None |
| `exchange_connections_active` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 232 | None |
| `execution_partial_fills_ratio` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 266 | None |
| `execution_rejection_rate` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 273 | None |
| `portfolio_value_usd` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 300 | None |
| `portfolio_drawdown_percent` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 306 | None |
| `portfolio_daily_pnl_usd` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 312 | None |
| `position_sizes_usd` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 326 | None |
| `exposure_utilization_percent` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 333 | None |
| `kill_switch_active` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 341 | None |
| `strategy_returns_percent` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 358 | None |
| `strategy_sharpe_ratio` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 365 | None |
| `strategy_win_rate` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 372 | None |
| `backtest_live_tracking_error_bps` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 380 | None |
| `backtest_live_parity_status` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 386 | None |
| `system_cpu_usage_percent` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 393 | None |
| `system_memory_usage_mb` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 399 | None |
| `system_disk_usage_percent` | gauge | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 405 | None |
| `trading_order_execution_duration_seconds` | histogram | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 129 | None |
| `trading_fill_size_usd` | histogram | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 145 | None |
| `trading_slippage_bps` | histogram | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 154 | None |
| `trading_signal_strength` | histogram | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 184 | None |
| `data_update_latency_seconds` | histogram | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 216 | None |
| `exchange_latency_seconds` | histogram | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 246 | None |
| `execution_quality_score` | histogram | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 258 | None |
| `execution_cost_bps` | histogram | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 288 | None |
| `cst_orders_total` | counter | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 46 | None |
| `cst_api_requests_total` | counter | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 114 | None |
| `cst_signals_generated_total` | counter | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 130 | None |
| `cst_alerts_total` | counter | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 169 | None |
| `cst_position_pnl_usd` | gauge | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 69 | None |
| `cst_portfolio_value_usd` | gauge | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 77 | None |
| `cst_daily_pnl_percent` | gauge | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 81 | None |
| `cst_max_drawdown_percent` | gauge | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 85 | None |
| `cst_kill_switch_active` | gauge | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 96 | None |
| `cst_data_source_last_update_timestamp` | gauge | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 103 | None |
| `cst_data_quality_score` | gauge | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 110 | None |
| `cst_predictions_accuracy_percent` | gauge | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 145 | None |
| `cst_agent_status` | gauge | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 153 | None |
| `cst_memory_usage_bytes` | gauge | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 160 | None |
| `cst_cpu_usage_percent` | gauge | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 164 | None |
| `cst_error_rate_per_minute` | gauge | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 173 | None |
| `cst_order_execution_seconds` | histogram | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 53 | None |
| `cst_slippage_percent` | histogram | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 61 | None |
| `cst_api_request_duration_seconds` | histogram | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 121 | None |
| `cst_signal_confidence` | histogram | technical_review_package/src/cryptosmarttrader/monitoring/prometheus_metrics.py | 137 | None |

## Discovered Alerts

| Name | Severity | File | Line | Query |
|------|----------|------|------|-------|
| `with self.assertRaisesRegex(` | warning | .cache/uv/archive-v0/0B1ELcZFKqVCRGLhmQmBy/torch/testing/_internal/common_utils.py | 4421 | `unknown` |
| `with self.assertWarnsRegex(` | warning | .cache/uv/archive-v0/0B1ELcZFKqVCRGLhmQmBy/torch/testing/_internal/common_utils.py | 4443 | `unknown` |
| `HighOrderErrorRate` | critical | src/cryptosmarttrader/observability/unified_metrics.py | 234 | `unknown` |
| `ExcessiveSlippage` | critical | src/cryptosmarttrader/observability/unified_metrics.py | 243 | `unknown` |
| `CriticalSlippage` | warning | src/cryptosmarttrader/observability/unified_metrics.py | 252 | `unknown` |
| `HighDrawdown` | critical | src/cryptosmarttrader/observability/unified_metrics.py | 262 | `unknown` |
| `EmergencyDrawdown` | critical | src/cryptosmarttrader/observability/unified_metrics.py | 271 | `unknown` |
| `NoSignalsReceived` | warning | src/cryptosmarttrader/observability/unified_metrics.py | 281 | `unknown` |
| `CriticalSignalGap` | warning | src/cryptosmarttrader/observability/unified_metrics.py | 290 | `unknown` |
| `ExchangeDisconnected` | critical | src/cryptosmarttrader/observability/unified_metrics.py | 300 | `unknown` |
| `LowDataQuality` | critical | src/cryptosmarttrader/observability/unified_metrics.py | 309 | `unknown` |
| `HighOrderErrorRate` | critical | src/cryptosmarttrader/observability/centralized_prometheus.py | 415 | `unknown` |
| `DrawdownTooHigh` | emergency | src/cryptosmarttrader/observability/centralized_prometheus.py | 426 | `unknown` |
| `NoSignals30m` | warning | src/cryptosmarttrader/observability/centralized_prometheus.py | 437 | `unknown` |
| `SlippageP95ExceedsBudget` | critical | src/cryptosmarttrader/observability/centralized_prometheus.py | 448 | `unknown` |
| `SystemHealthLow` | warning | src/cryptosmarttrader/observability/centralized_prometheus.py | 459 | `unknown` |
| `ExchangeDisconnected` | critical | src/cryptosmarttrader/observability/centralized_prometheus.py | 470 | `unknown` |
| `DataGapDetected` | warning | src/cryptosmarttrader/observability/centralized_prometheus.py | 481 | `unknown` |
| `KillSwitchActivated` | emergency | src/cryptosmarttrader/observability/centralized_prometheus.py | 492 | `unknown` |
| `AlertEvent):` | warning | src/cryptosmarttrader/observability/centralized_prometheus.py | 635 | `unknown` |
| `AlertEvent):` | warning | src/cryptosmarttrader/observability/centralized_prometheus.py | 645 | `unknown` |
| `SystemHealthCritical` | critical | src/cryptosmarttrader/observability/unified_metrics_alerting_system.py | 385 | `cst_system_health_score < 0.3` |
| `HighErrorRate` | critical | src/cryptosmarttrader/observability/unified_metrics_alerting_system.py | 398 | `rate(cst_errors_total[5m]) > 0.05` |
| `DrawdownExceedsLimit` | emergency | src/cryptosmarttrader/observability/unified_metrics_alerting_system.py | 410 | `cst_drawdown_current > 10.0` |
| `HighOrderLatency` | warning | src/cryptosmarttrader/observability/unified_metrics_alerting_system.py | 421 | `histogram_quantile(0.95, cst_order_latency_seconds_bucket) > 0.5` |
| `ExcessiveSlippage` | critical | src/cryptosmarttrader/observability/unified_metrics_alerting_system.py | 431 | `histogram_quantile(0.95, cst_slippage_bps_bucket) > 50` |
| `DataQualityDegraded` | warning | src/cryptosmarttrader/observability/unified_metrics_alerting_system.py | 442 | `cst_data_quality_score < 0.8` |
| `DataGapsDetected` | critical | src/cryptosmarttrader/observability/unified_metrics_alerting_system.py | 453 | `rate(cst_data_gaps_total{severity=\` |
| `ModelAccuracyDegraded` | warning | src/cryptosmarttrader/observability/unified_metrics_alerting_system.py | 464 | `cst_model_accuracy_score < 0.7` |
| `HighCpuUsage` | warning | src/cryptosmarttrader/observability/unified_metrics_alerting_system.py | 476 | `cst_cpu_usage_percent > 85` |
| `HighMemoryUsage` | critical | src/cryptosmarttrader/observability/unified_metrics_alerting_system.py | 486 | `cst_memory_usage_bytes / (1024*1024*1024) > 8` |
| `{rule.name}\n"` | warning | src/cryptosmarttrader/observability/unified_metrics_alerting_system.py | 827 | `unknown` |
| `\s*([^\n]+)', re.MULTILINE),` | warning | src/cryptosmarttrader/observability/metrics_migration_engine.py | 91 | `unknown` |
| `{alert_name} in {file_path}")` | warning | src/cryptosmarttrader/observability/metrics_migration_engine.py | 235 | `unknown` |
| `{alert.name}")` | warning | src/cryptosmarttrader/observability/metrics_migration_engine.py | 383 | `unknown` |
| `with self.assertRaisesRegex(` | warning | .pythonlibs/lib/python3.11/site-packages/torch/testing/_internal/common_utils.py | 4421 | `unknown` |
| `with self.assertWarnsRegex(` | warning | .pythonlibs/lib/python3.11/site-packages/torch/testing/_internal/common_utils.py | 4443 | `unknown` |
| `HighOrderErrorRate` | critical | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 234 | `unknown` |
| `ExcessiveSlippage` | critical | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 243 | `unknown` |
| `CriticalSlippage` | warning | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 252 | `unknown` |
| `HighDrawdown` | critical | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 262 | `unknown` |
| `EmergencyDrawdown` | critical | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 271 | `unknown` |
| `NoSignalsReceived` | warning | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 281 | `unknown` |
| `CriticalSignalGap` | warning | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 290 | `unknown` |
| `ExchangeDisconnected` | critical | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 300 | `unknown` |
| `LowDataQuality` | critical | technical_review_package/src/cryptosmarttrader/observability/unified_metrics.py | 309 | `unknown` |
| `HighOrderErrorRate` | critical | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 415 | `unknown` |
| `DrawdownTooHigh` | emergency | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 426 | `unknown` |
| `NoSignals30m` | warning | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 437 | `unknown` |
| `SlippageP95ExceedsBudget` | critical | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 448 | `unknown` |
| `SystemHealthLow` | warning | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 459 | `unknown` |
| `ExchangeDisconnected` | critical | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 470 | `unknown` |
| `DataGapDetected` | warning | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 481 | `unknown` |
| `KillSwitchActivated` | emergency | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 492 | `unknown` |
| `AlertEvent):` | warning | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 635 | `unknown` |
| `AlertEvent):` | warning | technical_review_package/src/cryptosmarttrader/observability/centralized_prometheus.py | 645 | `unknown` |

## Errors

- ❌ Error scanning .cache/uv/archive-v0/l7StDJ3bea8KVXz5p5JZR/joblib/test/test_func_inspect_special_encoding.py: 'utf-8' codec can't decode byte 0xa4 in position 64: invalid start byte
- ❌ Error scanning .pythonlibs/lib/python3.11/site-packages/joblib/test/test_func_inspect_special_encoding.py: 'utf-8' codec can't decode byte 0xa4 in position 64: invalid start byte
- ❌ Could not import unified metrics system: attempted relative import with no known parent package
- ❌ Could not import unified metrics system: attempted relative import with no known parent package

## Next Steps

1. ✅ **Review Migration:** Check that all critical metrics are included
2. ✅ **Test Integration:** Verify unified metrics system works correctly
3. ✅ **Update Code:** Replace old metric calls with unified system
4. ✅ **Deploy Alerts:** Configure alert routing and notifications
5. ✅ **Monitor Performance:** Watch for any degradation after migration

## Configuration

The unified metrics system provides:

- **Centralized Collection:** All metrics flow through single system
- **Real-time Alerting:** Multi-tier severity with escalation
- **Trend Analysis:** Automatic degradation detection
- **Multi-channel Notifications:** Slack, Email, PagerDuty integration
- **Performance Baselines:** Automatic baseline tracking
- **Circuit Breaker Integration:** Automatic failover protection

---
*Report generated by MetricsMigrationEngine*
