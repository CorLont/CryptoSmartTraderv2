#!/usr/bin/env python3
"""
Session State Manager - Centralized Streamlit session state management
"""

import streamlit as st
from typing import Any, Dict, Optional
from datetime import datetime
import logging


class SessionStateManager:
    """
    Centralized session state management for Streamlit dashboard

    Features:
    - Default value initialization
    - State persistence
    - Type validation
    - Session cleanup
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._defaults = self._get_default_state()

    def _get_default_state(self) -> Dict[str, Any]:
        """Define default session state values"""
        return {
            # Navigation
            "current_page": "market",
            "page_selection": "🏪 Market Overview",
            # Dashboard settings
            "auto_refresh": True,
            "refresh_interval": 30,
            "last_refresh_time": 0,
            "dark_mode": False,
            # System status
            "system_health_score": 0.0,
            "system_health_grade": "F",
            "trading_enabled": False,
            "trading_mode": "paper",  # paper, live
            # Portfolio data
            "portfolio_value": 0.0,
            "daily_pnl": 0.0,
            "active_positions": 0,
            "total_realized_pnl": 0.0,
            # Market data
            "selected_symbol": "BTC/USD",
            "selected_timeframe": "1h",
            "market_overview_symbols": ["BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD", "ADA/USD"],
            # Agent status
            "agents_last_update": None,
            "signals_today": 0,
            "successful_signals": 0,
            # Filters and preferences
            "min_confidence": 0.7,
            "show_only_profitable": False,
            "preferred_exchanges": ["kraken", "binance"],
            # Performance tracking
            "page_load_time": 0.0,
            "api_response_times": [],
            "memory_usage_mb": 0.0,
            "active_connections": 0,
            # Cache management
            "cache_manager_initialized": False,
            "cache_warm_up_complete": False,
            "async_updater_started": False,
            # UI state
            "sidebar_expanded": True,
            "show_advanced_metrics": False,
            "chart_height": 400,
            "table_page_size": 25,
            # Alerts and notifications
            "show_notifications": True,
            "notification_level": "info",  # debug, info, warning, error
            "last_alert_time": None,
            # Session metadata
            "session_start_time": datetime.utcnow(),
            "correlation_id": None,
            "user_preferences_loaded": False,
        }

    def initialize_session(self):
        """Initialize session state with default values"""

        for key, default_value in self._defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                self.logger.debug(f"Initialized session state: {key} = {default_value}")

        # Mark session as initialized
        if not st.session_state.get("session_initialized", False):
            st.session_state.session_initialized = True
            st.session_state.session_start_time = datetime.utcnow()
            self.logger.info("Session state initialized")

    def get(self, key: str, default: Any = None) -> Any:
        """Get session state value with optional default"""
        return st.session_state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set session state value with logging"""
        old_value = st.session_state.get(key)
        st.session_state[key] = value

        if old_value != value:
            self.logger.debug(f"Session state updated: {key} = {value}")

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple session state values"""
        for key, value in updates.items():
            self.set(key, value)

    def increment(self, key: str, amount: float = 1.0) -> float:
        """Increment numeric session state value"""
        current_value = self.get(key, 0)
        new_value = current_value + amount
        self.set(key, new_value)
        return new_value

    def toggle(self, key: str) -> bool:
        """Toggle boolean session state value"""
        current_value = self.get(key, False)
        new_value = not current_value
        self.set(key, new_value)
        return new_value

    def append_to_list(self, key: str, item: Any, max_length: int = 100) -> None:
        """Append item to list in session state with max length"""
        current_list = self.get(key, [])
        current_list.append(item)

        # Trim list if too long
        if len(current_list) > max_length:
            current_list = current_list[-max_length:]

        self.set(key, current_list)

    def clear_cache_related_state(self):
        """Clear cache-related session state"""
        cache_keys = ["cache_warm_up_complete", "last_refresh_time", "agents_last_update"]

        for key in cache_keys:
            if key in st.session_state:
                del st.session_state[key]

        self.logger.info("Cache-related session state cleared")

    def save_user_preferences(self):
        """Save user preferences (placeholder for future implementation)"""
        preferences = {
            "auto_refresh": self.get("auto_refresh"),
            "refresh_interval": self.get("refresh_interval"),
            "dark_mode": self.get("dark_mode"),
            "preferred_exchanges": self.get("preferred_exchanges"),
            "min_confidence": self.get("min_confidence"),
            "show_notifications": self.get("show_notifications"),
            "notification_level": self.get("notification_level"),
        }

        # In a real implementation, save to database or local storage
        self.logger.info("User preferences saved")
        return preferences

    def load_user_preferences(self, preferences: Optional[Dict[str, Any]] = None):
        """Load user preferences"""
        if preferences:
            self.update(preferences)
            self.set("user_preferences_loaded", True)
            self.logger.info("User preferences loaded")

    def reset_session(self):
        """Reset session state to defaults"""
        # Keep certain keys that shouldn't be reset
        keep_keys = ["correlation_id", "session_start_time"]

        for key in list(st.session_state.keys()):
            if key not in keep_keys:
                del st.session_state[key]

        # Reinitialize with defaults
        self.initialize_session()
        self.logger.info("Session state reset")

    def get_session_info(self) -> Dict[str, Any]:
        """Get session information for debugging"""
        return {
            "session_id": self.get("correlation_id"),
            "start_time": self.get("session_start_time"),
            "current_page": self.get("current_page"),
            "auto_refresh": self.get("auto_refresh"),
            "trading_enabled": self.get("trading_enabled"),
            "cache_warm_up_complete": self.get("cache_warm_up_complete"),
            "total_keys": len(st.session_state),
            "initialized": self.get("session_initialized", False),
        }

    def cleanup_old_data(self):
        """Clean up old data from session state"""
        # Clean up old API response times (keep last 100)
        api_times = self.get("api_response_times", [])
        if len(api_times) > 100:
            self.set("api_response_times", api_times[-100:])

        self.logger.debug("Session state cleanup completed")

    def validate_state(self) -> Dict[str, str]:
        """Validate session state and return any issues"""
        issues = {}

        # Validate required keys exist
        required_keys = ["current_page", "correlation_id", "system_health_score"]
        for key in required_keys:
            if key not in st.session_state:
                issues[key] = f"Missing required key: {key}"

        # Validate data types
        type_checks = {
            "system_health_score": (int, float),
            "auto_refresh": bool,
            "refresh_interval": (int, float),
            "active_positions": int,
        }

        for key, expected_types in type_checks.items():
            if key in st.session_state:
                value = st.session_state[key]
                if not isinstance(value, expected_types):
                    issues[key] = (
                        f"Invalid type for {key}: expected {expected_types}, got {type(value)}"
                    )

        # Validate ranges
        range_checks = {
            "system_health_score": (0, 100),
            "refresh_interval": (10, 300),
            "min_confidence": (0, 1),
        }

        for key, (min_val, max_val) in range_checks.items():
            if key in st.session_state:
                value = st.session_state[key]
                if isinstance(value, (int, float)) and not (min_val <= value <= max_val):
                    issues[key] = (
                        f"Value out of range for {key}: {value} not in [{min_val}, {max_val}]"
                    )

        if issues:
            self.logger.warning(f"Session state validation issues: {issues}")

        return issues


# Global session state manager
session_manager = SessionStateManager()
