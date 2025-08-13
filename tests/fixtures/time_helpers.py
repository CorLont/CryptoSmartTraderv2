#!/usr/bin/env python3
"""
Time Helpers - Deterministic time handling for tests
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch
import time
from contextlib import contextmanager
from typing import Union, Generator


class DeterministicTime:
    """
    Deterministic time management for tests

    Provides consistent, controllable time for testing time-dependent logic
    """

    def __init__(self, fixed_time: Union[str, datetime] = None):
        if isinstance(fixed_time, str):
            self.fixed_time = datetime.fromisoformat(fixed_time.replace("Z", "+00:00"))
        elif isinstance(fixed_time, datetime):
            self.fixed_time = fixed_time
        else:
            # Default to a fixed point in time
            self.fixed_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        self.current_time = self.fixed_time
        self.time_patches = []

    def advance_time(self, seconds: int = 0, minutes: int = 0, hours: int = 0, days: int = 0):
        """Advance the mock time by specified amounts"""
        from datetime import timedelta

        delta = timedelta(seconds=seconds, minutes=minutes, hours=hours, days=days)

        self.current_time += delta
        return self.current_time

    def set_time(self, new_time: Union[str, datetime]):
        """Set the mock time to a specific value"""
        if isinstance(new_time, str):
            self.current_time = datetime.fromisoformat(new_time.replace("Z", "+00:00"))
        else:
            self.current_time = new_time

        return self.current_time

    def mock_datetime_now(self, tz=None):
        """Mock function for datetime.now()"""
        if tz:
            return self.current_time.replace(tzinfo=tz)
        return self.current_time

    def mock_datetime_utcnow(self):
        """Mock function for datetime.utcnow()"""
        return self.current_time.replace(tzinfo=None)

    def mock_time_time(self):
        """Mock function for time.time()"""
        return self.current_time.timestamp()

    @contextmanager
    def freeze_time(self, frozen_time: Union[str, datetime] = None):
        """Context manager to freeze time during test execution"""
        if frozen_time:
            self.set_time(frozen_time)

        with (
            patch("datetime.datetime") as mock_datetime,
            patch("time.time", side_effect=self.mock_time_time),
            patch("time.sleep"),
        ):  # Prevent actual sleeping in tests
            # Configure datetime mock
            mock_datetime.now.side_effect = self.mock_datetime_now
            mock_datetime.utcnow.side_effect = self.mock_datetime_utcnow
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            # Forward other datetime attributes
            for attr in ["min", "max", "resolution", "today", "fromtimestamp", "fromisoformat"]:
                if hasattr(datetime, attr):
                    setattr(mock_datetime, attr, getattr(datetime, attr))

            yield self

    def get_timestamp_range(self, start_offset_hours: int = -24, end_offset_hours: int = 0):
        """Get a range of timestamps relative to current time"""
        from datetime import timedelta

        start_time = self.current_time + timedelta(hours=start_offset_hours)
        end_time = self.current_time + timedelta(hours=end_offset_hours)

        return start_time, end_time


@pytest.fixture
def deterministic_time():
    """Pytest fixture for deterministic time"""
    return DeterministicTime()


@pytest.fixture
def frozen_time():
    """Pytest fixture that automatically freezes time"""
    dt = DeterministicTime()
    with dt.freeze_time():
        yield dt


@pytest.fixture
def market_time():
    """Pytest fixture for market trading hours (9:30 AM EST on a weekday)"""
    # Monday, January 15, 2024, 9:30 AM EST (14:30 UTC)
    market_open = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
    dt = DeterministicTime(market_open)
    with dt.freeze_time():
        yield dt


@pytest.fixture
def weekend_time():
    """Pytest fixture for weekend time (markets closed)"""
    # Saturday, January 13, 2024, 12:00 PM UTC
    weekend = datetime(2024, 1, 13, 12, 0, 0, tzinfo=timezone.utc)
    dt = DeterministicTime(weekend)
    with dt.freeze_time():
        yield dt


def create_time_series(
    start_time: datetime, periods: int, freq_minutes: int = 60
) -> list[datetime]:
    """Create a deterministic time series for testing"""
    from datetime import timedelta

    timestamps = []
    current_time = start_time

    for _ in range(periods):
        timestamps.append(current_time)
        current_time += timedelta(minutes=freq_minutes)

    return timestamps


class TimeBasedTestCase:
    """Base class for tests that need time control"""

    def setup_method(self):
        """Setup deterministic time for each test method"""
        self.time_controller = DeterministicTime()

    def advance_time(self, **kwargs):
        """Convenience method to advance time"""
        return self.time_controller.advance_time(**kwargs)

    def set_time(self, new_time):
        """Convenience method to set time"""
        return self.time_controller.set_time(new_time)

    @contextmanager
    def time_freeze(self, frozen_time=None):
        """Convenience context manager for freezing time"""
        with self.time_controller.freeze_time(frozen_time):
            yield self.time_controller


# Decorators for easy test time control
def freeze_time_at(time_string: str):
    """Decorator to freeze time at a specific point"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            dt = DeterministicTime(time_string)
            with dt.freeze_time():
                return func(*args, **kwargs)

        return wrapper

    return decorator


def with_time_control(func):
    """Decorator to add time control to test function"""

    def wrapper(*args, **kwargs):
        time_controller = DeterministicTime()
        return func(*args, time_controller=time_controller, **kwargs)

    return wrapper
