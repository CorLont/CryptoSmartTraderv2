#!/usr/bin/env python3
"""
Backward compatibility aliases for Technical Analysis
DEPRECATED: Use src.cryptosmarttrader.analysis.enterprise_technical_analysis directly
"""

import warnings
from src.cryptosmarttrader.analysis.enterprise_technical_analysis import get_technical_analyzer

# Deprecated functions - use enterprise framework instead
def calculate_rsi(prices, period=14):
    warnings.warn("calculate_rsi is deprecated. Use get_technical_analyzer().calculate_indicator('RSI', prices, period=period)", DeprecationWarning)
    result = get_technical_analyzer().calculate_indicator("RSI", prices, period=period)
    return result.values

def calculate_macd(prices, fast=12, slow=26, signal=9):
    warnings.warn("calculate_macd is deprecated. Use get_technical_analyzer().calculate_indicator('MACD', prices, fast=fast, slow=slow, signal=signal)", DeprecationWarning)
    result = get_technical_analyzer().calculate_indicator("MACD", prices, fast=fast, slow=slow, signal=signal)
    return result.values["macd"], result.values["signal"], result.values["histogram"]

def calculate_bollinger(prices, period=20, std_dev=2.0):
    warnings.warn("calculate_bollinger is deprecated. Use get_technical_analyzer().calculate_indicator('BollingerBands', prices, period=period, std_dev=std_dev)", DeprecationWarning)
    result = get_technical_analyzer().calculate_indicator("BollingerBands", prices, period=period, std_dev=std_dev)
    return result.values["upper"], result.values["middle"], result.values["lower"]
