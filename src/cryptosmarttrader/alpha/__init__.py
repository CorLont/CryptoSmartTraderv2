"""
Alpha Generation Module - Core coin selection en portfolio construction

Dit module bevat de alpha motor voor coin picking met:
- Multi-factor signal generation
- Risk-adjusted portfolio construction  
- Kelly sizing met correlation caps
- Execution quality assessment
"""

from .coin_picker_alpha_motor import get_alpha_motor, CoinCandidate, SignalBucket
from .market_data_simulator import MarketDataSimulator

__all__ = [
    'get_alpha_motor',
    'CoinCandidate', 
    'SignalBucket',
    'MarketDataSimulator'
]