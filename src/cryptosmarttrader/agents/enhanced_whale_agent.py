"""
Enhanced Whale/On-Chain Analysis Agent
Addresses: async pipeline, label accuracy, event detection, false positives
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import time
from pathlib import Path
import hashlib

from utils.daily_logger import get_daily_logger

@dataclass
class WhaleTransaction:
    """Enhanced whale transaction with context"""
    tx_hash: str
    from_address: str
    to_address: str
    amount: float
    token: str
    usd_value: float
    timestamp: datetime
    transaction_type: str  # transfer, exchange_deposit, exchange_withdrawal, defi_interaction
    confidence: float  # 0 to 1
    context: str  # human-readable context
    labels: Dict[str, str]  # address labels
    false_positive_score: float  # 0 to 1, higher = more likely false positive

@dataclass
class OnChainEvent:
    """Important on-chain event detection"""
    event_type: str  # unlock, governance, exploit, large_mint
    token: str
    description: str
    impact_score: float  # 0 to 1
    timestamp: datetime
    related_addresses: List[str]
    confidence: float

class AddressLabeling:
    """Advanced address labeling and classification"""
    
    def __init__(self):
        self.known_labels = {}
        self.label_cache = {}
        self.load_known_labels()
        
    def load_known_labels(self):
        """Load known address labels from various sources"""
        # This would integrate with services like Etherscan, DeBank, etc.
        self.known_labels = {
            # Exchange addresses (examples)
            '0x28c6c06298d514db089934071355e5743bf21d60': {'type': 'exchange', 'name': 'Binance 14'},
            '0x21a31ee1afc51d94c2efccaa2092ad1028285549': {'type': 'exchange', 'name': 'Binance 15'},
            '0xdfd5293d8e347dfe59e90efd55b2956a1343963d': {'type': 'exchange', 'name': 'Binance 16'},
            # Add more known addresses
        }
    
    async def label_address(self, address: str) -> Dict[str, str]:
        """Label an address with type and name"""
        
        # Check cache first
        if address in self.label_cache:
            return self.label_cache[address]
        
        # Check known labels
        if address in self.known_labels:
            label = self.known_labels[address]
            self.label_cache[address] = label
            return label
        
        # Analyze address behavior
        label = await self._analyze_address_behavior(address)
        self.label_cache[address] = label
        return label
    
    async def _analyze_address_behavior(self, address: str) -> Dict[str, str]:
        """Analyze address to determine type"""
        
        # This would implement heuristics to classify addresses:
        # - Transaction patterns
        # - Contract analysis
        # - Interaction patterns
        
        # For now, return unknown
        return {'type': 'unknown', 'name': f'Unknown_{address[:8]}'}

class EventDetector:
    """Detect important on-chain events"""
    
    def __init__(self):
        self.logger = get_daily_logger().get_logger('whale')
        
    async def detect_events(self, token: str, timeframe_hours: int = 24) -> List[OnChainEvent]:
        """Detect important events for a token"""
        
        events = []
        
        # Token unlock detection
        unlock_events = await self._detect_token_unlocks(token, timeframe_hours)
        events.extend(unlock_events)
        
        # Large mint detection
        mint_events = await self._detect_large_mints(token, timeframe_hours)
        events.extend(mint_events)
        
        # Governance events
        governance_events = await self._detect_governance_events(token, timeframe_hours)
        events.extend(governance_events)
        
        return events
    
    async def _detect_token_unlocks(self, token: str, hours: int) -> List[OnChainEvent]:
        """Detect token unlock events"""
        # Implement actual unlock detection logic
        return []
    
    async def _detect_large_mints(self, token: str, hours: int) -> List[OnChainEvent]:
        """Detect large token minting events"""
        # Implement minting detection logic
        return []
    
    async def _detect_governance_events(self, token: str, hours: int) -> List[OnChainEvent]:
        """Detect governance proposals and votes"""
        # Implement governance event detection
        return []

class FalsePositiveFilter:
    """Filter out false positive whale movements"""
    
    def __init__(self):
        self.logger = get_daily_logger().get_logger('whale')
        
    def calculate_false_positive_score(self, 
                                     transaction: Dict,
                                     from_label: Dict[str, str],
                                     to_label: Dict[str, str]) -> float:
        """Calculate probability that this is a false positive"""
        
        score = 0.0
        
        # Internal exchange movements are often false positives
        if (from_label.get('type') == 'exchange' and 
            to_label.get('type') == 'exchange' and
            from_label.get('name', '').split()[0] == to_label.get('name', '').split()[0]):
            score += 0.8  # Same exchange
        
        # Contract interactions might be automated
        if to_label.get('type') == 'contract':
            score += 0.3
        
        # Round number amounts are often operational
        amount = transaction.get('amount', 0)
        if self._is_round_number(amount):
            score += 0.2
        
        # Frequent small transactions from same address
        if self._is_frequent_trader(transaction.get('from_address', '')):
            score += 0.4
        
        return min(score, 1.0)
    
    def _is_round_number(self, amount: float) -> bool:
        """Check if amount is a round number (operational transfer)"""
        # Check for round numbers like 1000, 5000, 10000, etc.
        round_numbers = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
        for num in round_numbers:
            if abs(amount - num) / num < 0.01:  # Within 1%
                return True
        return False
    
    def _is_frequent_trader(self, address: str) -> bool:
        """Check if address is a frequent trader (less significant)"""
        # This would check transaction frequency in database
        # For now, return False
        return False

class AsyncOnChainPipeline:
    """High-performance async on-chain data pipeline"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        self.logger = get_daily_logger().get_logger('whale')
        self.retry_config = {
            'max_retries': 3,
            'base_delay': 1.0,
            'max_delay': 10.0
        }
        
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_transactions(self, 
                                addresses: List[str],
                                min_value_usd: float = 100000) -> List[Dict]:
        """Fetch transactions for multiple addresses concurrently"""
        
        tasks = []
        for address in addresses:
            task = self._fetch_address_transactions(address, min_value_usd)
            tasks.append(task)
        
        # Execute with concurrency limit
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and filter exceptions
        all_transactions = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Error fetching transactions: {result}")
                continue
            if isinstance(result, list):
                all_transactions.extend(result)
        
        return all_transactions
    
    async def _fetch_address_transactions(self, 
                                        address: str,
                                        min_value_usd: float) -> List[Dict]:
        """Fetch transactions for single address with retry logic"""
        
        async with self.semaphore:
            for attempt in range(self.retry_config['max_retries']):
                try:
                    # Simulate API call (replace with actual implementation)
                    await asyncio.sleep(0.1)  # Simulate network delay
                    
                    # Mock transaction data
                    transactions = []
                    if attempt == 0:  # Simulate occasional failure
                        import random
                        if random.random() < 0.1:  # 10% failure rate
                            raise aiohttp.ClientError("Simulated API error")
                    
                    # Generate mock transactions
                    for i in range(random.randint(0, 5)):
                        transactions.append({
                            'hash': f"0x{''.join(random.choices('0123456789abcdef', k=64))}",
                            'from': address,
                            'to': f"0x{''.join(random.choices('0123456789abcdef', k=40))}",
                            'amount': random.uniform(min_value_usd, min_value_usd * 10),
                            'timestamp': time.time() - random.randint(0, 86400),
                            'token': 'ETH'
                        })
                    
                    return transactions
                    
                except Exception as e:
                    if attempt == self.retry_config['max_retries'] - 1:
                        self.logger.error(f"Failed to fetch transactions for {address} after {self.retry_config['max_retries']} attempts: {e}")
                        return []
                    
                    delay = min(
                        self.retry_config['base_delay'] * (2 ** attempt),
                        self.retry_config['max_delay']
                    )
                    await asyncio.sleep(delay)
        
        return []

class EnhancedWhaleAgent:
    """Professional whale detection with context awareness"""
    
    def __init__(self):
        self.logger = get_daily_logger().get_logger('whale')
        self.address_labeler = AddressLabeling()
        self.event_detector = EventDetector()
        self.fp_filter = FalsePositiveFilter()
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def analyze_whale_activity(self, 
                                   tokens: List[str],
                                   min_value_usd: float = 100000,
                                   timeframe_hours: int = 24) -> Dict[str, List[WhaleTransaction]]:
        """Comprehensive whale activity analysis"""
        
        results = {}
        
        for token in tokens:
            self.logger.info(f"Analyzing whale activity for {token}")
            
            # Get relevant addresses for this token
            relevant_addresses = await self._get_relevant_addresses(token)
            
            # Fetch transactions using async pipeline
            async with AsyncOnChainPipeline() as pipeline:
                raw_transactions = await pipeline.fetch_transactions(
                    relevant_addresses, min_value_usd
                )
            
            # Process and enrich transactions
            whale_transactions = []
            for tx in raw_transactions:
                enhanced_tx = await self._enrich_transaction(tx, token)
                if enhanced_tx and enhanced_tx.false_positive_score < 0.7:
                    whale_transactions.append(enhanced_tx)
            
            # Detect related events
            events = await self.event_detector.detect_events(token, timeframe_hours)
            
            # Sort by significance
            whale_transactions.sort(key=lambda x: x.usd_value * x.confidence, reverse=True)
            
            results[token] = whale_transactions
            
            self.logger.info(f"Found {len(whale_transactions)} significant whale transactions for {token}")
        
        return results
    
    async def _get_relevant_addresses(self, token: str) -> List[str]:
        """Get relevant addresses to monitor for a token"""
        
        # This would include:
        # - Top holders
        # - Exchange addresses
        # - DeFi protocol addresses
        # - Known whale addresses
        
        # Mock implementation
        addresses = [
            f"0x{''.join(['a'] * 40)}",  # Mock whale address
            f"0x{''.join(['b'] * 40)}",  # Mock exchange
            f"0x{''.join(['c'] * 40)}",  # Mock DeFi protocol
        ]
        
        return addresses
    
    async def _enrich_transaction(self, 
                                tx: Dict,
                                token: str) -> Optional[WhaleTransaction]:
        """Enrich transaction with labels and context"""
        
        try:
            # Label addresses
            from_label = await self.address_labeler.label_address(tx['from'])
            to_label = await self.address_labeler.label_address(tx['to'])
            
            # Determine transaction type
            tx_type = self._classify_transaction_type(from_label, to_label)
            
            # Calculate false positive score
            fp_score = self.fp_filter.calculate_false_positive_score(
                tx, from_label, to_label
            )
            
            # Generate context
            context = self._generate_context(tx, from_label, to_label, tx_type)
            
            # Calculate confidence
            confidence = 1.0 - fp_score
            
            return WhaleTransaction(
                tx_hash=tx['hash'],
                from_address=tx['from'],
                to_address=tx['to'],
                amount=tx['amount'],
                token=token,
                usd_value=tx['amount'],  # Simplified
                timestamp=datetime.fromtimestamp(tx['timestamp']),
                transaction_type=tx_type,
                confidence=confidence,
                context=context,
                labels={'from': from_label, 'to': to_label},
                false_positive_score=fp_score
            )
            
        except Exception as e:
            self.logger.error(f"Error enriching transaction {tx.get('hash', 'unknown')}: {e}")
            return None
    
    def _classify_transaction_type(self, 
                                 from_label: Dict[str, str],
                                 to_label: Dict[str, str]) -> str:
        """Classify transaction type based on labels"""
        
        from_type = from_label.get('type', 'unknown')
        to_type = to_label.get('type', 'unknown')
        
        if from_type == 'exchange' and to_type != 'exchange':
            return 'exchange_withdrawal'
        elif from_type != 'exchange' and to_type == 'exchange':
            return 'exchange_deposit'
        elif 'defi' in from_type or 'defi' in to_type:
            return 'defi_interaction'
        else:
            return 'transfer'
    
    def _generate_context(self, 
                         tx: Dict,
                         from_label: Dict[str, str],
                         to_label: Dict[str, str],
                         tx_type: str) -> str:
        """Generate human-readable context"""
        
        from_name = from_label.get('name', 'Unknown')
        to_name = to_label.get('name', 'Unknown')
        amount = tx['amount']
        
        if tx_type == 'exchange_withdrawal':
            return f"Large withdrawal of {amount:,.0f} from {from_name}"
        elif tx_type == 'exchange_deposit':
            return f"Large deposit of {amount:,.0f} to {to_name}"
        elif tx_type == 'defi_interaction':
            return f"DeFi interaction: {amount:,.0f} between {from_name} and {to_name}"
        else:
            return f"Large transfer of {amount:,.0f} from {from_name} to {to_name}"
    
    def get_status(self) -> Dict:
        """Get agent status"""
        return {
            'agent': 'enhanced_whale',
            'status': 'operational',
            'address_labels_cached': len(self.address_labeler.label_cache),
            'async_pipeline': True,
            'event_detection': True,
            'false_positive_filtering': True,
            'cache_size': len(self.cache)
        }

# Global instance
whale_agent = EnhancedWhaleAgent()