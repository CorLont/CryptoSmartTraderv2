#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Multimodal Data Processor
Processing text, time series, and graph features for deep learning models
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import re
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

@dataclass
class MultimodalConfig:
    """Configuration for multimodal processing"""
    # Text processing
    max_text_length: int = 512
    text_model_name: str = "distilbert-base-uncased"
    sentiment_threshold: float = 0.1
    
    # Time series processing
    sequence_length: int = 100
    time_features: List[str] = field(default_factory=lambda: ['hour', 'day_of_week', 'month'])
    technical_indicators: List[str] = field(default_factory=lambda: ['sma', 'ema', 'rsi', 'macd', 'bollinger'])
    
    # Graph processing
    max_graph_nodes: int = 1000
    node_feature_dim: int = 64
    edge_feature_dim: int = 32
    
    # Feature scaling
    normalize_features: bool = True
    feature_scaling_method: str = "standard"  # standard, minmax, robust

class TextProcessor:
    """Advanced text processing for sentiment and news analysis"""
    
    def __init__(self, config: MultimodalConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.TextProcessor")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.text_model = None
        
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
                self.text_model = AutoModel.from_pretrained(config.text_model_name)
                self.text_model.eval()
                self.logger.info(f"Loaded text model: {config.text_model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load transformer model: {e}")
                self.tokenizer = None
                self.text_model = None
        
        # Text preprocessing patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
        self.hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
        self.emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+')
    
    def process_text_data(self, text_data: List[str]) -> Dict[str, np.ndarray]:
        """
        Process text data into features
        
        Args:
            text_data: List of text strings
            
        Returns:
            Dictionary of processed text features
        """
        try:
            processed_features = {}
            
            # Clean and preprocess text
            cleaned_texts = [self._clean_text(text) for text in text_data]
            
            # Extract basic text features
            basic_features = self._extract_basic_text_features(cleaned_texts)
            processed_features.update(basic_features)
            
            # Extract sentiment features
            sentiment_features = self._extract_sentiment_features(cleaned_texts)
            processed_features.update(sentiment_features)
            
            # Extract embeddings if transformer model available
            if self.text_model is not None:
                embeddings = self._extract_text_embeddings(cleaned_texts)
                processed_features['text_embeddings'] = embeddings
            
            # Extract keyword features
            keyword_features = self._extract_keyword_features(cleaned_texts)
            processed_features.update(keyword_features)
            
            return processed_features
            
        except Exception as e:
            self.logger.error(f"Text processing failed: {e}")
            return {}
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.url_pattern.sub(' [URL] ', text)
        
        # Replace mentions and hashtags
        text = self.mention_pattern.sub(' [MENTION] ', text)
        text = self.hashtag_pattern.sub(' [HASHTAG] ', text)
        
        # Remove emojis (or replace with [EMOJI])
        text = self.emoji_pattern.sub(' [EMOJI] ', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_basic_text_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Extract basic text features"""
        features = {}
        
        # Text length features
        char_counts = np.array([len(text) for text in texts], dtype=np.float32)
        word_counts = np.array([len(text.split()) for text in texts], dtype=np.float32)
        
        features['text_char_count'] = char_counts
        features['text_word_count'] = word_counts
        features['text_avg_word_length'] = np.array([
            np.mean([len(word) for word in text.split()]) if text.split() else 0
            for text in texts
        ], dtype=np.float32)
        
        # Special token counts
        features['url_count'] = np.array([text.count('[URL]') for text in texts], dtype=np.float32)
        features['mention_count'] = np.array([text.count('[MENTION]') for text in texts], dtype=np.float32)
        features['hashtag_count'] = np.array([text.count('[HASHTAG]') for text in texts], dtype=np.float32)
        features['emoji_count'] = np.array([text.count('[EMOJI]') for text in texts], dtype=np.float32)
        
        # Punctuation features
        features['exclamation_count'] = np.array([text.count('!') for text in texts], dtype=np.float32)
        features['question_count'] = np.array([text.count('?') for text in texts], dtype=np.float32)
        features['uppercase_ratio'] = np.array([
            sum(1 for c in text if c.isupper()) / max(len(text), 1) for text in texts
        ], dtype=np.float32)
        
        return features
    
    def _extract_sentiment_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Extract sentiment features"""
        features = {}
        
        if HAS_TEXTBLOB:
            sentiments = []
            subjectivities = []
            
            for text in texts:
                try:
                    blob = TextBlob(text)
                    sentiments.append(blob.sentiment.polarity)
                    subjectivities.append(blob.sentiment.subjectivity)
                except:
                    sentiments.append(0.0)
                    subjectivities.append(0.0)
            
            features['sentiment_polarity'] = np.array(sentiments, dtype=np.float32)
            features['sentiment_subjectivity'] = np.array(subjectivities, dtype=np.float32)
        
        # Rule-based sentiment indicators
        bullish_keywords = ['moon', 'pump', 'bull', 'buy', 'bullish', 'up', 'rise', 'gain', 'profit']
        bearish_keywords = ['dump', 'bear', 'sell', 'bearish', 'down', 'fall', 'loss', 'crash']
        
        bullish_scores = np.array([
            sum(keyword in text.lower() for keyword in bullish_keywords) for text in texts
        ], dtype=np.float32)
        
        bearish_scores = np.array([
            sum(keyword in text.lower() for keyword in bearish_keywords) for text in texts
        ], dtype=np.float32)
        
        features['bullish_keyword_count'] = bullish_scores
        features['bearish_keyword_count'] = bearish_scores
        features['sentiment_ratio'] = np.where(
            bullish_scores + bearish_scores > 0,
            (bullish_scores - bearish_scores) / (bullish_scores + bearish_scores),
            0.0
        )
        
        return features
    
    def _extract_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract text embeddings using transformer model"""
        try:
            embeddings = []
            
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    max_length=self.config.max_text_length,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.text_model(**inputs)
                    # Use [CLS] token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                    embeddings.append(embedding)
            
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Text embedding extraction failed: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), 768), dtype=np.float32)
    
    def _extract_keyword_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Extract cryptocurrency and trading keyword features"""
        crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'blockchain', 'defi',
            'nft', 'altcoin', 'hodl', 'whale', 'mining', 'staking'
        ]
        
        trading_keywords = [
            'trade', 'trading', 'buy', 'sell', 'order', 'market', 'limit',
            'stop', 'portfolio', 'investment', 'exchange', 'volume'
        ]
        
        features = {}
        
        crypto_counts = np.array([
            sum(keyword in text.lower() for keyword in crypto_keywords) for text in texts
        ], dtype=np.float32)
        
        trading_counts = np.array([
            sum(keyword in text.lower() for keyword in trading_keywords) for text in texts
        ], dtype=np.float32)
        
        features['crypto_keyword_count'] = crypto_counts
        features['trading_keyword_count'] = trading_counts
        features['total_keyword_density'] = (crypto_counts + trading_counts) / np.maximum(
            np.array([len(text.split()) for text in texts]), 1
        )
        
        return features

class TimeSeriesProcessor:
    """Advanced time series processing for price and indicator data"""
    
    def __init__(self, config: MultimodalConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.TimeSeriesProcessor")
    
    def process_timeseries_data(self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Process time series data into features
        
        Args:
            price_data: Price data DataFrame
            volume_data: Optional volume data DataFrame
            
        Returns:
            Dictionary of processed time series features
        """
        try:
            processed_features = {}
            
            # Basic price features
            price_features = self._extract_price_features(price_data)
            processed_features.update(price_features)
            
            # Technical indicators
            technical_features = self._extract_technical_indicators(price_data)
            processed_features.update(technical_features)
            
            # Time-based features
            time_features = self._extract_time_features(price_data)
            processed_features.update(time_features)
            
            # Volume features if available
            if volume_data is not None:
                volume_features = self._extract_volume_features(volume_data, price_data)
                processed_features.update(volume_features)
            
            # Rolling statistics
            rolling_features = self._extract_rolling_statistics(price_data)
            processed_features.update(rolling_features)
            
            # Volatility features
            volatility_features = self._extract_volatility_features(price_data)
            processed_features.update(volatility_features)
            
            return processed_features
            
        except Exception as e:
            self.logger.error(f"Time series processing failed: {e}")
            return {}
    
    def _extract_price_features(self, price_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract basic price features"""
        features = {}
        
        # Determine price column
        price_col = None
        for col in ['close', 'price', 'Close', 'Price']:
            if col in price_data.columns:
                price_col = col
                break
        
        if price_col is None and len(price_data.columns) > 0:
            price_col = price_data.columns[0]
        
        if price_col is None:
            return features
        
        prices = price_data[price_col].values
        
        # Returns
        returns = np.diff(prices) / prices[:-1]
        returns = np.concatenate([[0], returns])  # Pad with zero for first value
        
        # Log returns
        log_returns = np.diff(np.log(prices + 1e-8))
        log_returns = np.concatenate([[0], log_returns])
        
        # Price ratios
        price_ma20 = pd.Series(prices).rolling(20, min_periods=1).mean().values
        price_ma50 = pd.Series(prices).rolling(50, min_periods=1).mean().values
        
        features['prices'] = prices.astype(np.float32)
        features['returns'] = returns.astype(np.float32)
        features['log_returns'] = log_returns.astype(np.float32)
        features['price_ma20_ratio'] = (prices / price_ma20).astype(np.float32)
        features['price_ma50_ratio'] = (prices / price_ma50).astype(np.float32)
        
        return features
    
    def _extract_technical_indicators(self, price_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract technical indicators"""
        features = {}
        
        # Determine price column
        price_col = None
        for col in ['close', 'price', 'Close', 'Price']:
            if col in price_data.columns:
                price_col = col
                break
        
        if price_col is None:
            return features
        
        prices = pd.Series(price_data[price_col].values)
        
        # Simple Moving Averages
        if 'sma' in self.config.technical_indicators:
            features['sma_5'] = prices.rolling(5, min_periods=1).mean().values.astype(np.float32)
            features['sma_10'] = prices.rolling(10, min_periods=1).mean().values.astype(np.float32)
            features['sma_20'] = prices.rolling(20, min_periods=1).mean().values.astype(np.float32)
        
        # Exponential Moving Averages
        if 'ema' in self.config.technical_indicators:
            features['ema_5'] = prices.ewm(span=5).mean().values.astype(np.float32)
            features['ema_10'] = prices.ewm(span=10).mean().values.astype(np.float32)
            features['ema_20'] = prices.ewm(span=20).mean().values.astype(np.float32)
        
        # RSI
        if 'rsi' in self.config.technical_indicators:
            rsi = self._calculate_rsi(prices)
            features['rsi'] = rsi.astype(np.float32)
        
        # MACD
        if 'macd' in self.config.technical_indicators:
            macd_line, macd_signal, macd_histogram = self._calculate_macd(prices)
            features['macd_line'] = macd_line.astype(np.float32)
            features['macd_signal'] = macd_signal.astype(np.float32)
            features['macd_histogram'] = macd_histogram.astype(np.float32)
        
        # Bollinger Bands
        if 'bollinger' in self.config.technical_indicators:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices)
            features['bb_upper'] = bb_upper.astype(np.float32)
            features['bb_middle'] = bb_middle.astype(np.float32)
            features['bb_lower'] = bb_lower.astype(np.float32)
            features['bb_position'] = ((prices - bb_lower) / (bb_upper - bb_lower)).fillna(0.5).values.astype(np.float32)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50).values
        except:
            return np.full(len(prices), 50.0)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD indicator"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal).mean()
            macd_histogram = macd_line - macd_signal
            
            return (
                macd_line.fillna(0).values,
                macd_signal.fillna(0).values,
                macd_histogram.fillna(0).values
            )
        except:
            return (
                np.zeros(len(prices)),
                np.zeros(len(prices)),
                np.zeros(len(prices))
            )
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        try:
            middle = prices.rolling(window=period, min_periods=1).mean()
            std = prices.rolling(window=period, min_periods=1).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            return (
                upper.fillna(prices).values,
                middle.fillna(prices).values,
                lower.fillna(prices).values
            )
        except:
            return (
                prices.values,
                prices.values,
                prices.values
            )
    
    def _extract_time_features(self, price_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract time-based features"""
        features = {}
        
        if price_data.index.dtype.kind in ['M', 'datetime64']:
            timestamps = price_data.index
        elif 'timestamp' in price_data.columns:
            timestamps = pd.to_datetime(price_data['timestamp'])
        else:
            # Create dummy timestamps
            timestamps = pd.date_range('2020-01-01', periods=len(price_data), freq='H')
        
        if 'hour' in self.config.time_features:
            features['hour'] = timestamps.hour.values.astype(np.float32)
            features['hour_sin'] = np.sin(2 * np.pi * timestamps.hour / 24).astype(np.float32)
            features['hour_cos'] = np.cos(2 * np.pi * timestamps.hour / 24).astype(np.float32)
        
        if 'day_of_week' in self.config.time_features:
            features['day_of_week'] = timestamps.dayofweek.values.astype(np.float32)
            features['dow_sin'] = np.sin(2 * np.pi * timestamps.dayofweek / 7).astype(np.float32)
            features['dow_cos'] = np.cos(2 * np.pi * timestamps.dayofweek / 7).astype(np.float32)
        
        if 'month' in self.config.time_features:
            features['month'] = timestamps.month.values.astype(np.float32)
            features['month_sin'] = np.sin(2 * np.pi * timestamps.month / 12).astype(np.float32)
            features['month_cos'] = np.cos(2 * np.pi * timestamps.month / 12).astype(np.float32)
        
        return features
    
    def _extract_volume_features(self, volume_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract volume-based features"""
        features = {}
        
        # Determine volume column
        volume_col = None
        for col in ['volume', 'Volume', 'vol']:
            if col in volume_data.columns:
                volume_col = col
                break
        
        if volume_col is None and len(volume_data.columns) > 0:
            volume_col = volume_data.columns[0]
        
        if volume_col is None:
            return features
        
        volume = volume_data[volume_col].values
        
        # Volume moving averages
        volume_ma10 = pd.Series(volume).rolling(10, min_periods=1).mean().values
        volume_ma20 = pd.Series(volume).rolling(20, min_periods=1).mean().values
        
        features['volume'] = volume.astype(np.float32)
        features['volume_ma10'] = volume_ma10.astype(np.float32)
        features['volume_ma20'] = volume_ma20.astype(np.float32)
        features['volume_ratio'] = (volume / volume_ma20).astype(np.float32)
        
        # Price-volume features
        if len(price_data) == len(volume_data):
            price_col = None
            for col in ['close', 'price', 'Close', 'Price']:
                if col in price_data.columns:
                    price_col = col
                    break
            
            if price_col is not None:
                prices = price_data[price_col].values
                returns = np.diff(prices) / prices[:-1]
                returns = np.concatenate([[0], returns])
                
                # Volume-weighted price features
                features['vwap'] = (prices * volume).astype(np.float32)
                features['price_volume_correlation'] = pd.Series(returns).rolling(20, min_periods=1).corr(
                    pd.Series(volume)
                ).fillna(0).values.astype(np.float32)
        
        return features
    
    def _extract_rolling_statistics(self, price_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract rolling statistical features"""
        features = {}
        
        price_col = None
        for col in ['close', 'price', 'Close', 'Price']:
            if col in price_data.columns:
                price_col = col
                break
        
        if price_col is None:
            return features
        
        prices = pd.Series(price_data[price_col].values)
        
        # Rolling statistics
        windows = [5, 10, 20]
        
        for window in windows:
            # Rolling mean
            features[f'rolling_mean_{window}'] = prices.rolling(window, min_periods=1).mean().values.astype(np.float32)
            
            # Rolling std
            features[f'rolling_std_{window}'] = prices.rolling(window, min_periods=1).std().fillna(0).values.astype(np.float32)
            
            # Rolling min/max
            features[f'rolling_min_{window}'] = prices.rolling(window, min_periods=1).min().values.astype(np.float32)
            features[f'rolling_max_{window}'] = prices.rolling(window, min_periods=1).max().values.astype(np.float32)
            
            # Rolling quantiles
            features[f'rolling_q25_{window}'] = prices.rolling(window, min_periods=1).quantile(0.25).values.astype(np.float32)
            features[f'rolling_q75_{window}'] = prices.rolling(window, min_periods=1).quantile(0.75).values.astype(np.float32)
        
        return features
    
    def _extract_volatility_features(self, price_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract volatility features"""
        features = {}
        
        price_col = None
        for col in ['close', 'price', 'Close', 'Price']:
            if col in price_data.columns:
                price_col = col
                break
        
        if price_col is None:
            return features
        
        prices = pd.Series(price_data[price_col].values)
        returns = prices.pct_change().fillna(0)
        
        # Historical volatility
        windows = [10, 20, 30]
        for window in windows:
            vol = returns.rolling(window, min_periods=1).std() * np.sqrt(252)  # Annualized
            features[f'volatility_{window}'] = vol.fillna(0).values.astype(np.float32)
        
        # GARCH-like volatility
        squared_returns = returns ** 2
        features['garch_volatility'] = squared_returns.ewm(span=20).mean().values.astype(np.float32)
        
        # Parkinson volatility (if high/low available)
        if all(col in price_data.columns for col in ['high', 'low']):
            high = price_data['high'].values
            low = price_data['low'].values
            parkinson_vol = np.sqrt(np.log(high / low) ** 2 / (4 * np.log(2)))
            features['parkinson_volatility'] = parkinson_vol.astype(np.float32)
        
        return features

class GraphProcessor:
    """Graph processing for network-based features"""
    
    def __init__(self, config: MultimodalConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.GraphProcessor")
    
    def process_graph_data(self, graph_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Process graph/network data into features
        
        Args:
            graph_data: Graph data dictionary
            
        Returns:
            Dictionary of processed graph features
        """
        try:
            if not HAS_NETWORKX:
                self.logger.warning("NetworkX not available - graph processing disabled")
                return {}
            
            processed_features = {}
            
            # Create network graph
            G = self._create_graph_from_data(graph_data)
            
            if G is None or len(G.nodes()) == 0:
                return {}
            
            # Node-level features
            node_features = self._extract_node_features(G)
            processed_features.update(node_features)
            
            # Graph-level features
            graph_features = self._extract_graph_features(G)
            processed_features.update(graph_features)
            
            return processed_features
            
        except Exception as e:
            self.logger.error(f"Graph processing failed: {e}")
            return {}
    
    def _create_graph_from_data(self, graph_data: Dict[str, Any]) -> Optional[nx.Graph]:
        """Create NetworkX graph from data"""
        try:
            G = nx.Graph()
            
            # Add nodes
            if 'nodes' in graph_data:
                for node_data in graph_data['nodes']:
                    if isinstance(node_data, dict):
                        node_id = node_data.get('id', len(G.nodes()))
                        G.add_node(node_id, **node_data)
                    else:
                        G.add_node(node_data)
            
            # Add edges
            if 'edges' in graph_data:
                for edge_data in graph_data['edges']:
                    if isinstance(edge_data, dict):
                        source = edge_data.get('source')
                        target = edge_data.get('target')
                        if source is not None and target is not None:
                            G.add_edge(source, target, **edge_data)
                    elif len(edge_data) >= 2:
                        G.add_edge(edge_data[0], edge_data[1])
            
            return G
            
        except Exception as e:
            self.logger.error(f"Graph creation failed: {e}")
            return None
    
    def _extract_node_features(self, G: nx.Graph) -> Dict[str, np.ndarray]:
        """Extract node-level features"""
        features = {}
        
        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(G)
            features['node_degree_centrality'] = np.array([
                degree_centrality.get(node, 0) for node in G.nodes()
            ], dtype=np.float32)
            
            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(G)
            features['node_betweenness_centrality'] = np.array([
                betweenness_centrality.get(node, 0) for node in G.nodes()
            ], dtype=np.float32)
            
            # Closeness centrality
            closeness_centrality = nx.closeness_centrality(G)
            features['node_closeness_centrality'] = np.array([
                closeness_centrality.get(node, 0) for node in G.nodes()
            ], dtype=np.float32)
            
            # Clustering coefficient
            clustering = nx.clustering(G)
            features['node_clustering'] = np.array([
                clustering.get(node, 0) for node in G.nodes()
            ], dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Node feature extraction failed: {e}")
        
        return features
    
    def _extract_graph_features(self, G: nx.Graph) -> Dict[str, np.ndarray]:
        """Extract graph-level features"""
        features = {}
        
        try:
            # Basic graph properties
            features['graph_num_nodes'] = np.array([len(G.nodes())], dtype=np.float32)
            features['graph_num_edges'] = np.array([len(G.edges())], dtype=np.float32)
            features['graph_density'] = np.array([nx.density(G)], dtype=np.float32)
            
            # Connectivity measures
            if nx.is_connected(G):
                features['graph_diameter'] = np.array([nx.diameter(G)], dtype=np.float32)
                features['graph_avg_shortest_path'] = np.array([nx.average_shortest_path_length(G)], dtype=np.float32)
            else:
                features['graph_diameter'] = np.array([0.0], dtype=np.float32)
                features['graph_avg_shortest_path'] = np.array([0.0], dtype=np.float32)
            
            # Clustering
            features['graph_avg_clustering'] = np.array([nx.average_clustering(G)], dtype=np.float32)
            features['graph_transitivity'] = np.array([nx.transitivity(G)], dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Graph feature extraction failed: {e}")
        
        return features

class MultimodalProcessor:
    """Main multimodal data processor"""
    
    def __init__(self, config: Optional[MultimodalConfig] = None):
        self.config = config or MultimodalConfig()
        self.logger = logging.getLogger(f"{__name__}.MultimodalProcessor")
        
        # Initialize processors
        self.text_processor = TextProcessor(self.config)
        self.timeseries_processor = TimeSeriesProcessor(self.config)
        self.graph_processor = GraphProcessor(self.config)
        
        # Feature scalers
        self.feature_scalers = {}
        
        self._lock = threading.RLock()
        
        self.logger.info("Multimodal processor initialized")
    
    def process_all_modalities(self, data_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Process all available data modalities
        
        Args:
            data_dict: Dictionary containing different types of data
            
        Returns:
            Dictionary of processed features for all modalities
        """
        with self._lock:
            all_features = {}
            
            # Process text data
            if 'text' in data_dict or 'sentiment' in data_dict:
                text_data = data_dict.get('text', data_dict.get('sentiment', []))
                if text_data:
                    text_features = self.text_processor.process_text_data(text_data)
                    all_features.update({f"text_{k}": v for k, v in text_features.items()})
            
            # Process time series data
            if 'price_data' in data_dict:
                price_data = data_dict['price_data']
                volume_data = data_dict.get('volume_data')
                
                ts_features = self.timeseries_processor.process_timeseries_data(price_data, volume_data)
                all_features.update({f"ts_{k}": v for k, v in ts_features.items()})
            
            # Process graph data
            if 'graph_data' in data_dict:
                graph_data = data_dict['graph_data']
                graph_features = self.graph_processor.process_graph_data(graph_data)
                all_features.update({f"graph_{k}": v for k, v in graph_features.items()})
            
            # Normalize features if requested
            if self.config.normalize_features:
                all_features = self._normalize_features(all_features)
            
            return all_features
    
    def _normalize_features(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize features using configured method"""
        try:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            
            scaler_class = {
                'standard': StandardScaler,
                'minmax': MinMaxScaler,
                'robust': RobustScaler
            }.get(self.config.feature_scaling_method, StandardScaler)
            
            normalized_features = {}
            
            for feature_name, feature_values in features.items():
                if feature_name not in self.feature_scalers:
                    self.feature_scalers[feature_name] = scaler_class()
                
                # Reshape for sklearn
                values_reshaped = feature_values.reshape(-1, 1)
                
                # Fit and transform
                normalized_values = self.feature_scalers[feature_name].fit_transform(values_reshaped)
                normalized_features[feature_name] = normalized_values.flatten().astype(np.float32)
            
            return normalized_features
            
        except Exception as e:
            self.logger.error(f"Feature normalization failed: {e}")
            return features
    
    def get_processor_summary(self) -> Dict[str, Any]:
        """Get summary of processor capabilities"""
        return {
            'text_processing': {
                'transformers_available': HAS_TRANSFORMERS,
                'textblob_available': HAS_TEXTBLOB,
                'model_name': self.config.text_model_name if HAS_TRANSFORMERS else None
            },
            'timeseries_processing': {
                'technical_indicators': self.config.technical_indicators,
                'time_features': self.config.time_features,
                'sequence_length': self.config.sequence_length
            },
            'graph_processing': {
                'networkx_available': HAS_NETWORKX,
                'max_nodes': self.config.max_graph_nodes
            },
            'feature_scaling': {
                'enabled': self.config.normalize_features,
                'method': self.config.feature_scaling_method,
                'fitted_scalers': len(self.feature_scalers)
            }
        }


# Singleton multimodal processor
_multimodal_processor = None
_processor_lock = threading.Lock()

def get_multimodal_processor(config: Optional[MultimodalConfig] = None) -> MultimodalProcessor:
    """Get the singleton multimodal processor"""
    global _multimodal_processor
    
    with _processor_lock:
        if _multimodal_processor is None:
            _multimodal_processor = MultimodalProcessor(config)
        return _multimodal_processor

def process_multimodal_data(data_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convenient function to process multimodal data"""
    processor = get_multimodal_processor()
    return processor.process_all_modalities(data_dict)