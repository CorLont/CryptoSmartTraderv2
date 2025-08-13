"""
Cross-Coin Feature Fusion Engine
Advanced multi-coin correlation and feature fusion for alpha detection
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import json

class CrossCoinFusionEngine:
    """
    Advanced cross-coin feature fusion for identifying market relationships
    and correlation-based alpha opportunities
    """

    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)

        # Correlation matrices for different feature types
        self.correlation_matrices = {
            'price': None,
            'volume': None,
            'sentiment': None,
            'whale': None,
            'technical': None
        }

        # Market structure analysis
        self.market_clusters = {}
        self.correlation_network = None

        # Feature fusion parameters
        self.fusion_config = {
            'correlation_threshold': 0.7,
            'cluster_count': 10,
            'fusion_window': 24,  # hours
            'min_coins_per_cluster': 5
        }

        # Cache for fusion results
        self.fusion_cache = {}

        self.logger.info("Cross-Coin Feature Fusion Engine initialized")

    async def analyze_cross_coin_correlations(self, coins_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlations across all coins for different feature types"""

        if len(coins_data) < 10:
            self.logger.warning("Insufficient coins for cross-coin analysis")
            return {}

        try:
            # Build correlation matrices for each feature type
            correlation_results = {}

            # Price correlation analysis
            price_corr = await self._analyze_price_correlations(coins_data)
            correlation_results['price_correlations'] = price_corr

            # Volume correlation analysis
            volume_corr = await self._analyze_volume_correlations(coins_data)
            correlation_results['volume_correlations'] = volume_corr

            # Sentiment correlation analysis
            sentiment_corr = await self._analyze_sentiment_correlations(coins_data)
            correlation_results['sentiment_correlations'] = sentiment_corr

            # Cross-feature fusion
            fusion_features = await self._create_fusion_features(coins_data)
            correlation_results['fusion_features'] = fusion_features

            # Market structure analysis
            market_structure = await self._analyze_market_structure(coins_data)
            correlation_results['market_structure'] = market_structure

            # Alpha opportunity detection via correlation
            alpha_opportunities = await self._detect_correlation_alpha(correlation_results)
            correlation_results['alpha_opportunities'] = alpha_opportunities

            self.logger.info(f"Cross-coin analysis completed for {len(coins_data)} coins")

            return correlation_results

        except Exception as e:
            self.logger.error(f"Cross-coin correlation analysis failed: {e}")
            return {}

    async def _analyze_price_correlations(self, coins_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze price movement correlations between coins"""

        try:
            # Extract price data
            price_matrix = []
            coin_symbols = []

            for coin_data in coins_data:
                symbol = coin_data.get('symbol', 'unknown')

                # Get price history
                price_history = coin_data.get('price_history', [])
                if len(price_history) >= 24:  # Minimum data points
                    # Calculate returns
                    prices = [p.get('close', 0) for p in price_history[-24:]]
                    returns = np.diff(prices) / prices[:-1]

                    price_matrix.append(returns)
                    coin_symbols.append(symbol)

            if len(price_matrix) < 5:
                return {'error': 'Insufficient price data for correlation analysis'}

            # Convert to matrix
            price_matrix = np.array(price_matrix)

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(price_matrix)

            # Find high correlation pairs
            high_correlations = []
            strong_correlations = []

            for i in range(len(coin_symbols)):
                for j in range(i + 1, len(coin_symbols)):
                    corr_value = correlation_matrix[i, j]

                    if abs(corr_value) > self.fusion_config['correlation_threshold']:
                        correlation_pair = {
                            'coin1': coin_symbols[i],
                            'coin2': coin_symbols[j],
                            'correlation': float(corr_value),
                            'type': 'positive' if corr_value > 0 else 'negative',
                            'strength': 'very_strong' if abs(corr_value) > 0.9 else 'strong'
                        }

                        high_correlations.append(correlation_pair)

                        if abs(corr_value) > 0.8:
                            strong_correlations.append(correlation_pair)

            # Store correlation matrix
            self.correlation_matrices['price'] = {
                'matrix': correlation_matrix,
                'symbols': coin_symbols
            }

            return {
                'correlation_matrix_shape': correlation_matrix.shape,
                'high_correlations': high_correlations[:20],  # Top 20
                'strong_correlations': strong_correlations,
                'average_correlation': float(np.mean(np.abs(correlation_matrix))),
                'max_correlation': float(np.max(np.abs(correlation_matrix[correlation_matrix < 1]))),
                'correlation_clusters': self._identify_correlation_clusters(correlation_matrix, coin_symbols)
            }

        except Exception as e:
            self.logger.error(f"Price correlation analysis failed: {e}")
            return {'error': str(e)}

    async def _analyze_volume_correlations(self, coins_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze volume correlations between coins"""

        try:
            volume_matrix = []
            coin_symbols = []

            for coin_data in coins_data:
                symbol = coin_data.get('symbol', 'unknown')

                # Get volume history
                price_history = coin_data.get('price_history', [])
                if len(price_history) >= 24:
                    volumes = [p.get('volume', 0) for p in price_history[-24:]]

                    # Normalize volumes (log transform to handle large differences)
                    log_volumes = np.log1p(volumes)  # log(1 + volume)

                    volume_matrix.append(log_volumes)
                    coin_symbols.append(symbol)

            if len(volume_matrix) < 5:
                return {'error': 'Insufficient volume data'}

            # Calculate volume correlation
            volume_matrix = np.array(volume_matrix)
            correlation_matrix = np.corrcoef(volume_matrix)

            # Find volume surge correlations
            volume_correlations = []

            for i in range(len(coin_symbols)):
                for j in range(i + 1, len(coin_symbols)):
                    corr_value = correlation_matrix[i, j]

                    if corr_value > 0.6:  # Lower threshold for volume
                        volume_correlations.append({
                            'coin1': coin_symbols[i],
                            'coin2': coin_symbols[j],
                            'volume_correlation': float(corr_value),
                            'implication': 'coordinated_interest' if corr_value > 0.8 else 'related_interest'
                        })

            return {
                'volume_correlations': volume_correlations[:15],
                'average_volume_correlation': float(np.mean(correlation_matrix)),
                'coordinated_volume_pairs': len([c for c in volume_correlations if c['volume_correlation'] > 0.8])
            }

        except Exception as e:
            self.logger.error(f"Volume correlation analysis failed: {e}")
            return {'error': str(e)}

    async def _analyze_sentiment_correlations(self, coins_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment correlations between coins"""

        try:
            sentiment_matrix = []
            coin_symbols = []

            for coin_data in coins_data:
                symbol = coin_data.get('symbol', 'unknown')
                sentiment_data = coin_data.get('sentiment_data', {})

                if sentiment_data.get('sentiment_score') is not None:
                    # Get sentiment features
                    sentiment_features = [
                        sentiment_data.get('sentiment_score', 0.5),
                        sentiment_data.get('mention_volume', 0) / 1000,  # Normalize
                        sentiment_data.get('sentiment_trend', 0),
                        sentiment_data.get('confidence', 0.5)
                    ]

                    sentiment_matrix.append(sentiment_features)
                    coin_symbols.append(symbol)

            if len(sentiment_matrix) < 5:
                return {'error': 'Insufficient sentiment data'}

            # Calculate sentiment correlation
            sentiment_matrix = np.array(sentiment_matrix)
            correlation_matrix = np.corrcoef(sentiment_matrix)

            # Find sentiment trend correlations
            sentiment_correlations = []

            for i in range(len(coin_symbols)):
                for j in range(i + 1, len(coin_symbols)):
                    corr_value = correlation_matrix[i, j]

                    if corr_value > 0.5:
                        sentiment_correlations.append({
                            'coin1': coin_symbols[i],
                            'coin2': coin_symbols[j],
                            'sentiment_correlation': float(corr_value),
                            'market_narrative': self._identify_market_narrative(coin_symbols[i], coin_symbols[j])
                        })

            return {
                'sentiment_correlations': sentiment_correlations[:10],
                'market_sentiment_clusters': self._cluster_by_sentiment(sentiment_matrix, coin_symbols)
            }

        except Exception as e:
            self.logger.error(f"Sentiment correlation analysis failed: {e}")
            return {'error': str(e)}

    async def _create_fusion_features(self, coins_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create cross-coin fusion features for enhanced prediction"""

        try:
            fusion_features = {}

            # Market-wide features
            market_features = self._calculate_market_features(coins_data)
            fusion_features['market_features'] = market_features

            # Sector-based features
            sector_features = self._calculate_sector_features(coins_data)
            fusion_features['sector_features'] = sector_features

            # Correlation-based features
            correlation_features = self._calculate_correlation_features(coins_data)
            fusion_features['correlation_features'] = correlation_features

            # Network-based features
            network_features = self._calculate_network_features(coins_data)
            fusion_features['network_features'] = network_features

            return fusion_features

        except Exception as e:
            self.logger.error(f"Fusion features creation failed: {e}")
            return {}

    def _calculate_market_features(self, coins_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate market-wide aggregate features"""

        try:
            # Aggregate price movements
            price_changes = []
            volumes = []
            market_caps = []

            for coin_data in coins_data:
                price_history = coin_data.get('price_history', [])
                if len(price_history) >= 2:
                    recent_price = price_history[-1].get('close', 0)
                    previous_price = price_history[-2].get('close', 0)

                    if previous_price > 0:
                        price_change = (recent_price - previous_price) / previous_price
                        price_changes.append(price_change)

                    volumes.append(price_history[-1].get('volume', 0))

                    # Estimate market cap (price * volume as proxy)
                    market_caps.append(recent_price * price_history[-1].get('volume', 0))

            market_features = {
                'market_momentum': float(np.mean(price_changes)) if price_changes else 0,
                'market_volatility': float(np.std(price_changes)) if price_changes else 0,
                'total_volume': float(np.sum(volumes)),
                'market_cap_concentration': float(np.std(market_caps) / np.mean(market_caps)) if market_caps else 0,
                'bullish_coins_ratio': float(len([p for p in price_changes if p > 0]) / len(price_changes)) if price_changes else 0,
                'extreme_movers': len([p for p in price_changes if abs(p) > 0.1])
            }

            return market_features

        except Exception as e:
            self.logger.error(f"Market features calculation failed: {e}")
            return {}

    def _calculate_sector_features(self, coins_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate sector-based aggregated features"""

        try:
            # Simple sector classification based on coin names/symbols
            sectors = {
                'defi': [],
                'layer1': [],
                'meme': [],
                'ai': [],
                'gaming': [],
                'other': []
            }

            for coin_data in coins_data:
                symbol = coin_data.get('symbol', '').lower()
                sector = self._classify_coin_sector(symbol)
                sectors[sector].append(coin_data)

            sector_features = {}

            for sector_name, sector_coins in sectors.items():
                if len(sector_coins) >= 3:  # Minimum coins for sector analysis
                    sector_momentum = self._calculate_sector_momentum(sector_coins)
                    sector_features[f'{sector_name}_momentum'] = sector_momentum
                    sector_features[f'{sector_name}_coin_count'] = len(sector_coins)

            return sector_features

        except Exception as e:
            self.logger.error(f"Sector features calculation failed: {e}")
            return {}

    def _classify_coin_sector(self, symbol: str) -> str:
        """Simple sector classification based on symbol"""

        defi_indicators = ['uni', 'sushi', 'cake', 'curve', 'aave', 'comp', 'mkr', 'snx']
        layer1_indicators = ['eth', 'btc', 'ada', 'sol', 'dot', 'avax', 'near', 'atom']
        meme_indicators = ['doge', 'shib', 'pepe', 'floki', 'baby', 'meme', 'bonk']
        ai_indicators = ['ai', 'fet', 'agix', 'ocean', 'rndr']
        gaming_indicators = ['axs', 'sand', 'mana', 'gala', 'enj', 'chr']

        symbol_lower = symbol.lower()

        if any(indicator in symbol_lower for indicator in defi_indicators):
            return 'defi'
        elif any(indicator in symbol_lower for indicator in layer1_indicators):
            return 'layer1'
        elif any(indicator in symbol_lower for indicator in meme_indicators):
            return 'meme'
        elif any(indicator in symbol_lower for indicator in ai_indicators):
            return 'ai'
        elif any(indicator in symbol_lower for indicator in gaming_indicators):
            return 'gaming'
        else:
            return 'other'

    def _calculate_sector_momentum(self, sector_coins: List[Dict[str, Any]]) -> float:
        """Calculate momentum for a specific sector"""

        try:
            sector_changes = []

            for coin_data in sector_coins:
                price_history = coin_data.get('price_history', [])
                if len(price_history) >= 2:
                    recent = price_history[-1].get('close', 0)
                    previous = price_history[-2].get('close', 0)

                    if previous > 0:
                        change = (recent - previous) / previous
                        sector_changes.append(change)

            if sector_changes:
                return float(np.mean(sector_changes))
            else:
                return 0.0

        except Exception:
            return 0.0

    def _calculate_correlation_features(self, coins_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate correlation-based fusion features"""

        try:
            correlation_features = {}

            if self.correlation_matrices['price'] is not None:
                corr_matrix = self.correlation_matrices['price']['matrix']
                symbols = self.correlation_matrices['price']['symbols']

                # Find each coin's correlation with market
                for i, symbol in enumerate(symbols):
                    # Correlation with all other coins (market correlation)
                    other_correlations = np.concatenate([corr_matrix[i, :i], corr_matrix[i, i+1:]])

                    correlation_features[f'{symbol}_market_correlation'] = float(np.mean(other_correlations))
                    correlation_features[f'{symbol}_correlation_volatility'] = float(np.std(other_correlations))
                    correlation_features[f'{symbol}_max_correlation'] = float(np.max(np.abs(other_correlations)))

            return correlation_features

        except Exception as e:
            self.logger.error(f"Correlation features calculation failed: {e}")
            return {}

    def _calculate_network_features(self, coins_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate network-based features using correlation network"""

        try:
            if self.correlation_matrices['price'] is None:
                return {}

            corr_matrix = self.correlation_matrices['price']['matrix']
            symbols = self.correlation_matrices['price']['symbols']

            # Build correlation network
            G = nx.Graph()

            # Add nodes
            for symbol in symbols:
                G.add_node(symbol)

            # Add edges for strong correlations
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    correlation = corr_matrix[i, j]
                    if abs(correlation) > 0.6:  # Strong correlation threshold
                        G.add_edge(symbols[i], symbols[j], weight=abs(correlation))

            # Calculate network features
            network_features = {
                'network_density': nx.density(G),
                'average_clustering': nx.average_clustering(G),
                'number_of_components': nx.number_connected_components(G)
            }

            # Individual node features
            centrality = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G)

            for symbol in symbols:
                network_features[f'{symbol}_centrality'] = centrality.get(symbol, 0)
                network_features[f'{symbol}_betweenness'] = betweenness.get(symbol, 0)

            self.correlation_network = G

            return network_features

        except Exception as e:
            self.logger.error(f"Network features calculation failed: {e}")
            return {}

    def _identify_correlation_clusters(self, correlation_matrix: np.ndarray, symbols: List[str]) -> List[Dict[str, Any]]:
        """Identify clusters of highly correlated coins"""

        try:
            # Use KMeans clustering on correlation matrix
            n_clusters = min(self.fusion_config['cluster_count'], len(symbols) // 2)

            if n_clusters < 2:
                return []

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(correlation_matrix)

            # Group coins by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(symbols[i])

            # Format cluster results
            cluster_results = []
            for cluster_id, cluster_coins in clusters.items():
                if len(cluster_coins) >= self.fusion_config['min_coins_per_cluster']:
                    # Calculate intra-cluster correlation
                    cluster_indices = [symbols.index(coin) for coin in cluster_coins]
                    sub_matrix = correlation_matrix[np.ix_(cluster_indices, cluster_indices)]
                    avg_correlation = np.mean(sub_matrix[np.triu_indices_from(sub_matrix, k=1)])

                    cluster_results.append({
                        'cluster_id': int(cluster_id),
                        'coins': cluster_coins,
                        'average_correlation': float(avg_correlation),
                        'cluster_size': len(cluster_coins)
                    })

            return cluster_results

        except Exception as e:
            self.logger.error(f"Correlation clustering failed: {e}")
            return []

    async def _analyze_market_structure(self, coins_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall market structure and relationships"""

        try:
            market_structure = {
                'total_coins_analyzed': len(coins_data),
                'timestamp': datetime.now().isoformat()
            }

            # Market concentration analysis
            market_caps = []
            for coin_data in coins_data:
                price_history = coin_data.get('price_history', [])
                if price_history:
                    latest = price_history[-1]
                    estimated_market_cap = latest.get('close', 0) * latest.get('volume', 0)
                    market_caps.append(estimated_market_cap)

            if market_caps:
                market_caps = np.array(market_caps)
                total_market_cap = np.sum(market_caps)

                # Calculate concentration
                sorted_caps = np.sort(market_caps)[::-1]
                top_10_concentration = np.sum(sorted_caps[:10]) / total_market_cap if total_market_cap > 0 else 0

                market_structure.update({
                    'market_concentration_top10': float(top_10_concentration),
                    'gini_coefficient': self._calculate_gini_coefficient(market_caps),
                    'market_cap_distribution': {
                        'mean': float(np.mean(market_caps)),
                        'std': float(np.std(market_caps)),
                        'median': float(np.median(market_caps))
                    }
                })

            # Correlation structure analysis
            if self.correlation_matrices['price'] is not None:
                corr_matrix = self.correlation_matrices['price']['matrix']

                # Analyze correlation distribution
                upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

                market_structure.update({
                    'correlation_structure': {
                        'mean_correlation': float(np.mean(upper_triangle)),
                        'correlation_std': float(np.std(upper_triangle)),
                        'high_correlation_pairs': int(np.sum(np.abs(upper_triangle) > 0.7)),
                        'negative_correlation_pairs': int(np.sum(upper_triangle < -0.3))
                    }
                })

            return market_structure

        except Exception as e:
            self.logger.error(f"Market structure analysis failed: {e}")
            return {}

    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for market concentration"""

        try:
            values = np.sort(values)
            n = len(values)
            cumsum = np.cumsum(values)

            return float((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n)

        except Exception:
            return 0.0

    async def _detect_correlation_alpha(self, correlation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect alpha opportunities based on correlation analysis"""

        try:
            alpha_opportunities = []

            # Look for correlation divergence opportunities
            price_correlations = correlation_results.get('price_correlations', {}).get('high_correlations', [])

            for correlation_pair in price_correlations:
                coin1 = correlation_pair['coin1']
                coin2 = correlation_pair['coin2']
                correlation_strength = correlation_pair['correlation']

                # Look for temporary divergence in highly correlated pairs
                # This suggests potential reversion opportunity

                alpha_opportunity = {
                    'type': 'correlation_divergence',
                    'primary_coin': coin1,
                    'reference_coin': coin2,
                    'correlation_strength': correlation_strength,
                    'opportunity_reason': f'High correlation ({correlation_strength:.3f}) suggests potential reversion',
                    'confidence': min(0.8, abs(correlation_strength)),
                    'strategy': 'mean_reversion' if correlation_strength > 0 else 'divergence_play'
                }

                alpha_opportunities.append(alpha_opportunity)

            # Look for sector momentum opportunities
            sector_features = correlation_results.get('fusion_features', {}).get('sector_features', {})

            for feature_name, value in sector_features.items():
                if 'momentum' in feature_name and abs(value) > 0.05:  # 5% sector momentum
                    sector_name = feature_name.replace('_momentum', '')

                    alpha_opportunity = {
                        'type': 'sector_momentum',
                        'sector': sector_name,
                        'momentum_value': value,
                        'opportunity_reason': f'Strong sector momentum: {value:.3f}',
                        'confidence': min(0.9, abs(value) * 10),  # Scale momentum to confidence
                        'strategy': 'momentum_follow' if value > 0 else 'contrarian'
                    }

                    alpha_opportunities.append(alpha_opportunity)

            # Sort by confidence
            alpha_opportunities.sort(key=lambda x: x['confidence'], reverse=True)

            return alpha_opportunities[:10]  # Top 10 opportunities

        except Exception as e:
            self.logger.error(f"Correlation alpha detection failed: {e}")
            return []

    def _cluster_by_sentiment(self, sentiment_matrix: np.ndarray, symbols: List[str]) -> List[Dict[str, Any]]:
        """Cluster coins by sentiment patterns"""

        try:
            if len(sentiment_matrix) < 5:
                return []

            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(sentiment_matrix)

            # Cluster
            n_clusters = min(5, len(symbols) // 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_features)

            # Group by clusters
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    'symbol': symbols[i],
                    'sentiment_features': sentiment_matrix[i].tolist()
                })

            # Format results
            cluster_results = []
            for cluster_id, cluster_data in clusters.items():
                if len(cluster_data) >= 3:
                    cluster_results.append({
                        'cluster_id': int(cluster_id),
                        'coins': [item['symbol'] for item in cluster_data],
                        'cluster_size': len(cluster_data),
                        'sentiment_pattern': 'similar_sentiment_behavior'
                    })

            return cluster_results

        except Exception as e:
            self.logger.error(f"Sentiment clustering failed: {e}")
            return []

    def _identify_market_narrative(self, coin1: str, coin2: str) -> str:
        """Identify potential market narrative for correlated sentiment"""

        # Simple narrative identification based on coin types
        sectors = {
            coin1: self._classify_coin_sector(coin1),
            coin2: self._classify_coin_sector(coin2)
        }

        if sectors[coin1] == sectors[coin2]:
            return f"sector_narrative_{sectors[coin1]}"
        else:
            return "cross_sector_correlation"

    def get_fusion_status(self) -> Dict[str, Any]:
        """Get current status of cross-coin fusion analysis"""

        return {
            'correlation_matrices_built': sum(1 for matrix in self.correlation_matrices.values() if matrix is not None),
            'correlation_network_nodes': self.correlation_network.number_of_nodes() if self.correlation_network else 0,
            'correlation_network_edges': self.correlation_network.number_of_edges() if self.correlation_network else 0,
            'market_clusters_identified': len(self.market_clusters),
            'fusion_cache_size': len(self.fusion_cache),
            'last_analysis_time': datetime.now().isoformat()
        }
