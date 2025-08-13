#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Feature Discovery Engine
Continuous feature discovery and optimization with live regime adaptation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import warnings
import time
from collections import defaultdict, deque
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.feature_selection import SelectKBest, f_regression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .automated_feature_engineering import get_automated_feature_engineer, FeatureEngineeringConfig

class DiscoveryMode(Enum):
    EXPLORATORY = "exploratory"          # Wide exploration of feature space
    EXPLOITATIVE = "exploitative"        # Focus on refining known good features
    REGIME_ADAPTIVE = "regime_adaptive"  # Adapt to market regime changes
    PERFORMANCE_DRIVEN = "performance_driven"  # Focus on performance improvement

class FeatureStatus(Enum):
    CANDIDATE = "candidate"      # New feature being evaluated
    ACTIVE = "active"           # Feature currently in use
    DEPRECATED = "deprecated"   # Feature no longer useful
    TESTING = "testing"         # Feature under A/B testing
    CHAMPION = "champion"       # Best performing feature in category

@dataclass
class FeatureCandidate:
    """Candidate feature for evaluation"""
    name: str
    feature_data: pd.Series
    creation_method: str
    source_features: List[str]
    performance_score: float = 0.0
    regime_performance: Dict[str, float] = field(default_factory=dict)
    status: FeatureStatus = FeatureStatus.CANDIDATE
    created_at: datetime = field(default_factory=datetime.now)
    last_evaluated: Optional[datetime] = None
    usage_count: int = 0
    stability_score: float = 0.0

@dataclass
class DiscoveryConfig:
    """Configuration for feature discovery engine"""
    # Discovery parameters
    max_candidates_per_iteration: int = 20
    max_active_features: int = 200
    evaluation_window_size: int = 100
    min_performance_improvement: float = 0.02

    # Regime adaptation
    regime_switch_detection_window: int = 50
    regime_adaptation_threshold: float = 0.1
    feature_regime_memory_size: int = 1000

    # Performance tracking
    performance_history_size: int = 500
    stability_evaluation_periods: int = 10
    min_stability_score: float = 0.6

    # Discovery strategies
    exploration_ratio: float = 0.3  # 30% exploration, 70% exploitation
    feature_interaction_depth: int = 3
    genetic_algorithm_enabled: bool = True

    # Online learning
    online_adaptation_rate: float = 0.1
    feature_decay_rate: float = 0.02
    champion_protection_threshold: float = 0.8

@dataclass
class RegimeState:
    """Current market regime state"""
    regime_id: str
    confidence: float
    duration: int  # Periods in this regime
    active_features: List[str]
    performance_scores: Dict[str, float]
    last_updated: datetime = field(default_factory=datetime.now)

class GeneticFeatureEvolver:
    """Genetic algorithm for feature evolution"""

    def __init__(self, config: DiscoveryConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.GeneticFeatureEvolver")

        # Genetic algorithm parameters
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.elite_size = 10

    def evolve_features(self, parent_features: List[FeatureCandidate],
                       performance_scores: Dict[str, float]) -> List[FeatureCandidate]:
        """Evolve features using genetic algorithm"""
        try:
            if not parent_features:
                return []

            # Select elite features (best performers)
            elite_features = self._select_elite(parent_features, performance_scores)

            # Generate offspring through crossover and mutation
            offspring = []

            for _ in range(self.population_size - len(elite_features)):
                # Select parents
                parent1, parent2 = self._select_parents(parent_features, performance_scores)

                # Crossover
                if np.random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1

                # Mutation
                if np.random.random() < self.mutation_rate:
                    child = self._mutate(child)

                if child:
                    offspring.append(child)

            # Combine elite and offspring
            next_generation = elite_features + offspring

            return next_generation[:self.config.max_candidates_per_iteration]

        except Exception as e:
            self.logger.error(f"Feature evolution failed: {e}")
            return parent_features

    def _select_elite(self, features: List[FeatureCandidate],
                     performance_scores: Dict[str, float]) -> List[FeatureCandidate]:
        """Select elite features based on performance"""
        try:
            # Sort by performance score
            scored_features = []
            for feature in features:
                score = performance_scores.get(feature.name, feature.performance_score)
                scored_features.append((feature, score))

            scored_features.sort(key=lambda x: x[1], reverse=True)

            return [feature for feature, _ in scored_features[:self.elite_size]]

        except Exception:
            return features[:self.elite_size]

    def _select_parents(self, features: List[FeatureCandidate],
                       performance_scores: Dict[str, float]) -> Tuple[FeatureCandidate, FeatureCandidate]:
        """Select parents using tournament selection"""
        try:
            def tournament_select(tournament_size: int = 3) -> FeatureCandidate:
                tournament = np.random.choice(range(100), size=10, replace=False)
                best = max(tournament, key=lambda f: performance_scores.get(f.name, f.performance_score))
                return best

            return tournament_select(), tournament_select()

        except Exception:
            return np.random.normal(0, 1)

    def _crossover(self, parent1: FeatureCandidate, parent2: FeatureCandidate) -> Optional[FeatureCandidate]:
        """Create offspring by combining parent features"""
        try:
            # Combine source features from both parents
            combined_sources = list(set(parent1.source_features + parent2.source_features))

            # Create new feature name
            child_name = f"crossover_{parent1.name}_{parent2.name}_{int(time.time())}"

            # Simple crossover: average the feature values if they have the same index
            if len(parent1.feature_data) == len(parent2.feature_data):
                child_data = (parent1.feature_data + parent2.feature_data) / 2
            else:
                # Use parent1 data if lengths differ
                child_data = parent1.feature_data.copy()

            child = FeatureCandidate(
                name=child_name,
                feature_data=child_data,
                creation_method="genetic_crossover",
                source_features=combined_sources
            )

            return child

        except Exception as e:
            self.logger.error(f"Crossover failed: {e}")
            return None

    def _mutate(self, feature: FeatureCandidate) -> Optional[FeatureCandidate]:
        """Mutate a feature"""
        try:
            # Apply random transformation
            mutations = [
                lambda x: x * (1 + np.random.normal(0, 1))),  # Gaussian noise
                lambda x: x + np.random.normal(0, 1) * 0.1, len(x)),  # Additive noise
                lambda x: x ** (1 + np.random.normal(0, 1)),  # Power transformation
                lambda x: np.log1p(np.abs(x)) * np.sign(x),  # Log transformation
            ]

            mutation_func = np.random.normal(0, 1)
            mutated_data = mutation_func(feature.feature_data.values)

            # Create mutated feature
            mutated_name = f"mutated_{feature.name}_{int(time.time())}"

            mutated_feature = FeatureCandidate(
                name=mutated_name,
                feature_data=pd.Series(mutated_data, index=feature.feature_data.index),
                creation_method="genetic_mutation",
                source_features=feature.source_features.copy()

            return mutated_feature

        except Exception as e:
            self.logger.error(f"Mutation failed: {e}")
            return None

class FeatureDiscoveryEngine:
    """Main feature discovery and optimization engine"""

    def __init__(self, config: Optional[DiscoveryConfig] = None):
        self.config = config or DiscoveryConfig()
        self.logger = logging.getLogger(f"{__name__}.FeatureDiscoveryEngine")

        # Core components
        self.automated_engineer = get_automated_feature_engineer()
        self.genetic_evolver = GeneticFeatureEvolver(self.config)

        # Feature management
        self.feature_candidates: Dict[str, FeatureCandidate] = {}
        self.active_features: Dict[str, FeatureCandidate] = {}
        self.champion_features: Dict[str, FeatureCandidate] = {}

        # Performance tracking
        self.performance_history: deque = deque(maxlen=self.config.performance_history_size)
        self.feature_performance_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Regime tracking
        self.current_regime: Optional[RegimeState] = None
        self.regime_history: deque = deque(maxlen=100)
        self.regime_feature_mapping: Dict[str, Set[str]] = defaultdict(set)

        # Discovery state
        self.discovery_mode = DiscoveryMode.EXPLORATORY
        self.last_discovery_run: Optional[datetime] = None
        self.discovery_iteration = 0

        self._lock = threading.RLock()

        # Start background discovery process
        self._start_discovery_thread()

        self.logger.info("Feature Discovery Engine initialized")

    def discover_features(self, data: pd.DataFrame, target_column: str,
                         regime_column: Optional[str] = None) -> List[FeatureCandidate]:
        """Main feature discovery method"""
        with self._lock:
            try:
                self.logger.info(f"Starting feature discovery iteration {self.discovery_iteration}")

                # Update regime if provided
                if regime_column and regime_column in data.columns:
                    self._update_regime_state(data[regime_column].iloc[-1])

                # Generate baseline features
                baseline_features = self._generate_baseline_features(data, target_column)

                # Determine discovery strategy based on mode
                if self.discovery_mode == DiscoveryMode.EXPLORATORY:
                    new_candidates = self._exploratory_discovery(data, target_column, baseline_features)
                elif self.discovery_mode == DiscoveryMode.EXPLOITATIVE:
                    new_candidates = self._exploitative_discovery(data, target_column, baseline_features)
                elif self.discovery_mode == DiscoveryMode.REGIME_ADAPTIVE:
                    new_candidates = self._regime_adaptive_discovery(data, target_column, baseline_features)
                else:  # PERFORMANCE_DRIVEN
                    new_candidates = self._performance_driven_discovery(data, target_column, baseline_features)

                # Evaluate candidates
                evaluated_candidates = self._evaluate_candidates(new_candidates, data, target_column)

                # Update feature candidates
                self._update_feature_candidates(evaluated_candidates)

                # Promote best candidates to active features
                self._promote_candidates_to_active()

                # Genetic evolution step
                if self.config.genetic_algorithm_enabled:
                    evolved_features = self._genetic_evolution_step(data, target_column)
                    self._update_feature_candidates(evolved_features)

                # Adapt discovery mode based on performance
                self._adapt_discovery_mode()

                self.discovery_iteration += 1
                self.last_discovery_run = datetime.now()

                self.logger.info(f"Discovery completed: {len(evaluated_candidates)} new candidates, "
                               f"{len(self.active_features)} active features")

                return evaluated_candidates

            except Exception as e:
                self.logger.error(f"Feature discovery failed: {e}")
                return []

    def _generate_baseline_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Generate baseline features using automated feature engineering"""
        try:
            # Use automated feature engineer for baseline features
            baseline_features = self.automated_engineer.transform(data, target_column)
            return baseline_features

        except Exception as e:
            self.logger.error(f"Baseline feature generation failed: {e}")
            return data

    def _exploratory_discovery(self, data: pd.DataFrame, target_column: str,
                             baseline_features: pd.DataFrame) -> List[FeatureCandidate]:
        """Exploratory feature discovery - wide search"""
        try:
            candidates = []

            # REMOVED: Mock data pattern not allowed in production
            feature_cols = [col for col in baseline_features.columns if col != target_column]

            for _ in range(self.config.max_candidates_per_iteration // 2):
                # REMOVED: Mock data pattern not allowed in production
                selected_features = np.random.normal(0, 1)),
                    replace=False
                )

                # Create interaction feature
                interaction_data = baseline_features[selected_features[0]].copy()
                for feature in selected_features[1:]:
                    # REMOVED: Mock data pattern not allowed in production
                    operation = np.random.normal(0, 1)

                    if operation == 'multiply':
                        interaction_data *= baseline_features[feature]
                    elif operation == 'add':
                        interaction_data += baseline_features[feature]
                    elif operation == 'divide':
                        interaction_data /= (baseline_features[feature] + 1e-8)
                    else:  # subtract
                        interaction_data -= baseline_features[feature]

                candidate_name = f"explore_{'_'.join(selected_features)}_{int(time.time())}"

                candidate = FeatureCandidate(
                    name=candidate_name,
                    feature_data=interaction_data.fillna(0),
                    creation_method="exploratory_random",
                    source_features=list(selected_features)

                candidates.append(candidate)

            # REMOVED: Mock data pattern not allowed in production
            for _ in range(self.config.max_candidates_per_iteration // 2):
                source_feature = np.random.normal(0, 1)
                source_data = baseline_features[source_feature]

                # REMOVED: Mock data pattern not allowed in production
                transformations = [
                    lambda x: np.log1p(np.abs(x)),
                    lambda x: np.sqrt(np.abs(x)),
                    lambda x: x ** 2,
                    lambda x: 1 / (x + 1e-8),
                    lambda x: x.rolling(5).mean().fillna(x),
                    lambda x: x.rolling(10).std().fillna(0)
                ]

                transform_func = np.random.normal(0, 1)
                transformed_data = transform_func(source_data)

                candidate_name = f"explore_transform_{source_feature}_{int(time.time())}"

                candidate = FeatureCandidate(
                    name=candidate_name,
                    feature_data=transformed_data.fillna(0),
                    creation_method="exploratory_transform",
                    source_features=[source_feature]
                )

                candidates.append(candidate)

            return candidates

        except Exception as e:
            self.logger.error(f"Exploratory discovery failed: {e}")
            return []

    def _exploitative_discovery(self, data: pd.DataFrame, target_column: str,
                              baseline_features: pd.DataFrame) -> List[FeatureCandidate]:
        """Exploitative discovery - refine known good features"""
        try:
            candidates = []

            # Get top performing features
            top_features = self._get_top_performing_features(5)

            if not top_features:
                # Fallback to exploratory if no top features
                return self._exploratory_discovery(data, target_column, baseline_features)

            # Refine top features
            for feature_name in top_features:
                if feature_name in baseline_features.columns:
                    source_data = baseline_features[feature_name]

                    # Create variations of top features
                    variations = [
                        source_data.rolling(3).mean().fillna(source_data),
                        source_data.rolling(7).mean().fillna(source_data),
                        source_data.ewm(span=5).mean(),
                        source_data.ewm(span=12).mean(),
                        source_data * 1.1,
                        source_data * 0.9
                    ]

                    for i, variation in enumerate(variations):
                        candidate_name = f"exploit_{feature_name}_var_{i}_{int(time.time())}"

                        candidate = FeatureCandidate(
                            name=candidate_name,
                            feature_data=variation.fillna(0),
                            creation_method="exploitative_refinement",
                            source_features=[feature_name]
                        )

                        candidates.append(candidate)

                        if len(candidates) >= self.config.max_candidates_per_iteration:
                            break

            return candidates[:self.config.max_candidates_per_iteration]

        except Exception as e:
            self.logger.error(f"Exploitative discovery failed: {e}")
            return []

    def _regime_adaptive_discovery(self, data: pd.DataFrame, target_column: str,
                                 baseline_features: pd.DataFrame) -> List[FeatureCandidate]:
        """Regime-adaptive discovery - adapt to market conditions"""
        try:
            candidates = []

            if not self.current_regime:
                # No regime info, fallback to exploratory
                return self._exploratory_discovery(data, target_column, baseline_features)

            # Get regime-specific features
            regime_features = self.regime_feature_mapping.get(self.current_regime.regime_id, set())

            # Adapt existing regime features
            for feature_name in list(regime_features)[:10]:  # Limit to top 10
                if feature_name in baseline_features.columns:
                    source_data = baseline_features[feature_name]

                    # Create regime-adapted versions
                    regime_adaptations = [
                        source_data.rolling(self.current_regime.duration).mean().fillna(source_data),
                        source_data.ewm(alpha=self.config.online_adaptation_rate).mean(),
                        source_data * (1 + self.current_regime.confidence * 0.1)
                    ]

                    for i, adaptation in enumerate(regime_adaptations):
                        candidate_name = f"regime_{self.current_regime.regime_id}_{feature_name}_{i}_{int(time.time())}"

                        candidate = FeatureCandidate(
                            name=candidate_name,
                            feature_data=adaptation.fillna(0),
                            creation_method="regime_adaptive",
                            source_features=[feature_name]
                        )

                        candidates.append(candidate)

            # Fill remaining slots with exploratory features
            remaining_slots = self.config.max_candidates_per_iteration - len(candidates)
            if remaining_slots > 0:
                exploratory_candidates = self._exploratory_discovery(data, target_column, baseline_features)
                candidates.extend(exploratory_candidates[:remaining_slots])

            return candidates

        except Exception as e:
            self.logger.error(f"Regime adaptive discovery failed: {e}")
            return []

    def _performance_driven_discovery(self, data: pd.DataFrame, target_column: str,
                                    baseline_features: pd.DataFrame) -> List[FeatureCandidate]:
        """Performance-driven discovery - focus on improving model performance"""
        try:
            candidates = []

            # Get features with recent performance decline
            declining_features = self._get_declining_features()

            # Create improved versions of declining features
            for feature_name in declining_features[:5]:
                if feature_name in baseline_features.columns:
                    source_data = baseline_features[feature_name]

                    # Performance improvement techniques
                    improvements = [
                        source_data.rolling(window=20).apply(lambda x: x.mean() + x.std()),  # Mean + std
                        source_data.rank(pct=True),  # Rank transformation
                        (source_data - source_data.rolling(50).mean()) / source_data.rolling(50).std(),  # Z-score
                        source_data.ewm(alpha=0.3).mean()  # Faster EMA
                    ]

                    for i, improvement in enumerate(improvements):
                        candidate_name = f"improve_{feature_name}_{i}_{int(time.time())}"

                        candidate = FeatureCandidate(
                            name=candidate_name,
                            feature_data=improvement.fillna(0),
                            creation_method="performance_improvement",
                            source_features=[feature_name]
                        )

                        candidates.append(candidate)

            # Fill with best feature combinations
            remaining_slots = self.config.max_candidates_per_iteration - len(candidates)
            if remaining_slots > 0:
                combination_candidates = self._create_performance_combinations(
                    baseline_features, target_column, remaining_slots
                )
                candidates.extend(combination_candidates)

            return candidates

        except Exception as e:
            self.logger.error(f"Performance driven discovery failed: {e}")
            return []

    def _evaluate_candidates(self, candidates: List[FeatureCandidate],
                           data: pd.DataFrame, target_column: str) -> List[FeatureCandidate]:
        """Evaluate feature candidates for performance"""
        try:
            if not HAS_SKLEARN:
                self.logger.warning("Scikit-learn not available for candidate evaluation")
                return candidates

            target = data[target_column].fillna(0)

            for candidate in candidates:
                try:
                    # Align candidate data with target
                    feature_data = candidate.feature_data.fillna(0)

                    # Ensure same length
                    min_length = min(len(feature_data), len(target))
                    if min_length < 10:  # Need minimum samples
                        candidate.performance_score = 0.0
                        continue

                    X = feature_data.iloc[-min_length:].values.reshape(-1, 1)
                    y = target.iloc[-min_length:].values

                    # Quick evaluation using correlation and simple model
                    correlation = np.corrcoef(X.flatten(), y)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0

                    # Simple linear model evaluation
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()

                    # Cross-validation score
                    cv_scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
                    cv_score = -cv_scores.mean() if len(cv_scores) > 0 else 1e6

                    # Combined performance score
                    performance_score = abs(correlation) * 0.7 + (1 / (1 + cv_score)) * 0.3
                    candidate.performance_score = performance_score
                    candidate.last_evaluated = datetime.now()

                    # Calculate stability score
                    candidate.stability_score = self._calculate_stability_score(feature_data)

                except Exception as e:
                    self.logger.warning(f"Candidate evaluation failed for {candidate.name}: {e}")
                    candidate.performance_score = 0.0

            return candidates

        except Exception as e:
            self.logger.error(f"Candidate evaluation failed: {e}")
            return candidates

    def _calculate_stability_score(self, feature_data: pd.Series) -> float:
        """Calculate stability score for feature"""
        try:
            if len(feature_data) < 20:
                return 0.0

            # Split into periods and calculate consistency
            period_size = len(feature_data) // self.config.stability_evaluation_periods
            if period_size < 2:
                return 0.0

            period_means = []
            for i in range(self.config.stability_evaluation_periods):
                start_idx = i * period_size
                end_idx = start_idx + period_size
                if end_idx <= len(feature_data):
                    period_mean = feature_data.iloc[start_idx:end_idx].mean()
                    if not np.isnan(period_mean):
                        period_means.append(period_mean)

            if len(period_means) < 2:
                return 0.0

            # Calculate coefficient of variation
            mean_of_means = np.mean(period_means)
            std_of_means = np.std(period_means)

            if mean_of_means == 0:
                return 0.0

            cv = std_of_means / abs(mean_of_means)
            stability_score = 1 / (1 + cv)  # Higher is more stable

            return stability_score

        except Exception:
            return 0.0

    def _update_feature_candidates(self, candidates: List[FeatureCandidate]):
        """Update feature candidate registry"""
        try:
            for candidate in candidates:
                self.feature_candidates[candidate.name] = candidate

                # Update performance cache
                self.feature_performance_cache[candidate.name].append(candidate.performance_score)

                # Regime-specific tracking
                if self.current_regime:
                    candidate.regime_performance[self.current_regime.regime_id] = candidate.performance_score

            # Cleanup old candidates
            self._cleanup_old_candidates()

        except Exception as e:
            self.logger.error(f"Feature candidate update failed: {e}")

    def _promote_candidates_to_active(self):
        """Promote best candidates to active features"""
        try:
            # Sort candidates by performance
            sorted_candidates = sorted(
                self.feature_candidates.values(),
                key=lambda c: c.performance_score,
                reverse=True
            )

            promotion_count = 0
            max_promotions = min(10, self.config.max_active_features - len(self.active_features))

            for candidate in sorted_candidates:
                if promotion_count >= max_promotions:
                    break

                # Promotion criteria
                if (candidate.performance_score > self.config.min_performance_improvement and
                    candidate.stability_score > self.config.min_stability_score and
                    candidate.status == FeatureStatus.CANDIDATE):

                    # Promote to active
                    candidate.status = FeatureStatus.ACTIVE
                    self.active_features[candidate.name] = candidate

                    # Remove from candidates
                    if candidate.name in self.feature_candidates:
                        del self.feature_candidates[candidate.name]

                    # Update regime mapping
                    if self.current_regime:
                        self.regime_feature_mapping[self.current_regime.regime_id].add(candidate.name)

                    promotion_count += 1
                    self.logger.info(f"Promoted feature to active: {candidate.name} (score: {candidate.performance_score:.4f})")

            # Deprecate poor performing active features
            self._deprecate_poor_features()

        except Exception as e:
            self.logger.error(f"Feature promotion failed: {e}")

    def _deprecate_poor_features(self):
        """Deprecate poor performing active features"""
        try:
            to_deprecate = []

            for feature_name, feature in self.active_features.items():
                # Check recent performance
                recent_scores = list(self.feature_performance_cache[feature_name])[-10:]

                if len(recent_scores) >= 5:
                    avg_recent_score = np.mean(recent_scores)

                    # Deprecate if performance dropped significantly
                    if avg_recent_score < self.config.min_performance_improvement * 0.5:
                        to_deprecate.append(feature_name)

            # Deprecate features
            for feature_name in to_deprecate:
                if feature_name in self.active_features:
                    feature = self.active_features[feature_name]
                    feature.status = FeatureStatus.DEPRECATED
                    del self.active_features[feature_name]

                    self.logger.info(f"Deprecated feature: {feature_name}")

        except Exception as e:
            self.logger.error(f"Feature deprecation failed: {e}")

    def _genetic_evolution_step(self, data: pd.DataFrame, target_column: str) -> List[FeatureCandidate]:
        """Perform genetic evolution step"""
        try:
            # Get current active features as parents
            parent_features = list(self.active_features.values())

            if len(parent_features) < 2:
                return []

            # Get performance scores
            performance_scores = {
                feature.name: feature.performance_score
                for feature in parent_features
            }

            # Evolve features
            evolved_features = self.genetic_evolver.evolve_features(parent_features, performance_scores)

            # Evaluate evolved features
            target = data[target_column].fillna(0)
            evaluated_evolved = self._evaluate_candidates(evolved_features, data, target_column)

            return evaluated_evolved

        except Exception as e:
            self.logger.error(f"Genetic evolution failed: {e}")
            return []

    def _update_regime_state(self, current_regime_id: str):
        """Update current regime state"""
        try:
            if not self.current_regime or self.current_regime.regime_id != current_regime_id:
                # Regime change detected
                self.current_regime = RegimeState(
                    regime_id=current_regime_id,
                    confidence=1.0,  # Would be calculated from regime detection
                    duration=1,
                    active_features=list(self.active_features.keys())

                self.regime_history.append(self.current_regime)
                self.logger.info(f"Regime change detected: {current_regime_id}")

                # Switch to regime adaptive mode
                self.discovery_mode = DiscoveryMode.REGIME_ADAPTIVE
            else:
                # Same regime, increment duration
                self.current_regime.duration += 1

        except Exception as e:
            self.logger.error(f"Regime state update failed: {e}")

    def _adapt_discovery_mode(self):
        """Adapt discovery mode based on performance"""
        try:
            if len(self.performance_history) < 20:
                return

            recent_performance = list(self.performance_history)[-10:]
            older_performance = list(self.performance_history)[-20:-10:]

            recent_avg = np.mean(recent_performance)
            older_avg = np.mean(older_performance)

            performance_trend = recent_avg - older_avg

            # Adapt mode based on performance trend
            if performance_trend > 0.05:
                # Good performance, continue exploitation
                self.discovery_mode = DiscoveryMode.EXPLOITATIVE
            elif performance_trend < -0.05:
                # Poor performance, try exploration
                self.discovery_mode = DiscoveryMode.EXPLORATORY
            else:
                # Stable performance, focus on performance improvement
                self.discovery_mode = DiscoveryMode.PERFORMANCE_DRIVEN

        except Exception as e:
            self.logger.error(f"Discovery mode adaptation failed: {e}")

    def _get_top_performing_features(self, n: int = 10) -> List[str]:
        """Get top performing features"""
        try:
            feature_scores = []

            for feature_name, feature in self.active_features.items():
                recent_scores = list(self.feature_performance_cache[feature_name])[-5:]
                if recent_scores:
                    avg_score = np.mean(recent_scores)
                    feature_scores.append((feature_name, avg_score))

            feature_scores.sort(key=lambda x: x[1], reverse=True)
            return [name for name, _ in feature_scores[:n]]

        except Exception:
            return []

    def _get_declining_features(self) -> List[str]:
        """Get features with declining performance"""
        try:
            declining = []

            for feature_name, feature in self.active_features.items():
                scores = list(self.feature_performance_cache[feature_name])

                if len(scores) >= 10:
                    recent_avg = np.mean(scores[-5:])
                    older_avg = np.mean(scores[-10:-5])

                    if recent_avg < older_avg * 0.9:  # 10% decline
                        declining.append(feature_name)

            return declining

        except Exception:
            return []

    def _create_performance_combinations(self, baseline_features: pd.DataFrame,
                                       target_column: str, max_features: int) -> List[FeatureCandidate]:
        """Create feature combinations for performance improvement"""
        try:
            candidates = []
            top_features = self._get_top_performing_features(5)

            # Create combinations of top features
            for i, feature1 in enumerate(top_features):
                for j, feature2 in enumerate(top_features):
                    if i < j and len(candidates) < max_features:  # Avoid duplicates
                        if feature1 in baseline_features.columns and feature2 in baseline_features.columns:
                            # Create combination
                            data1 = baseline_features[feature1]
                            data2 = baseline_features[feature2]

                            combined_data = data1 * data2  # Simple multiplication

                            candidate_name = f"combo_{feature1}_{feature2}_{int(time.time())}"

                            candidate = FeatureCandidate(
                                name=candidate_name,
                                feature_data=combined_data.fillna(0),
                                creation_method="performance_combination",
                                source_features=[feature1, feature2]
                            )

                            candidates.append(candidate)

            return candidates

        except Exception as e:
            self.logger.error(f"Performance combination creation failed: {e}")
            return []

    def _cleanup_old_candidates(self):
        """Clean up old candidates to manage memory"""
        try:
            if len(self.feature_candidates) <= self.config.max_candidates_per_iteration * 2:
                return

            # Remove oldest candidates with poor performance
            candidates_by_age = sorted(
                self.feature_candidates.items(),
                key=lambda x: x[1].created_at
            )

            to_remove = []
            for name, candidate in candidates_by_age:
                if candidate.performance_score < self.config.min_performance_improvement * 0.25:
                    to_remove.append(name)

                    if len(to_remove) >= len(self.feature_candidates) // 4:  # Remove 25%
                        break

            for name in to_remove:
                del self.feature_candidates[name]
                if name in self.feature_performance_cache:
                    del self.feature_performance_cache[name]

            if to_remove:
                self.logger.info(f"Cleaned up {len(to_remove)} old candidates")

        except Exception as e:
            self.logger.error(f"Candidate cleanup failed: {e}")

    def _start_discovery_thread(self):
        """Start background discovery thread"""
        def discovery_loop():
            while True:
                try:
                    time.sleep(300)  # Run every 5 minutes

                    # Background maintenance tasks
                    self._cleanup_old_candidates()
                    self._deprecate_poor_features()

                except Exception as e:
                    self.logger.error(f"Discovery thread error: {e}")

        discovery_thread = threading.Thread(target=discovery_loop, daemon=True)
        discovery_thread.start()

    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get comprehensive discovery summary"""
        with self._lock:
            return {
                'discovery_mode': self.discovery_mode.value,
                'discovery_iteration': self.discovery_iteration,
                'total_candidates': len(self.feature_candidates),
                'active_features': len(self.active_features),
                'champion_features': len(self.champion_features),
                'current_regime': self.current_regime.regime_id if self.current_regime else None,
                'regime_feature_mapping': {
                    regime: len(features) for regime, features in self.regime_feature_mapping.items()
                },
                'top_performing_features': self._get_top_performing_features(10),
                'declining_features': self._get_declining_features(),
                'avg_feature_performance': np.mean([
                    f.performance_score for f in self.active_features.values()
                ]) if self.active_features else 0.0,
                'last_discovery_run': self.last_discovery_run.isoformat() if self.last_discovery_run else None
            }


# Singleton feature discovery engine
_feature_discovery_engine = None
_fde_lock = threading.Lock()

def get_feature_discovery_engine(config: Optional[DiscoveryConfig] = None) -> FeatureDiscoveryEngine:
    """Get the singleton feature discovery engine"""
    global _feature_discovery_engine

    with _fde_lock:
        if _feature_discovery_engine is None:
            _feature_discovery_engine = FeatureDiscoveryEngine(config)
        return _feature_discovery_engine

def discover_features_for_regime(data: pd.DataFrame, target_column: str,
                                regime_column: Optional[str] = None) -> List[FeatureCandidate]:
    """Convenient function to discover features for current market regime"""
    engine = get_feature_discovery_engine()
    return engine.discover_features(data, target_column, regime_column)
