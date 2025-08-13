#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Synthetic Data Augmentation Engine
Generates synthetic market scenarios for edge case training and stress testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import json
from pathlib import Path

# ML imports
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SyntheticScenario:
    """Represents a synthetic market scenario"""
    scenario_type: str
    description: str
    data: pd.DataFrame
    metadata: Dict[str, Any]
    risk_level: str
    probability: float
    timestamp: datetime

class ScenarioGenerator(ABC):
    """Abstract base class for scenario generators"""

    @abstractmethod
    def generate_scenario(self, base_data: pd.DataFrame, **kwargs) -> SyntheticScenario:
        """Generate a synthetic scenario based on base data"""
        pass

class BlackSwanGenerator(ScenarioGenerator):
    """Generates black swan event scenarios"""

    def __init__(self, random_seed: Optional[int] = None):
        self.logger = logging.getLogger(__name__)
        self.random_seed = random_seed

    def generate_scenario(self, base_data: pd.DataFrame, **kwargs) -> SyntheticScenario:
        """Generate black swan market crash scenario"""

        severity = kwargs.get('severity', 'moderate')  # mild, moderate, severe
        duration_days = kwargs.get('duration_days', 7)
        seed = kwargs.get('seed', self.random_seed)

        # Set deterministic seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Deep copy base data to prevent SettingWithCopy warnings
        synthetic_data = base_data.copy(deep=True)

        # Define crash parameters
        crash_params = {
            'mild': {'drop_pct': -0.3, 'volatility_mult': 3.0},
            'moderate': {'drop_pct': -0.5, 'volatility_mult': 5.0},
            'severe': {'drop_pct': -0.8, 'volatility_mult': 8.0}
        }

        params = crash_params.get(severity, crash_params['moderate'])

        # Generate crash scenario
        n_points = min(len(synthetic_data), duration_days * 24)  # Assuming hourly data

        for col in synthetic_data.columns:
            if 'price' in col.lower():
                # Copy-safe writes using .loc with index masks
                idx = synthetic_data.index[:n_points]

                # Exponential decay crash
                crash_factor = np.exp(np.linspace(0, np.log(1 + params['drop_pct']), n_points))
                synthetic_data.loc[idx, col] = synthetic_data.loc[idx, col].to_numpy() * crash_factor

                # Increased volatility
                volatility = synthetic_data[col].pct_change().std()
                noise = np.random.normal(0, 1)
                synthetic_data.loc[idx, col] = synthetic_data.loc[idx, col].to_numpy() * (1 + noise)

            elif 'volume' in col.lower():
                # Volume spike during crash - copy-safe
                idx = synthetic_data.index[:n_points]
                volume_spike = np.random.lognormal(1.5, 0.5, n_points)
                synthetic_data.loc[idx, col] = synthetic_data.loc[idx, col].to_numpy() * volume_spike

        scenario = SyntheticScenario(
            scenario_type="black_swan",
            description=f"{severity.title()} market crash scenario",
            data=synthetic_data,
            metadata={
                'severity': severity,
                'duration_days': duration_days,
                'drop_percentage': params['drop_pct'] * 100,
                'volatility_multiplier': params['volatility_mult']
            },
            risk_level="high",
            probability=0.01 if severity == 'severe' else 0.05,
            timestamp=datetime.now()

        return scenario

class RegimeShiftGenerator(ScenarioGenerator):
    """Generates market regime shift scenarios"""

    def __init__(self, random_seed: Optional[int] = None):
        self.logger = logging.getLogger(__name__)
        self.random_seed = random_seed

    def generate_scenario(self, base_data: pd.DataFrame, **kwargs) -> SyntheticScenario:
        """Generate market regime shift scenario"""

        from_regime = kwargs.get('from_regime', 'bull')  # bull, bear, sideways
        to_regime = kwargs.get('to_regime', 'bear')
        transition_days = kwargs.get('transition_days', 14)
        seed = kwargs.get('seed', self.random_seed)

        # Set deterministic seed if provided
        if seed is not None:
            np.random.seed(seed)

        synthetic_data = base_data.copy(deep=True)

        # Define regime characteristics
        regime_params = {
            'bull': {'trend': 0.002, 'volatility': 0.02},
            'bear': {'trend': -0.001, 'volatility': 0.03},
            'sideways': {'trend': 0.0, 'volatility': 0.015}
        }

        from_params = regime_params[from_regime]
        to_params = regime_params[to_regime]

        n_transition = min(len(synthetic_data), transition_days * 24)

        for col in synthetic_data.columns:
            if 'price' in col.lower():
                # Gradual trend change - copy-safe approach
                trend_change = np.linspace(from_params['trend'], to_params['trend'], n_transition)
                volatility_change = np.linspace(from_params['volatility'], to_params['volatility'], n_transition)

                # Work on a copy of the column values to avoid chained indexing
                price_values = synthetic_data[col].values.copy()

                for i in range(n_transition):
                    if i > 0:
                        daily_return = trend_change[i] + np.random.normal(0, 1)
                        price_values[i] = price_values[i-1] * (1 + daily_return)

                # Apply all changes at once using .loc
                synthetic_data.loc[:n_transition-1, col] = price_values[:n_transition]

                # Make the rest of the series consistent with new regime (not just transition)
                if n_transition < len(synthetic_data):
                    remaining_points = len(synthetic_data) - n_transition
                    for i in range(remaining_points):
                        idx = n_transition + i
                        if idx < len(synthetic_data):
                            daily_return = to_params['trend'] + np.random.normal(0, 1)
                            price_values[idx] = price_values[idx-1] * (1 + daily_return)

                    # Update the remaining part
                    synthetic_data.loc[n_transition:, col] = price_values[n_transition:]

        scenario = SyntheticScenario(
            scenario_type="regime_shift",
            description=f"Market regime shift from {from_regime} to {to_regime}",
            data=synthetic_data,
            metadata={
                'from_regime': from_regime,
                'to_regime': to_regime,
                'transition_days': transition_days
            },
            risk_level="medium",
            probability=0.1,
            timestamp=datetime.now()

        return scenario

class FlashCrashGenerator(ScenarioGenerator):
    """Generates flash crash scenarios"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_scenario(self, base_data: pd.DataFrame, **kwargs) -> SyntheticScenario:
        """Generate flash crash scenario"""

        crash_magnitude = kwargs.get('crash_magnitude', 0.15)  # 15% flash crash
        recovery_hours = kwargs.get('recovery_hours', 2)

        synthetic_data = base_data.copy()

        # Flash crash at random point
        crash_start = random.randint(1, 100) - recovery_hours - 10
        crash_end = crash_start + recovery_hours

        for col in synthetic_data.columns:
            if 'price' in col.lower():
                # Sudden drop
                synthetic_data[col].iloc[crash_start] *= (1 - crash_magnitude)

                # Gradual recovery
                recovery_points = np.linspace(
                    synthetic_data[col].iloc[crash_start],
                    synthetic_data[col].iloc[crash_start] * (1 + crash_magnitude * 0.9),  # 90% recovery
                    recovery_hours
                )

                for i, recovery_price in enumerate(recovery_points):
                    if crash_start + i + 1 < len(synthetic_data):
                        synthetic_data[col].iloc[crash_start + i + 1] = recovery_price

            elif 'volume' in col.lower():
                # Volume spike during flash crash
                synthetic_data[col].iloc[crash_start:crash_end] *= 10

        scenario = SyntheticScenario(
            scenario_type="flash_crash",
            description=f"{crash_magnitude*100:.1f}% flash crash with {recovery_hours}h recovery",
            data=synthetic_data,
            metadata={
                'crash_magnitude': crash_magnitude,
                'recovery_hours': recovery_hours,
                'crash_start_index': crash_start
            },
            risk_level="high",
            probability=0.02,
            timestamp=datetime.now()

        return scenario

class WhaleManipulationGenerator(ScenarioGenerator):
    """Generates whale manipulation scenarios"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_scenario(self, base_data: pd.DataFrame, **kwargs) -> SyntheticScenario:
        """Generate whale manipulation scenario"""

        manipulation_type = kwargs.get('type', 'pump_dump')  # pump_dump, accumulation, distribution
        intensity = kwargs.get('intensity', 'medium')  # low, medium, high

        synthetic_data = base_data.copy()

        manipulation_params = {
            'low': {'price_impact': 0.05, 'volume_mult': 3},
            'medium': {'price_impact': 0.12, 'volume_mult': 5},
            'high': {'price_impact': 0.25, 'volume_mult': 10}
        }

        params = manipulation_params[intensity]

        if manipulation_type == 'pump_dump':
            # Pump phase (first 30% of data)
            pump_length = len(synthetic_data) // 3
            dump_start = pump_length
            dump_length = len(synthetic_data) // 6

            for col in synthetic_data.columns:
                if 'price' in col.lower():
                    # Pump phase
                    pump_factor = np.linspace(1, 1 + params['price_impact'], pump_length)
                    synthetic_data[col].iloc[:pump_length] *= pump_factor

                    # Dump phase
                    dump_factor = np.linspace(1, 1 - params['price_impact'] * 1.2, dump_length)
                    synthetic_data[col].iloc[dump_start:dump_start+dump_length] *= dump_factor

                elif 'volume' in col.lower():
                    # Volume spikes during manipulation
                    synthetic_data[col].iloc[:pump_length] *= params['volume_mult']
                    synthetic_data[col].iloc[dump_start:dump_start+dump_length] *= params['volume_mult'] * 1.5

        scenario = SyntheticScenario(
            scenario_type="whale_manipulation",
            description=f"{manipulation_type.replace('_', ' ').title()} whale manipulation ({intensity} intensity)",
            data=synthetic_data,
            metadata={
                'manipulation_type': manipulation_type,
                'intensity': intensity,
                'price_impact': params['price_impact']
            },
            risk_level="high",
            probability=0.05,
            timestamp=datetime.now()

        return scenario

class AdversarialNoiseGenerator(ScenarioGenerator):
    """Generates adversarial noise scenarios for model robustness"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_scenario(self, base_data: pd.DataFrame, **kwargs) -> SyntheticScenario:
        """Generate adversarial noise scenario"""

        noise_type = kwargs.get('noise_type', 'gaussian')  # gaussian, uniform, laplace
        noise_intensity = kwargs.get('noise_intensity', 0.02)

        synthetic_data = base_data.copy()

        for col in synthetic_data.columns:
            if any(keyword in col.lower() for keyword in ['price', 'volume', 'return']):
                noise = np.zeros(len(synthetic_data))
                if noise_type == 'gaussian':
                    noise = np.random.normal(0, 1))
                elif noise_type == 'uniform':
                    noise = np.random.normal(0, 1))
                elif noise_type == 'laplace':
                    noise = np.random.laplace(0, noise_intensity/2, len(synthetic_data))

                synthetic_data[col] += synthetic_data[col] * noise

        scenario = SyntheticScenario(
            scenario_type="adversarial_noise",
            description=f"{noise_type.title()} adversarial noise (intensity: {noise_intensity})",
            data=synthetic_data,
            metadata={
                'noise_type': noise_type,
                'noise_intensity': noise_intensity
            },
            risk_level="low",
            probability=0.3,
            timestamp=datetime.now()

        return scenario

class SyntheticDataAugmentationEngine:
    """Main engine for synthetic data augmentation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize generators
        self.generators = {
            'black_swan': BlackSwanGenerator(),
            'regime_shift': RegimeShiftGenerator(),
            'flash_crash': FlashCrashGenerator(),
            'whale_manipulation': WhaleManipulationGenerator(),
            'adversarial_noise': AdversarialNoiseGenerator()
        }

        self.generated_scenarios = []

    def generate_scenario_suite(self, base_data: pd.DataFrame, scenario_types: Optional[List[str]] = None) -> List[SyntheticScenario]:
        """Generate a comprehensive suite of synthetic scenarios"""

        if scenario_types is None:
            scenario_types = list(self.generators.keys())

        scenarios = []

        for scenario_type in scenario_types:
            if scenario_type in self.generators:
                try:
                    # Generate multiple variants of each scenario type
                    if scenario_type == 'black_swan':
                        for severity in ['mild', 'moderate', 'severe']:
                            scenario = self.generators[scenario_type].generate_scenario(
                                base_data, severity=severity, duration_days=np.random.normal(0, 1)
                            scenarios.append(scenario)

                    elif scenario_type == 'regime_shift':
                        regime_combinations = [
                            ('bull', 'bear'), ('bear', 'bull'), ('bull', 'sideways'), ('sideways', 'bear')
                        ]
                        for from_regime, to_regime in regime_combinations:
                            scenario = self.generators[scenario_type].generate_scenario(
                                base_data, from_regime=from_regime, to_regime=to_regime
                            )
                            scenarios.append(scenario)

                    elif scenario_type == 'flash_crash':
                        for magnitude in [0.1, 0.15, 0.25]:
                            scenario = self.generators[scenario_type].generate_scenario(
                                base_data, crash_magnitude=magnitude, recovery_hours=np.random.normal(0, 1)
                            scenarios.append(scenario)

                    elif scenario_type == 'whale_manipulation':
                        for manipulation_type in ['pump_dump']:
                            for intensity in ['low', 'medium', 'high']:
                                scenario = self.generators[scenario_type].generate_scenario(
                                    base_data, type=manipulation_type, intensity=intensity
                                )
                                scenarios.append(scenario)

                    elif scenario_type == 'adversarial_noise':
                        for noise_type in ['gaussian', 'uniform', 'laplace']:
                            scenario = self.generators[scenario_type].generate_scenario(
                                base_data, noise_type=noise_type, noise_intensity=np.random.normal(0, 1)
                            scenarios.append(scenario)

                except Exception as e:
                    self.logger.error(f"Failed to generate {scenario_type} scenario: {e}")

        self.generated_scenarios.extend(scenarios)
        return scenarios

    def stress_test_model(self, model, scenarios: List[SyntheticScenario]) -> Dict[str, Any]:
        """Stress test a model against synthetic scenarios"""

        results = {
            'total_scenarios': len(scenarios),
            'scenario_results': [],
            'overall_robustness': 0.0,
            'risk_breakdown': {},
            'timestamp': datetime.now()
        }

        for scenario in scenarios:
            try:
                # Prepare features (simplified - assumes model has predict method)
                features = self._prepare_features(scenario.data)

                if hasattr(model, 'predict'):
                    predictions = model.predict(features)

                    # Calculate stability metrics
                    stability_score = self._calculate_stability(predictions)

                    scenario_result = {
                        'scenario_type': scenario.scenario_type,
                        'description': scenario.description,
                        'risk_level': scenario.risk_level,
                        'stability_score': stability_score,
                        'prediction_variance': np.var(predictions) if len(predictions) > 1 else 0,
                        'extreme_predictions': np.sum(np.abs(predictions) > 2) / len(predictions) if len(predictions) > 0 else 0
                    }

                    results['scenario_results'].append(scenario_result)

                    # Update risk breakdown
                    if scenario.risk_level not in results['risk_breakdown']:
                        results['risk_breakdown'][scenario.risk_level] = []
                    results['risk_breakdown'][scenario.risk_level].append(stability_score)

            except Exception as e:
                self.logger.error(f"Stress test failed for scenario {scenario.scenario_type}: {e}")

        # Calculate overall robustness
        if results['scenario_results']:
            results['overall_robustness'] = np.mean([r['stability_score'] for r in results['scenario_results']])

        return results

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features from scenario data (simplified)"""

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        features = data[numeric_cols].fillna(0).values

        # Basic scaling
        if features.shape[0] > 1:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)

        return np.array(features)

    def _calculate_stability(self, predictions: np.ndarray) -> float:
        """Calculate stability score for predictions"""

        if len(predictions) == 0:
            return 0.0

        # Stability based on prediction variance and extreme values
        variance_score = max(0.0, 1.0 - float(np.var(predictions)) / 10.0)  # Normalize variance
        extreme_penalty = float(np.sum(np.abs(predictions) > 3)) / len(predictions)  # Penalty for extreme predictions

        stability = max(0.0, variance_score - extreme_penalty)
        return stability

    def get_augmentation_summary(self) -> Dict[str, Any]:
        """Get summary of generated scenarios"""

        summary = {
            'total_scenarios': len(self.generated_scenarios),
            'scenario_types': {},
            'risk_distribution': {},
            'coverage_metrics': {},
            'last_updated': datetime.now()
        }

        for scenario in self.generated_scenarios:
            # Count by type
            if scenario.scenario_type not in summary['scenario_types']:
                summary['scenario_types'][scenario.scenario_type] = 0
            summary['scenario_types'][scenario.scenario_type] += 1

            # Count by risk level
            if scenario.risk_level not in summary['risk_distribution']:
                summary['risk_distribution'][scenario.risk_level] = 0
            summary['risk_distribution'][scenario.risk_level] += 1

        # Calculate coverage metrics
        if self.generated_scenarios:
            summary['coverage_metrics'] = {
                'avg_probability': np.mean([s.probability for s in self.generated_scenarios]),
                'high_risk_coverage': summary['risk_distribution'].get('high', 0) / len(self.generated_scenarios),
                'scenario_diversity': len(summary['scenario_types']) / len(self.generated_scenarios)
            }

        return summary

    def save_scenarios(self, filepath: str):
        """Save generated scenarios to file"""

        scenarios_data = []
        for scenario in self.generated_scenarios:
            scenario_dict = {
                'scenario_type': scenario.scenario_type,
                'description': scenario.description,
                'metadata': scenario.metadata,
                'risk_level': scenario.risk_level,
                'probability': scenario.probability,
                'timestamp': scenario.timestamp.isoformat(),
                'data_shape': scenario.data.shape,
                'data_columns': list(scenario.data.columns)
            }
            scenarios_data.append(scenario_dict)

        with open(filepath, 'w') as f:
            json.dump(scenarios_data, f, indent=2)

        self.logger.info(f"Saved {len(scenarios_data)} scenarios to {filepath}")

# Global instance
_synthetic_engine = None

def get_synthetic_augmentation_engine() -> SyntheticDataAugmentationEngine:
    """Get or create synthetic data augmentation engine"""
    global _synthetic_engine

    if _synthetic_engine is None:
        _synthetic_engine = SyntheticDataAugmentationEngine()

    return _synthetic_engine

def generate_stress_test_scenarios(base_data: pd.DataFrame, scenario_types: Optional[List[str]] = None) -> List[SyntheticScenario]:
    """Generate stress test scenarios for edge case training"""

    engine = get_synthetic_augmentation_engine()
    return engine.generate_scenario_suite(base_data, scenario_types)

def evaluate_model_robustness(model, base_data: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate model robustness against synthetic scenarios"""

    engine = get_synthetic_augmentation_engine()
    scenarios = engine.generate_scenario_suite(base_data)

    return engine.stress_test_model(model, scenarios)

if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    # Generate demo data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    demo_data = pd.DataFrame({
        'btc_price': 40000 + np.cumsum(np.random.normal(0, 1)),
        'eth_price': 2500 + np.cumsum(np.random.normal(0, 1)),
        'btc_volume': 1000 + np.abs(np.random.normal(0, 1))
    })

    # Generate scenarios
    scenarios = generate_stress_test_scenarios(demo_data)

    print(f"Generated {len(scenarios)} stress test scenarios")
    for scenario in scenarios[:5]:
        print(f"- {scenario.description} (Risk: {scenario.risk_level})")
