"""
CryptoSmartTrader V2 - System Validator
Complete system validation and health checks
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class SystemValidator:
    """Complete system validation and integrity checks"""
    
    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)
        
        # Core system components to validate
        self.core_components = [
            'cache_manager',
            'config',
            'data_manager', 
            'monitoring_system',
            'real_time_pipeline',
            'multi_horizon_ml'
        ]
        
        # System requirements
        self.requirements = {
            'min_memory_mb': 512,
            'required_packages': ['pandas', 'numpy', 'ccxt', 'lightgbm', 'streamlit'],
            'required_directories': ['models', 'data', 'logs'],
            'required_files': ['config.json', 'containers.py', 'app.py']
        }
    
    def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete system validation"""
        try:
            self.logger.info("Starting complete system validation")
            
            validation_results = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'PASS',
                'components': {},
                'errors': [],
                'warnings': [],
                'recommendations': []
            }
            
            # Validate core components
            for component in self.core_components:
                result = self._validate_component(component)
                validation_results['components'][component] = result
                
                if not result['status']:
                    validation_results['errors'].extend(result.get('errors', []))
                    validation_results['overall_status'] = 'FAIL'
            
            # Validate system requirements
            req_result = self._validate_requirements()
            validation_results['components']['system_requirements'] = req_result
            
            if not req_result['status']:
                validation_results['errors'].extend(req_result.get('errors', []))
                validation_results['overall_status'] = 'FAIL'
            
            # Validate data integrity
            data_result = self._validate_data_integrity()
            validation_results['components']['data_integrity'] = data_result
            
            if not data_result['status']:
                validation_results['warnings'].extend(data_result.get('warnings', []))
            
            # Validate ML pipeline
            ml_result = self._validate_ml_pipeline()
            validation_results['components']['ml_pipeline'] = ml_result
            
            if not ml_result['status']:
                validation_results['warnings'].extend(ml_result.get('warnings', []))
            
            # Generate recommendations
            validation_results['recommendations'] = self._generate_recommendations(validation_results)
            
            self.logger.info(f"System validation completed: {validation_results['overall_status']}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"System validation failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'ERROR',
                'error': str(e)
            }
    
    def _validate_component(self, component_name: str) -> Dict[str, Any]:
        """Validate individual system component"""
        try:
            result = {
                'component': component_name,
                'status': False,
                'errors': [],
                'info': {}
            }
            
            # Try to get component from container
            if hasattr(self.container, component_name):
                try:
                    component = getattr(self.container, component_name)()
                    result['status'] = True
                    result['info']['initialized'] = True
                    
                    # Component-specific validation
                    if component_name == 'cache_manager':
                        result['info']['cache_size'] = len(component._cache) if hasattr(component, '_cache') else 0
                        result['info']['memory_usage_mb'] = component.get_total_memory_usage() if hasattr(component, 'get_total_memory_usage') else 0
                    
                    elif component_name == 'real_time_pipeline':
                        result['info']['pipeline_active'] = getattr(component, 'pipeline_active', False)
                        result['info']['task_count'] = len(getattr(component, 'pipeline_tasks', {}))
                    
                    elif component_name == 'multi_horizon_ml':
                        result['info']['models_loaded'] = len(getattr(component, 'models', {}))
                        result['info']['horizons'] = list(getattr(component, 'horizons', {}).keys())
                    
                except Exception as e:
                    result['errors'].append(f"Component initialization failed: {e}")
                    result['status'] = False
            else:
                result['errors'].append(f"Component {component_name} not found in container")
            
            return result
            
        except Exception as e:
            return {
                'component': component_name,
                'status': False,
                'errors': [f"Validation error: {e}"]
            }
    
    def _validate_requirements(self) -> Dict[str, Any]:
        """Validate system requirements"""
        try:
            result = {
                'component': 'system_requirements',
                'status': True,
                'errors': [],
                'info': {}
            }
            
            # Check required packages
            missing_packages = []
            for package in self.requirements['required_packages']:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                result['errors'].append(f"Missing packages: {missing_packages}")
                result['status'] = False
            
            result['info']['packages_checked'] = len(self.requirements['required_packages'])
            result['info']['missing_packages'] = missing_packages
            
            # Check required directories
            missing_dirs = []
            for dir_name in self.requirements['required_directories']:
                dir_path = Path(dir_name)
                if not dir_path.exists():
                    try:
                        dir_path.mkdir(parents=True, exist_ok=True)
                        self.logger.info(f"Created missing directory: {dir_name}")
                    except Exception as e:
                        missing_dirs.append(dir_name)
            
            result['info']['directories_checked'] = len(self.requirements['required_directories'])
            result['info']['missing_directories'] = missing_dirs
            
            # Check required files
            missing_files = []
            for file_name in self.requirements['required_files']:
                file_path = Path(file_name)
                if not file_path.exists():
                    missing_files.append(file_name)
            
            if missing_files:
                result['errors'].append(f"Missing files: {missing_files}")
                result['status'] = False
            
            result['info']['files_checked'] = len(self.requirements['required_files'])
            result['info']['missing_files'] = missing_files
            
            return result
            
        except Exception as e:
            return {
                'component': 'system_requirements',
                'status': False,
                'errors': [f"Requirements validation failed: {e}"]
            }
    
    def _validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity and completeness"""
        try:
            result = {
                'component': 'data_integrity',
                'status': True,
                'warnings': [],
                'info': {}
            }
            
            try:
                cache_manager = self.container.cache_manager()
                
                # Check for validated data
                price_data_count = 0
                sentiment_data_count = 0
                whale_data_count = 0
                
                for key in cache_manager._cache.keys():
                    if key.startswith('validated_price_data_'):
                        price_data_count += 1
                    elif key.startswith('validated_sentiment_'):
                        sentiment_data_count += 1
                    elif key.startswith('validated_whale_'):
                        whale_data_count += 1
                
                result['info']['price_data_coins'] = price_data_count
                result['info']['sentiment_data_coins'] = sentiment_data_count
                result['info']['whale_data_coins'] = whale_data_count
                
                # Check data completeness
                if price_data_count == 0:
                    result['warnings'].append("No validated price data found")
                    result['status'] = False
                
                if sentiment_data_count == 0:
                    result['warnings'].append("No validated sentiment data found")
                
                if whale_data_count == 0:
                    result['warnings'].append("No validated whale data found")
                
                # Check for dummy data (should not exist)
                dummy_data_keys = [k for k in cache_manager._cache.keys() if 'dummy' in k.lower()]
                if dummy_data_keys:
                    result['warnings'].append(f"Found dummy data entries: {len(dummy_data_keys)}")
                
                result['info']['total_cache_entries'] = len(cache_manager._cache)
                result['info']['dummy_entries_found'] = len(dummy_data_keys)
                
            except Exception as e:
                result['warnings'].append(f"Cache validation failed: {e}")
                result['status'] = False
            
            return result
            
        except Exception as e:
            return {
                'component': 'data_integrity',
                'status': False,
                'warnings': [f"Data integrity check failed: {e}"]
            }
    
    def _validate_ml_pipeline(self) -> Dict[str, Any]:
        """Validate ML pipeline functionality"""
        try:
            result = {
                'component': 'ml_pipeline',
                'status': True,
                'warnings': [],
                'info': {}
            }
            
            try:
                # Check multi-horizon ML system
                ml_system = self.container.multi_horizon_ml()
                
                result['info']['models_available'] = len(ml_system.models)
                result['info']['horizons_configured'] = list(ml_system.horizons.keys())
                result['info']['training_config'] = ml_system.training_config
                
                # Check if models need training
                retrain_needed = ml_system.check_retrain_needed()
                result['info']['models_needing_retrain'] = sum(retrain_needed.values())
                
                if result['info']['models_available'] == 0:
                    result['warnings'].append("No ML models loaded - training required")
                
                if result['info']['models_needing_retrain'] > 0:
                    result['warnings'].append(f"{result['info']['models_needing_retrain']} models need retraining")
                
            except Exception as e:
                result['warnings'].append(f"ML system validation failed: {e}")
                result['status'] = False
            
            return result
            
        except Exception as e:
            return {
                'component': 'ml_pipeline',
                'status': False,
                'warnings': [f"ML pipeline validation failed: {e}"]
            }
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate system improvement recommendations"""
        recommendations = []
        
        try:
            # Check component failures
            for comp_name, comp_result in validation_results['components'].items():
                if not comp_result.get('status', True):
                    recommendations.append(f"Fix {comp_name} component errors")
            
            # Check data integrity
            data_integrity = validation_results['components'].get('data_integrity', {})
            info = data_integrity.get('info', {})
            
            if info.get('price_data_coins', 0) < 10:
                recommendations.append("Increase price data collection - aim for 50+ coins")
            
            if info.get('sentiment_data_coins', 0) < 5:
                recommendations.append("Enhance sentiment data scraping coverage")
            
            if info.get('dummy_entries_found', 0) > 0:
                recommendations.append("Remove all dummy data entries (strict requirement)")
            
            # Check ML pipeline
            ml_pipeline = validation_results['components'].get('ml_pipeline', {})
            ml_info = ml_pipeline.get('info', {})
            
            if ml_info.get('models_available', 0) == 0:
                recommendations.append("Train initial ML models for all horizons")
            
            if ml_info.get('models_needing_retrain', 0) > 0:
                recommendations.append("Schedule model retraining for degraded models")
            
            # General recommendations
            if validation_results['overall_status'] == 'FAIL':
                recommendations.append("Priority: Fix all critical errors before production use")
            
            if len(validation_results['warnings']) > 5:
                recommendations.append("Address system warnings to improve reliability")
            
        except Exception as e:
            recommendations.append(f"Recommendation generation failed: {e}")
        
        return recommendations
    
    def fix_common_issues(self) -> Dict[str, Any]:
        """Automatically fix common system issues"""
        try:
            self.logger.info("Starting automatic issue resolution")
            
            fixes_applied = {
                'timestamp': datetime.now().isoformat(),
                'fixes': [],
                'errors': []
            }
            
            # Create missing directories
            for dir_name in self.requirements['required_directories']:
                dir_path = Path(dir_name)
                if not dir_path.exists():
                    try:
                        dir_path.mkdir(parents=True, exist_ok=True)
                        fixes_applied['fixes'].append(f"Created directory: {dir_name}")
                    except Exception as e:
                        fixes_applied['errors'].append(f"Failed to create {dir_name}: {e}")
            
            # Initialize cache if needed
            try:
                cache_manager = self.container.cache_manager()
                if not hasattr(cache_manager, '_cache'):
                    cache_manager._cache = {}
                    fixes_applied['fixes'].append("Initialized cache manager")
            except Exception as e:
                fixes_applied['errors'].append(f"Cache initialization failed: {e}")
            
            # Clear any dummy data
            try:
                cache_manager = self.container.cache_manager()
                dummy_keys = [k for k in cache_manager._cache.keys() if 'dummy' in k.lower()]
                for key in dummy_keys:
                    cache_manager.delete(key)
                    fixes_applied['fixes'].append(f"Removed dummy data: {key}")
            except Exception as e:
                fixes_applied['errors'].append(f"Dummy data cleanup failed: {e}")
            
            self.logger.info(f"Applied {len(fixes_applied['fixes'])} fixes")
            
            return fixes_applied
            
        except Exception as e:
            self.logger.error(f"Automatic issue resolution failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'fixes': [],
                'errors': [str(e)]
            }