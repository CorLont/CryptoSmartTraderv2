#!/usr/bin/env python3
"""
Import Path Resolver - Fixes import mismatches
Ensures consistent import paths across all modules
"""

import sys
from pathlib import Path
from typing import Any, Optional


class ImportPathResolver:
    """Resolves import path issues across the system"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self._setup_paths()

    def _setup_paths(self):
        """Setup consistent import paths"""

        # Add project root to Python path if not present
        root_str = str(self.project_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        # Add core directory to path
        core_path = str(self.project_root / "core")
        if core_path not in sys.path:
            sys.path.insert(1, core_path)

    def safe_import(self, module_path: str, fallback_paths: list = None) -> Optional[Any]:
        """Safely import module with fallback paths"""

        # Try primary import path
        try:
            parts = module_path.split(".")
            import importlib
            module = importlib.import_module(module_path)
            return module
        except ImportError:
            pass

        # Try fallback paths
        if fallback_paths:
            for fallback in fallback_paths:
                try:
                    parts = fallback.split(".")
                    module = importlib.import_module(fallback)
                    return module
                except ImportError:
                    continue

        return None

    def get_synthetic_augmentation_module(self):
        """Get synthetic data augmentation with path resolution"""

        # Try different possible paths
        paths_to_try = [
            "core.synthetic_data_augmentation",
            "synthetic_data_augmentation",
            "core.synthetic_data_generator",
        ]

        for path in paths_to_try:
            module = self.safe_import(path)
            if module:
                return module

        return None

    def get_confidence_gate_module(self):
        """Get confidence gate with path resolution"""

        # Always use unified gate now
        paths_to_try = [
            "core.unified_confidence_gate",
            "core.strict_confidence_gate",
            "orchestration.strict_gate_standalone",
        ]

        for path in paths_to_try:
            module = self.safe_import(path)
            if module:
                return module

        return None

    def get_logging_module(self):
        """Get consistent logging module"""

        # Always use consolidated logging
        paths_to_try = [
            "core.consolidated_logging_manager",
            "core.unified_structured_logger",
            "core.structured_logger",
        ]

        for path in paths_to_try:
            module = self.safe_import(path)
            if module:
                return module

        return None


# Global resolver instance
_resolver: Optional[ImportPathResolver] = None


def get_import_resolver() -> ImportPathResolver:
    """Get global import resolver instance"""
    global _resolver
    if _resolver is None:
        _resolver = ImportPathResolver()
    return _resolver


# Convenience functions for common imports
def get_logger_function():
    """Get the correct logger function regardless of import path"""
    resolver = get_import_resolver()
    logging_module = resolver.get_logging_module()

    if logging_module:
        # Try different logger function names
        for func_name in [
            "get_consolidated_logger",
            "get_unified_logger",
            "get_structured_logger",
            "get_logger",
        ]:
            if hasattr(logging_module, func_name):
                return getattr(logging_module, func_name)

    # Ultimate fallback to standard logging
    import logging

    return logging.getLogger


def get_confidence_gate_function():
    """Get the correct confidence gate function"""
    resolver = get_import_resolver()
    gate_module = resolver.get_confidence_gate_module()

    if gate_module:
        # Try different function names
        for func_name in [
            "get_unified_confidence_gate",
            "get_strict_confidence_gate",
            "apply_strict_gate_orchestration",
        ]:
            if hasattr(gate_module, func_name):
                return getattr(gate_module, func_name)

    return None


def get_synthetic_classes():
    """Get synthetic data classes with fallbacks"""
    resolver = get_import_resolver()
    synth_module = resolver.get_synthetic_augmentation_module()

    classes = {}
    if synth_module:
        for class_name in ["BlackSwanGenerator", "SyntheticScenario", "RegimeShiftGenerator"]:
            if hasattr(synth_module, class_name):
                classes[class_name] = getattr(synth_module, class_name)

    return classes
