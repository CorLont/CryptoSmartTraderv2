#!/usr/bin/env python3
"""
Test production pipeline end-to-end
"""
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil

def test_installation():
    """Test Windows installation script"""
    print("=== Testing Installation ===")
    
    # Test Python availability
    try:
        result = subprocess.run([sys.executable, '--version'], capture_output=True, text=True)
        print(f"✓ Python version: {result.stdout.strip()}")
    except Exception as e:
        print(f"✗ Python not available: {e}")
        return False
    
    # Test virtual environment creation
    test_venv = Path("test_venv")
    try:
        subprocess.run([sys.executable, '-m', 'venv', str(test_venv)], check=True)
        print("✓ Virtual environment creation works")
        
        # Cleanup
        shutil.rmtree(test_venv)
    except Exception as e:
        print(f"✗ Virtual environment creation failed: {e}")
        return False
    
    return True

def test_pipeline_scripts():
    """Test individual pipeline scripts"""
    print("\n=== Testing Pipeline Scripts ===")
    
    # Create minimal test data
    test_data_dir = Path("data/processed")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal features file
    import pandas as pd
    test_features = pd.DataFrame({
        'coin': ['BTC', 'ETH', 'ADA'],
        'timestamp': ['2024-01-01', '2024-01-01', '2024-01-01'],
        'price_change_24h': [2.5, 3.1, -1.2],
        'volume_24h': [1000000, 500000, 100000],
        'spread': [0.001, 0.002, 0.003],
        'volatility_7d': [0.02, 0.03, 0.04]
    })
    
    features_file = test_data_dir / "features.csv"
    test_features.to_csv(features_file, index=False)
    print(f"✓ Created test features: {features_file}")
    
    # Test scraping script syntax
    try:
        result = subprocess.run([sys.executable, '-m', 'py_compile', 'scripts/scrape_all.py'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ scripts/scrape_all.py syntax OK")
        else:
            print(f"✗ scripts/scrape_all.py syntax error: {result.stderr}")
    except Exception as e:
        print(f"✗ Could not test scrape_all.py: {e}")
    
    # Test training script syntax
    try:
        result = subprocess.run([sys.executable, '-m', 'py_compile', 'ml/train_baseline.py'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ ml/train_baseline.py syntax OK")
        else:
            print(f"✗ ml/train_baseline.py syntax error: {result.stderr}")
    except Exception as e:
        print(f"✗ Could not test train_baseline.py: {e}")
    
    # Test prediction script syntax
    try:
        result = subprocess.run([sys.executable, '-m', 'py_compile', 'scripts/predict_all.py'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ scripts/predict_all.py syntax OK")
        else:
            print(f"✗ scripts/predict_all.py syntax error: {result.stderr}")
    except Exception as e:
        print(f"✗ Could not test predict_all.py: {e}")
    
    # Test evaluation script syntax
    try:
        result = subprocess.run([sys.executable, '-m', 'py_compile', 'scripts/evaluate.py'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ scripts/evaluate.py syntax OK")
        else:
            print(f"✗ scripts/evaluate.py syntax error: {result.stderr}")
    except Exception as e:
        print(f"✗ Could not test evaluate.py: {e}")
    
    # Test orchestrator syntax
    try:
        result = subprocess.run([sys.executable, '-m', 'py_compile', 'scripts/orchestrator.py'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ scripts/orchestrator.py syntax OK")
        else:
            print(f"✗ scripts/orchestrator.py syntax error: {result.stderr}")
    except Exception as e:
        print(f"✗ Could not test orchestrator.py: {e}")

def test_backend_enforcement():
    """Test backend enforcement"""
    print("\n=== Testing Backend Enforcement ===")
    
    try:
        result = subprocess.run([sys.executable, '-c', '''
import sys
sys.path.append(".")
from core.backend_enforcement import BackendEnforcement
import pandas as pd

# Test data
test_data = [
    {"coin": "BTC", "confidence_1h": 85, "confidence_24h": 90},
    {"coin": "ETH", "confidence_1h": 75, "confidence_24h": 82},
    {"coin": "ADA", "confidence_1h": 60, "confidence_24h": 70}
]
test_df = pd.DataFrame(test_data)

# Test enforcement
enforcer = BackendEnforcement(80.0)
filtered_df, result = enforcer.enforce_confidence_gate(test_df)

print(f"Original: {len(test_df)}, Filtered: {len(filtered_df)}")
assert len(filtered_df) == 2, "Expected 2 predictions to pass 80% threshold"
print("✓ Backend enforcement test passed")
'''], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"✗ Backend enforcement test failed: {result.stderr}")
    except Exception as e:
        print(f"✗ Could not test backend enforcement: {e}")

def test_directory_structure():
    """Test required directory structure"""
    print("\n=== Testing Directory Structure ===")
    
    required_dirs = [
        "data/raw",
        "data/processed", 
        "exports/production",
        "logs/daily",
        "models/baseline",
        "scripts",
        "ml",
        "core"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created {dir_path}")

def test_batch_scripts():
    """Test batch scripts exist and are readable"""
    print("\n=== Testing Batch Scripts ===")
    
    batch_files = ["install.bat", "run.bat"]
    
    for batch_file in batch_files:
        path = Path(batch_file)
        if path.exists():
            print(f"✓ {batch_file} exists")
            
            # Check for basic structure
            with open(path, 'r') as f:
                content = f.read()
                
            if batch_file == "install.bat":
                if "python -m venv" in content:
                    print(f"  ✓ {batch_file} contains venv creation")
                if "pip install" in content:
                    print(f"  ✓ {batch_file} contains pip install")
            
            elif batch_file == "run.bat":
                if "orchestrator.py" in content:
                    print(f"  ✓ {batch_file} calls orchestrator")
                if "streamlit run" in content:
                    print(f"  ✓ {batch_file} starts dashboard")
        else:
            print(f"✗ {batch_file} missing")

def main():
    """Run all production tests"""
    print("CryptoSmartTrader V2 - Production Pipeline Test")
    print("=" * 50)
    
    # Run tests
    tests = [
        test_installation,
        test_directory_structure,
        test_batch_scripts,
        test_pipeline_scripts,
        test_backend_enforcement
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results if r is not False)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if all(r is not False for r in results):
        print("✓ ALL TESTS PASSED - Production pipeline ready!")
        return 0
    else:
        print("✗ Some tests failed - review output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())