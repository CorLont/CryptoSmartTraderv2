# Testing Strategy - CryptoSmartTrader V2

## 📋 Test Configuration

### Pytest Settings (pytest.ini)

#### Core Configuration
```ini
[pytest]
addopts = -q --strict-markers --maxfail=1
```

**Flags Explained:**
- `-q` - Quiet output, less verbose
- `--strict-markers` - Fail if unregistered markers are used
- `--maxfail=1` - Stop after first failure for fast feedback

### 🏷️ Test Markers

#### Core Test Categories

| Marker | Purpose | Usage |
|--------|---------|-------|
| `unit` | Snelle unit tests zonder externe dependencies | `@pytest.mark.unit` |
| `integration` | Gebruikt externe API's of I/O | `@pytest.mark.integration` |
| `slow` | Langzame of resource-intensieve tests | `@pytest.mark.slow` |

#### Specialized Categories

| Marker | Purpose | Example |
|--------|---------|---------|
| `smoke` | Basis functionaliteit tests voor CI/CD | Quick health checks |
| `regression` | Tests voor eerder gevonden bugs | Bug fix validation |
| `performance` | Performance en timing gerelateerde tests | Speed benchmarks |
| `security` | Security en vulnerability tests | Authentication tests |
| `api` | Tests die externe API endpoints gebruiken | Kraken API tests |
| `ml` | Machine learning model tests | Model validation |
| `trading` | Trading logic en financial calculations | Portfolio tests |
| `temporal` | Temporal validation en time-series tests | Time series validation |
| `dashboard` | Streamlit dashboard tests | UI component tests |
| `api_key` | Tests requiring API keys | External service tests |
| `property` | Property-based tests using Hypothesis | Fuzz testing |

### 🚀 Test Execution Strategies

#### Development Workflow
```bash
# Quick unit tests during development
pytest -m "unit and not slow"

# Integration tests before commit
pytest -m "integration"

# Full test suite before release
pytest
```

#### CI/CD Pipeline
```bash
# Smoke tests (fastest)
pytest -m "smoke" --maxfail=1

# Unit tests
pytest -m "unit and not slow" --maxfail=3

# Integration tests (if API keys available)
pytest -m "integration" --maxfail=5
```

#### Performance Testing
```bash
# Performance benchmarks
pytest -m "performance" --benchmark-only

# Slow tests (manual trigger)
pytest -m "slow" --timeout=300
```

### 📊 Test Categories by Component

#### Core System Tests
```python
@pytest.mark.unit
def test_config_validation():
    """Fast configuration validation"""
    pass

@pytest.mark.integration
@pytest.mark.api_key
def test_kraken_connection():
    """External API integration"""
    pass
```

#### ML/AI Component Tests
```python
@pytest.mark.ml
@pytest.mark.slow
def test_model_training():
    """Model training validation"""
    pass

@pytest.mark.ml
@pytest.mark.performance
def test_prediction_speed():
    """Prediction performance benchmark"""
    pass
```

#### Trading System Tests
```python
@pytest.mark.trading
@pytest.mark.unit
def test_portfolio_calculation():
    """Fast trading calculations"""
    pass

@pytest.mark.trading
@pytest.mark.integration
def test_order_execution():
    """Trading system integration"""
    pass
```

#### Security Tests
```python
@pytest.mark.security
def test_api_key_protection():
    """Ensure API keys are not exposed"""
    pass

@pytest.mark.security
@pytest.mark.api
def test_authentication():
    """Authentication security"""
    pass
```

### 🔄 Test Automation

#### Pre-commit Hooks
```bash
# Fast tests before commit
pytest -m "unit and smoke" --maxfail=1
```

#### Continuous Integration
```yaml
# GitHub Actions example
- name: Unit Tests
  run: pytest -m "unit" --cov=core

- name: Integration Tests  
  run: pytest -m "integration"
  env:
    KRAKEN_API_KEY: ${{ secrets.KRAKEN_API_KEY }}
```

#### Nightly Testing
```bash
# Comprehensive nightly tests
pytest -m "slow or performance" --cov=. --html=report.html
```

### 📈 Coverage Strategy

#### Target Coverage by Component
- **Core:** >90% line coverage
- **Agents:** >85% line coverage  
- **API:** >80% line coverage
- **Utils:** >95% line coverage

#### Coverage Commands
```bash
# Generate coverage report
pytest --cov=core --cov=agents --cov-report=html

# Coverage with missing lines
pytest --cov=. --cov-report=term-missing

# Fail if coverage below threshold
pytest --cov=core --cov-fail-under=85
```

### 🛠️ Test Data Management

#### Test Data Principles
1. **No Real API Keys in Tests** - Use environment variables
2. **Deterministic Test Data** - Fixed timestamps, predictable data
3. **Isolation** - Each test independent
4. **Cleanup** - Auto-cleanup of test artifacts

#### Example Test Structure
```python
@pytest.mark.integration
@pytest.mark.api_key
def test_market_data_fetch():
    """Test real market data fetching"""
    if not os.getenv('KRAKEN_API_KEY'):
        pytest.skip("API key required")
    
    # Test implementation
    pass

@pytest.mark.unit
def test_price_calculation():
    """Test price calculations with mock data"""
    # Fast unit test with predictable data
    pass
```

### 🚨 Error Handling Strategy

#### Pytest Configuration
- **Strict Markers:** Prevents typos in test markers
- **Fast Failure:** `--maxfail=1` for quick feedback
- **Clear Output:** `-q` for focused error messages

#### Test Reliability
```python
@pytest.mark.slow
@pytest.mark.api
def test_external_service():
    """Handle flaky external services"""
    @retry(stop=stop_after_attempt(3))
    def _test_call():
        # Actual test logic
        pass
```

### 📋 Test Maintenance

#### Regular Tasks
- **Weekly:** Review test performance, update slow tests
- **Monthly:** Check coverage gaps, add missing tests
- **Quarterly:** Refactor test structure, update markers

#### Marker Usage Guidelines
1. **Tag All Tests** - Every test should have appropriate markers
2. **Multiple Markers** - Tests can have multiple markers
3. **Consistent Naming** - Use Dutch for consistency
4. **Clear Descriptions** - Marker descriptions should be self-explanatory

---

**Last Updated:** 2025-01-11  
**Pytest Version:** 7.0+  
**Coverage Target:** >85% overall