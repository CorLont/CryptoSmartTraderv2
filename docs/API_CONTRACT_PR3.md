# API Contract - PR3 Implementation

## 🎯 FastAPI App Factory + Health Endpoint with Tests

### Hybrid Implementation: PR3 Simplicity + Enterprise Features

Dit PR implementeert een schone API contract door PR3's eenvoudige app factory patroon te combineren met onze bestaande enterprise API infrastructuur.

#### ✅ PR3 Features Implemented

##### 1. Clean App Factory Pattern
```python
def get_app() -> FastAPI:
    """PR3 Style App Factory"""
    app = FastAPI(title="CryptoSmartTrader API", version="0.1.0")
    from .routers.health import router as health_router
    app.include_router(health_router)
    return app
```

**Usage:**
```bash
# PR3 Style startup
uv run uvicorn src.cryptosmarttrader.api.app:get_app --factory --host 0.0.0.0 --port 8001
```

##### 2. Simple Health Endpoint
```python
@router.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    """PR3 Style Simple Health Check"""
    return HealthOut(status="ok", score=0.97)
```

**Response Format:**
```json
{
    "status": "ok",
    "score": 0.97
}
```

##### 3. E2E API Testing
```python
@pytest.mark.api
async def test_health_endpoint(client):
    """PR3 Style Health Endpoint Test"""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) >= {"status", "score"}
    assert data["status"] in {"ok", "degraded", "fail"}
```

### 🏗️ Enterprise Extensions

#### Enhanced Health Endpoints
- **Simple Health**: `/health` - PR3 style basic status
- **Detailed Health**: `/health/detailed` - Enterprise system metrics

#### Enterprise Health Response
```json
{
    "status": "ok",
    "score": 0.856,
    "timestamp": 1673457600.123,
    "details": {
        "memory_percent": 45.2,
        "disk_percent": 12.8,
        "cpu_count": 4,
        "memory_available_gb": 2.1
    },
    "services": {
        "data_dir": "ok",
        "models_dir": "ok", 
        "logs_dir": "ok",
        "cache_dir": "missing"
    }
}
```

#### API Testing Framework
- **Async Test Client**: httpx.AsyncClient voor FastAPI testing
- **Test Fixtures**: Herbruikbare client fixture
- **Test Markers**: @pytest.mark.api voor API tests
- **Performance Tests**: Response time validation

### 📊 Implementation Details

#### Directory Structure
```
src/cryptosmarttrader/api/
├── app.py                 # App factory (PR3 + Enterprise)
├── routers/
│   └── health.py          # Health endpoints
└── models/               # Pydantic response models

tests/api/
├── __init__.py
├── conftest.py           # API test configuration
└── test_health.py        # Health endpoint tests
```

#### Dependencies Added
```toml
[project]
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "httpx>=0.25.0",      # For testing
    "anyio>=4.0.0",       # Async test support
    # ... existing dependencies
]
```

### 🚀 Testing & Validation

#### API Test Execution
```bash
# Run all API tests
uv run pytest tests/api/ -v

# Run specific health tests  
uv run pytest tests/api/test_health.py -v

# Run with performance markers
uv run pytest -m "api and performance" -v
```

#### Manual API Testing
```bash
# Start PR3 API server
uv run uvicorn src.cryptosmarttrader.api.app:get_app --factory --host 0.0.0.0 --port 8001

# Test endpoints
curl http://localhost:8001/health
curl http://localhost:8001/health/detailed
```

#### Expected Responses
- ✅ **Status Code**: 200 OK
- ✅ **Content-Type**: application/json
- ✅ **Response Time**: < 100ms voor health endpoints
- ✅ **Schema Validation**: Pydantic model validation

### 🔧 Architecture Benefits

#### Dual App Pattern
- **get_app()**: PR3 style minimal API voor testing
- **create_app()**: Enterprise API met middleware, security, metrics
- **Flexibility**: Choose appropriate app based on use case

#### Test Infrastructure
- **Isolated Testing**: Async client fixtures
- **Fast Feedback**: Simple health endpoint tests
- **Enterprise Testing**: Detailed health monitoring tests
- **Performance Validation**: Response time thresholds

#### API Contract Validation
- **Type Safety**: Pydantic models ensure consistent responses  
- **Schema Documentation**: Automatic OpenAPI/Swagger docs
- **Validation**: Request/response validation via FastAPI
- **Testing**: Comprehensive test coverage voor API contracts

### 📈 PR3 Integration Results

#### Features Successfully Integrated
1. ✅ **Clean App Factory**: Simple get_app() function
2. ✅ **Health Endpoint**: /health met HealthOut model
3. ✅ **E2E Testing**: AsyncClient met httpx
4. ✅ **Type Safety**: Pydantic response models
5. ✅ **Performance**: Fast, lightweight health checks

#### Enterprise Features Maintained
- ✅ **Comprehensive Middleware**: CORS, security, timing
- ✅ **Structured Logging**: Request/response logging
- ✅ **Advanced Health**: System metrics en service status
- ✅ **API Documentation**: Swagger/ReDoc integration
- ✅ **Error Handling**: Enterprise error responses

### 🔄 Development Workflow

#### API Development
1. **Define Models**: Pydantic request/response models
2. **Create Endpoints**: FastAPI router definitions
3. **Write Tests**: AsyncClient test implementation
4. **Validate Contract**: Test API schema compliance

#### Testing Strategy
```bash
# Development cycle
uv run pytest tests/api/ --tb=short
uv run uvicorn src.cryptosmarttrader.api.app:get_app --factory --reload
curl http://localhost:8001/health
```

---

**Implementation Date:** 2025-01-11  
**PR Bundle:** PR3 API Contract + Tests  
**Status:** ✅ Complete - Clean API Factory + Enterprise Extensions  
**Test Coverage:** Health endpoints, Response validation, Performance testing