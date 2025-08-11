# API Documentation - CryptoSmartTrader V2

## 🚀 Enterprise FastAPI Architecture

### Type-Safe API Router Skeleton

Complete enterprise-grade API implementation with comprehensive type safety, validation, and error handling.

#### Core Components

```
src/cryptosmarttrader/api/
├── __init__.py              # API package exports
├── app.py                   # FastAPI application factory
├── dependencies.py          # Dependency injection
├── models/                  # Pydantic response models
│   ├── __init__.py         # Model exports
│   ├── health.py           # Health check models
│   ├── market.py           # Market data models
│   ├── trading.py          # Trading models
│   └── agents.py           # Agent models
└── routers/                # API route handlers
    ├── __init__.py         # Router exports
    ├── health.py           # Health endpoints
    ├── market.py           # Market data endpoints
    ├── trading.py          # Trading endpoints
    └── agents.py           # Agent monitoring endpoints
```

### 📊 API Endpoints

#### Health Monitoring (`/api/v1/health/`)

| Endpoint | Method | Response Model | Description |
|----------|--------|----------------|-------------|
| `/health/` | GET | `HealthOut` | Basic system health |
| `/health/detailed` | GET | `HealthDetailOut` | Comprehensive health check |

**Example Response:**
```json
{
  "status": "healthy",
  "score": 0.95,
  "timestamp": "2025-01-11T12:00:00Z"
}
```

#### Market Data (`/api/v1/market/`)

| Endpoint | Method | Response Model | Description |
|----------|--------|----------------|-------------|
| `/market/data` | GET | `MarketDataOut` | Real-time market data |
| `/market/coin/{symbol}` | GET | `PriceData` | Specific coin data |
| `/market/top-gainers` | GET | `List[PriceData]` | Top gaining cryptocurrencies |
| `/market/top-losers` | GET | `List[PriceData]` | Top losing cryptocurrencies |

**Query Parameters:**
- `limit`: Number of results (1-1000)
- `sort_by`: Sort field (market_cap, volume, change)
- `min_volume`: Minimum 24h volume filter

#### Trading (`/api/v1/trading/`)

| Endpoint | Method | Response Model | Description |
|----------|--------|----------------|-------------|
| `/trading/signals` | GET | `List[SignalOut]` | Current trading signals |
| `/trading/signals/{symbol}` | GET | `List[SignalOut]` | Symbol-specific signals |
| `/trading/portfolio` | GET | `PortfolioOut` | Portfolio summary |
| `/trading/positions` | GET | `List[PositionOut]` | Active positions |

**Query Parameters:**
- `min_confidence`: Minimum confidence threshold (0.0-1.0)
- `limit`: Number of signals (1-200)
- `hours`: Signal history period (1-168)

#### Agents (`/api/v1/agents/`)

| Endpoint | Method | Response Model | Description |
|----------|--------|----------------|-------------|
| `/agents/status` | GET | `List[AgentStatus]` | All agent status |
| `/agents/status/{agent_name}` | GET | `AgentStatus` | Specific agent status |
| `/agents/metrics` | GET | `List[AgentMetrics]` | Agent performance metrics |
| `/agents/performance` | GET | `List[AgentPerformance]` | Agent performance analysis |

### 🔧 Type-Safe Models

#### Health Models
```python
class HealthOut(BaseModel):
    status: HealthStatus = Field(..., description="Overall system health status")
    score: float = Field(..., ge=0.0, le=1.0, description="Health score (0.0-1.0)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HealthDetailOut(BaseModel):
    status: HealthStatus
    score: float
    services: Dict[str, ServiceHealth]
    system_metrics: Dict[str, Any]
    trading_status: str  # GO/NO-GO
    warnings: List[str]
    errors: List[str]
```

#### Market Data Models
```python
class PriceData(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., BTC/USD)")
    price: Decimal = Field(..., description="Current price")
    volume_24h: Decimal = Field(..., description="24-hour trading volume")
    change_24h: float = Field(..., description="24-hour price change percentage")
    timestamp: datetime

class MarketDataOut(BaseModel):
    coins: List[PriceData]
    total_count: int
    last_updated: datetime
    data_source: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
```

#### Trading Models
```python
class SignalOut(BaseModel):
    symbol: str
    signal_type: SignalType  # BUY, SELL, HOLD
    confidence: float = Field(..., ge=0.0, le=1.0)
    strength: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    agent_source: str

class PortfolioOut(BaseModel):
    total_value: Decimal
    available_balance: Decimal
    unrealized_pnl: Decimal
    positions: List[PositionOut]
    position_count: int
```

### 🛡️ Enterprise Features

#### Security Middleware
- **TrustedHostMiddleware**: Restricts allowed hosts
- **CORSMiddleware**: Configurable CORS policy
- **Request timing**: Performance monitoring
- **Global exception handler**: Centralized error handling

#### Type Safety
- **Pydantic models**: Automatic validation and serialization
- **Type hints**: Full type annotation coverage
- **Enum constraints**: Strict value validation
- **Field validation**: Range and format constraints

#### Performance Monitoring
- **Request timing headers**: X-Process-Time
- **Performance metrics logging**: Response time tracking
- **Error rate monitoring**: Exception tracking
- **Prometheus integration**: Metrics collection

### 🔄 Dependency Injection

#### Settings Injection
```python
from ..dependencies import get_settings

@router.get("/endpoint")
async def endpoint(settings: Settings = Depends(get_settings)):
    # Use validated settings
    pass
```

#### Orchestrator Injection
```python
from ..dependencies import get_orchestrator

@router.get("/endpoint") 
async def endpoint(orchestrator=Depends(get_orchestrator)):
    # Use orchestrator for business logic
    data = await orchestrator.get_data()
    return data
```

### 📈 Usage Examples

#### Health Check
```bash
curl http://localhost:8001/api/v1/health/
curl http://localhost:8001/api/v1/health/detailed
```

#### Market Data
```bash
# Get top 50 coins by market cap
curl "http://localhost:8001/api/v1/market/data?limit=50&sort_by=market_cap"

# Get Bitcoin data
curl http://localhost:8001/api/v1/market/coin/BTC

# Get top gainers with minimum volume
curl "http://localhost:8001/api/v1/market/top-gainers?limit=20&min_volume=1000000"
```

#### Trading Signals
```bash
# Get high-confidence signals
curl "http://localhost:8001/api/v1/trading/signals?min_confidence=0.8&limit=20"

# Get Bitcoin signals from last 24 hours
curl "http://localhost:8001/api/v1/trading/signals/BTC?hours=24"

# Get portfolio summary
curl http://localhost:8001/api/v1/trading/portfolio
```

#### Agent Monitoring
```bash
# Get all agent status
curl http://localhost:8001/api/v1/agents/status

# Get specific agent performance
curl "http://localhost:8001/api/v1/agents/performance/sentiment_agent?days=7"
```

### 🚀 API Application Factory

#### FastAPI Configuration
```python
def create_app() -> FastAPI:
    app = FastAPI(
        title="CryptoSmartTrader V2 API",
        description="Enterprise Trading Intelligence API",
        version="2.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    # Add security and CORS middleware
    # Include routers with prefix /api/v1
    # Setup exception handling
    
    return app
```

#### Startup Configuration
- **Lifespan management**: Graceful startup/shutdown
- **Settings validation**: Fail-fast configuration
- **Logging setup**: Structured JSON logging
- **Health endpoint**: Simple /health for load balancers

### 📊 API Documentation

#### Interactive Documentation
- **Swagger UI**: http://localhost:8001/api/docs
- **ReDoc**: http://localhost:8001/api/redoc
- **OpenAPI Schema**: http://localhost:8001/api/openapi.json

#### Response Format
All endpoints return consistent JSON responses with:
- Type-safe Pydantic models
- Proper HTTP status codes
- Detailed error messages
- Timestamp information
- Correlation IDs for tracing

---

**Last Updated:** 2025-01-11  
**API Version:** 2.0.0  
**FastAPI Version:** 0.100+  
**Type Safety:** Complete Pydantic Coverage