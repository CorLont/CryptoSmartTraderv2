# api/main.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
from contextlib import asynccontextmanager

from containers import ApplicationContainer
from models.validation_models import (
    CoinSymbolRequest, MarketDataRequest, PredictionRequest,
    TradingSignalRequest, BacktestRequest, AgentConfigRequest,
    HealthCheckResponse, MetricsResponse, ErrorResponse
)
from utils.metrics import metrics_server
from config.logging_config import setup_logging
from config.settings import config

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize dependency container
container = ApplicationContainer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for FastAPI app"""
    # Startup
    logger.info("Starting CryptoSmartTrader API", extra={"component": "api_startup"})
    yield
    # Shutdown
    logger.info("Shutting down CryptoSmartTrader API", extra={"component": "api_shutdown"})


# Create FastAPI app
app = FastAPI(
    title="CryptoSmartTrader V2 API",
    description="Professional cryptocurrency trading intelligence system with multi-agent architecture",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency injection
async def get_config_manager():
    """Get configuration manager dependency"""
    return container.config()


async def get_health_monitor():
    """Get health monitor dependency"""
    return container.health_monitor()


async def get_data_manager():
    """Get data manager dependency"""
    return container.data_manager()


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions with structured response"""
    metrics_server.record_error("http_error", "api")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", extra={"component": "api_error"})
    metrics_server.record_error("internal_error", "api")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )


# Health and monitoring endpoints
@app.get("/health", response_model=HealthCheckResponse)
async def health_check(health_monitor=Depends(get_health_monitor)):
    """Get system health status"""
    try:
        metrics_server.record_request("/health", "GET")
        health_data = health_monitor.get_system_health()
        
        return HealthCheckResponse(
            status="healthy" if health_data.get("grade", "F") in ["A", "B"] else "degraded",
            components=health_data.get("component_health", {}),
            overall_grade=health_data.get("grade", "F"),
            overall_grade_numeric=health_data.get("score", 0.0) / 100.0,
            details=health_data
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(health_monitor=Depends(get_health_monitor)):
    """Get system metrics"""
    try:
        metrics_server.record_request("/metrics", "GET")
        health_data = health_monitor.get_system_health()
        
        return MetricsResponse(
            request_count=1000,  # This would come from actual metrics
            error_rate=0.02,
            avg_response_time=0.15,
            health_score=health_data.get("score", 0.0) / 100.0,
            active_agents=len(health_data.get("component_health", {})),
            cache_hit_ratio=0.85
        )
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics retrieval failed")


# Market data endpoints
@app.post("/api/v1/market-data")
async def get_market_data(
    request: MarketDataRequest,
    data_manager=Depends(get_data_manager)
):
    """Get market data for specified symbols"""
    try:
        metrics_server.record_request("/api/v1/market-data", "POST")
        
        # Get market data through data manager
        data = await data_manager.get_market_data(
            symbols=request.symbols,
            timeframe=request.timeframe.value,
            limit=request.limit
        )
        
        return {"status": "success", "data": data}
        
    except Exception as e:
        logger.error(f"Market data request failed: {e}")
        raise HTTPException(status_code=500, detail="Market data request failed")


# Prediction endpoints
@app.post("/api/v1/predictions")
async def get_predictions(
    request: PredictionRequest,
    container_dep=Depends(lambda: container)
):
    """Get ML predictions for a symbol"""
    try:
        metrics_server.record_request("/api/v1/predictions", "POST")
        
        ml_agent = container_dep.ml_predictor_agent()
        predictions = await ml_agent.predict_price(
            symbol=request.symbol,
            horizons=request.prediction_horizons,
            confidence_threshold=request.confidence_threshold
        )
        
        return {"status": "success", "predictions": predictions}
        
    except Exception as e:
        logger.error(f"Prediction request failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction request failed")


# Trading signal endpoints
@app.post("/api/v1/signals")
async def get_trading_signals(
    request: TradingSignalRequest,
    container_dep=Depends(lambda: container)
):
    """Generate trading signals for a symbol"""
    try:
        metrics_server.record_request("/api/v1/signals", "POST")
        
        trade_agent = container_dep.trade_executor_agent()
        signals = await trade_agent.generate_signals(
            symbol=request.symbol,
            position_size=request.position_size,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            risk_tolerance=request.risk_tolerance
        )
        
        return {"status": "success", "signals": signals}
        
    except Exception as e:
        logger.error(f"Trading signal request failed: {e}")
        raise HTTPException(status_code=500, detail="Trading signal request failed")


# Backtesting endpoints
@app.post("/api/v1/backtest")
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    container_dep=Depends(lambda: container)
):
    """Run backtesting analysis"""
    try:
        metrics_server.record_request("/api/v1/backtest", "POST")
        
        backtest_agent = container_dep.backtest_agent()
        
        # Run backtest in background
        background_tasks.add_task(
            backtest_agent.run_backtest,
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            strategy=request.strategy_name,
            parameters=request.parameters
        )
        
        return {"status": "accepted", "message": "Backtest started in background"}
        
    except Exception as e:
        logger.error(f"Backtest request failed: {e}")
        raise HTTPException(status_code=500, detail="Backtest request failed")


# Agent management endpoints
@app.get("/api/v1/agents")
async def get_agents(health_monitor=Depends(get_health_monitor)):
    """Get all agents status"""
    try:
        metrics_server.record_request("/api/v1/agents", "GET")
        agent_status = health_monitor.get_agent_status()
        return {"status": "success", "agents": agent_status}
        
    except Exception as e:
        logger.error(f"Agent status request failed: {e}")
        raise HTTPException(status_code=500, detail="Agent status request failed")


@app.put("/api/v1/agents/{agent_name}/config")
async def update_agent_config(
    agent_name: str,
    request: AgentConfigRequest,
    config_manager=Depends(get_config_manager)
):
    """Update agent configuration"""
    try:
        metrics_server.record_request(f"/api/v1/agents/{agent_name}/config", "PUT")
        
        # Update agent configuration
        success = config_manager.update_config({
            f"agents.{agent_name}": {
                "enabled": request.enabled,
                "parameters": request.parameters,
                "priority": request.priority
            }
        })
        
        if success:
            return {"status": "success", "message": f"Agent {agent_name} configuration updated"}
        else:
            raise HTTPException(status_code=400, detail="Configuration update failed")
            
    except Exception as e:
        logger.error(f"Agent config update failed: {e}")
        raise HTTPException(status_code=500, detail="Agent config update failed")


# Coin management endpoints
@app.get("/api/v1/coins")
async def get_coins(container_dep=Depends(lambda: container)):
    """Get supported cryptocurrency list"""
    try:
        metrics_server.record_request("/api/v1/coins", "GET")
        coin_registry = container_dep.coin_registry()
        coins = coin_registry.get_all_coins(active_only=True)
        return {"status": "success", "coins": coins}
        
    except Exception as e:
        logger.error(f"Coins request failed: {e}")
        raise HTTPException(status_code=500, detail="Coins request failed")


@app.post("/api/v1/coins/search")
async def search_coins(
    request: CoinSymbolRequest,
    container_dep=Depends(lambda: container)
):
    """Search for cryptocurrency symbols"""
    try:
        metrics_server.record_request("/api/v1/coins/search", "POST")
        coin_registry = container_dep.coin_registry()
        
        results = []
        for symbol in request.symbols:
            matches = coin_registry.search_coins(symbol, limit=5)
            results.extend(matches)
        
        return {"status": "success", "results": results}
        
    except Exception as e:
        logger.error(f"Coin search failed: {e}")
        raise HTTPException(status_code=500, detail="Coin search failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config.server_host,
        port=8001,  # Different port from Streamlit
        reload=config.debug_mode
    )