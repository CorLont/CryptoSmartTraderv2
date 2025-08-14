"""Agents API Router - Agent Status and Performance Monitoring"""

from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List, Optional, Dict, Any

from ..models.agents import AgentStatus, AgentMetrics, AgentPerformance
from ..dependencies import get_orchestrator, get_settings
from ...config import Settings

router = APIRouter(tags=["agents"], prefix="/agents")


@router.get("/status", response_model=List[AgentStatus], summary="Get All Agent Status")
async def get_agents_status(
    orchestrator=Depends(get_orchestrator), settings: Settings = Depends(get_settings)
) -> List[AgentStatus]:
    """
    Get operational status of all agents

    Returns current state, uptime, and error information for each agent
    """
    try:
        # Get agent status from orchestrator
        agents_status = await orchestrator.get_all_agent_status()

        return [
            AgentStatus(
                name=agent["name"],
                state=agent["state"],
                uptime_seconds=agent["uptime_seconds"],
                last_activity=agent["last_activity"],
                error_message=agent.get("error_message"),
            )
            for agent in agents_status
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent status: {str(e)}")


@router.get("/status/{agent_name}", response_model=AgentStatus, summary="Get Agent Status")
async def get_agent_status(agent_name: str, orchestrator=Depends(get_orchestrator)) -> AgentStatus:
    """
    Get status for a specific agent

    Returns detailed status information for the specified agent
    """
    try:
        # Get specific agent status from orchestrator
        agent_data = await orchestrator.get_agent_status(agent_name)

        if not agent_data:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")

        return AgentStatus(
            name=agent_data["name"],
            state=agent_data["state"],
            uptime_seconds=agent_data["uptime_seconds"],
            last_activity=agent_data["last_activity"],
            error_message=agent_data.get("error_message"),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve agent {agent_name} status: {str(e)}"
        )


@router.get("/metrics", response_model=List[AgentMetrics], summary="Get Agent Metrics")
async def get_agents_metrics(orchestrator=Depends(get_orchestrator)) -> List[AgentMetrics]:
    """
    Get performance metrics for all agents

    Returns processing statistics, response times, and success rates
    """
    try:
        # Get agent metrics from orchestrator
        agents_metrics = await orchestrator.get_all_agent_metrics()

        return [
            AgentMetrics(
                name=metric["name"],
                requests_processed=metric["requests_processed"],
                average_response_time_ms=metric["average_response_time_ms"],
                success_rate=metric["success_rate"],
                error_count=metric["error_count"],
                last_reset=metric["last_reset"],
            )
            for metric in agents_metrics
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent metrics: {str(e)}")


@router.get("/performance", response_model=List[AgentPerformance], summary="Get Agent Performance")
async def get_agents_performance(
    days: int = Query(default=7, ge=1, le=90, description="Performance evaluation period in days"),
    orchestrator=Depends(get_orchestrator),
) -> List[AgentPerformance]:
    """
    Get performance analysis for all agents

    Returns accuracy, precision, recall and other performance metrics
    """
    try:
        # Get agent performance from orchestrator
        agents_performance = await orchestrator.get_all_agent_performance(evaluation_days=days)

        return [
            AgentPerformance(
                name=perf["name"],
                accuracy=perf.get("accuracy"),
                precision=perf.get("precision"),
                recall=perf.get("recall"),
                f1_score=perf.get("f1_score"),
                confidence_score=perf["confidence_score"],
                recommendations_count=perf["recommendations_count"],
                successful_predictions=perf["successful_predictions"],
                evaluation_period_days=days,
            )
            for perf in agents_performance
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve agent performance: {str(e)}"
        )


@router.get(
    "/performance/{agent_name}", response_model=AgentPerformance, summary="Get Agent Performance"
)
async def get_agent_performance(
    agent_name: str,
    days: int = Query(default=7, ge=1, le=90, description="Performance evaluation period in days"),
    orchestrator=Depends(get_orchestrator),
) -> AgentPerformance:
    """
    Get performance analysis for a specific agent

    Returns detailed performance metrics for the specified agent
    """
    try:
        # Get specific agent performance from orchestrator
        perf_data = await orchestrator.get_agent_performance(
            agent_name=agent_name, evaluation_days=days
        )

        if not perf_data:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_name} not found or no performance data available",
            )

        return AgentPerformance(
            name=perf_data["name"],
            accuracy=perf_data.get("accuracy"),
            precision=perf_data.get("precision"),
            recall=perf_data.get("recall"),
            f1_score=perf_data.get("f1_score"),
            confidence_score=perf_data["confidence_score"],
            recommendations_count=perf_data["recommendations_count"],
            successful_predictions=perf_data["successful_predictions"],
            evaluation_period_days=days,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve agent {agent_name} performance: {str(e)}"
        )
