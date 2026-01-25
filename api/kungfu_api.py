"""
Kung Fu Tournament API endpoints for AI Service
Handles tournament scheduling using LLM
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import time

from services.tournament_scheduling_service import TournamentSchedulingService

logger = logging.getLogger(__name__)

# Create router for kungfu endpoints
router = APIRouter(prefix="/kungfu", tags=["kungfu"])

# Pydantic models
class TournamentSchedulingRequest(BaseModel):
    planning_events: List[Dict[str, Any]]
    rings: List[Dict[str, Any]]
    head_judges: List[Dict[str, Any]]
    side_judges: List[Dict[str, Any]]
    helpers: List[Dict[str, Any]]
    dummies: List[Dict[str, Any]]
    time_slots: List[Dict[str, Any]]

class TournamentSchedulingResponse(BaseModel):
    scheduled_events: List[Dict[str, Any]]
    success: bool
    message: str
    processing_time: float

# Global service instance (will be initialized with model/tokenizer)
tournament_service: TournamentSchedulingService = None

def initialize_service(model, tokenizer):
    """Initialize tournament scheduling service with model and tokenizer"""
    global tournament_service
    tournament_service = TournamentSchedulingService(model, tokenizer)

@router.post("/schedule", response_model=TournamentSchedulingResponse)
async def schedule_tournament(request: TournamentSchedulingRequest):
    """
    Generate tournament schedule using LLM-based constraint solving
    Replaces OptaPlanner with AI logic
    """
    try:
        logger.info(f"Scheduling tournament with {len(request.planning_events)} events")
        
        if tournament_service is None:
            raise HTTPException(status_code=503, detail="Tournament service not initialized")
        
        # Convert request to dictionary format expected by service
        tournament_data = {
            "planning_events": [event.dict() for event in request.planning_events] if request.planning_events and hasattr(request.planning_events[0], 'dict') else request.planning_events,
            "rings": [ring.dict() for ring in request.rings] if request.rings and hasattr(request.rings[0], 'dict') else request.rings,
            "head_judges": [judge.dict() for judge in request.head_judges] if request.head_judges and hasattr(request.head_judges[0], 'dict') else request.head_judges,
            "side_judges": [judge.dict() for judge in request.side_judges] if request.side_judges and hasattr(request.side_judges[0], 'dict') else request.side_judges,
            "helpers": [helper.dict() for helper in request.helpers] if request.helpers and hasattr(request.helpers[0], 'dict') else request.helpers,
            "dummies": [dummy.dict() for dummy in request.dummies] if request.dummies and hasattr(request.dummies[0], 'dict') else request.dummies,
            "time_slots": [slot.dict() for slot in request.time_slots] if request.time_slots and hasattr(request.time_slots[0], 'dict') else request.time_slots
        }
        
        # Generate schedule
        start_time = time.time()
        
        scheduled_events = await tournament_service.generate_schedule(tournament_data)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Successfully scheduled {len(scheduled_events)} events in {processing_time:.2f} seconds")
        
        return TournamentSchedulingResponse(
            scheduled_events=scheduled_events,
            success=True,
            message=f"Successfully scheduled {len(scheduled_events)} events",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error scheduling tournament: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tournament scheduling failed: {str(e)}")

@router.get("/health")
async def tournament_health():
    """Check tournament scheduling service health"""
    try:
        if tournament_service is None:
            return {"status": "NOT_INITIALIZED", "service": "Tournament Scheduling"}
        
        # Simple health check - just verify the service is initialized
        return {
            "status": "OK", 
            "service": "Tournament Scheduling",
            "model_loaded": tournament_service.model is not None,
            "tokenizer_loaded": tournament_service.tokenizer is not None
        }
        
    except Exception as e:
        logger.error(f"Tournament service health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tournament service unhealthy: {str(e)}")
