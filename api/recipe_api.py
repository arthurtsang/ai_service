"""
Recipe API endpoints for AI Service
Handles recipe analysis, import, categorization, and chat
"""

import asyncio
import inspect
import uuid
import time
from fastapi import APIRouter, HTTPException, Request, Depends, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import os

from services.recipe_analysis_service import RecipeAnalysisService
from services.import_service import import_recipe_from_url, ImportRecipeRequest, ImportRecipeResponse
from services.video_import_service import import_recipe_from_video_url
from services.chat_service import generate_chat_response, ChatRequest, ChatResponse
from utils.llm_thread import run_llm_in_thread
import torch

logger = logging.getLogger(__name__)

# In-memory store for async import jobs: job_id -> { status, url, result?, error?, created_at, started_at?, step?, updated_at? }
_import_jobs: Dict[str, Dict[str, Any]] = {}
_video_import_jobs: Dict[str, Dict[str, Any]] = {}

# Create router for recipe endpoints
router = APIRouter(prefix="/recipe", tags=["recipe"])

# Security for internal API access
security = HTTPBearer(auto_error=False)

async def verify_internal_access(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """
    Verify that the request is from an internal service (localhost) or has a valid API key.
    This protects the AI service from being called directly by external users.
    """
    # Allow requests from localhost (internal services)
    client_host = request.client.host if request.client else None
    if client_host in ["127.0.0.1", "localhost", "::1"]:
        return True
    
    # Check for API key in Authorization header
    api_key = os.getenv("AI_SERVICE_API_KEY")
    if api_key and credentials and credentials.credentials == api_key:
        return True
    
    # Reject external requests without valid credentials
    raise HTTPException(
        status_code=403,
        detail="Access denied. This is an internal service endpoint."
    )

# Pydantic models
class RecipeAnalysisRequest(BaseModel):
    title: str
    description: Optional[str] = None
    ingredients: str
    instructions: str

class ImportRecipeRequest(BaseModel):
    url: str

class AutoCategoryRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    ingredients: Optional[str] = None
    instructions: Optional[str] = None

class ChatRequest(BaseModel):
    question: str


class DebugPromptRequest(BaseModel):
    """Request body for /recipe/debug-prompt - test a raw prompt and see exact model output."""
    prompt: str
    max_new_tokens: int = 512


# Global service instances (will be initialized with model/tokenizer)
recipe_service: Optional[RecipeAnalysisService] = None
model = None
tokenizer = None
device = None

def initialize_services(loaded_model, loaded_tokenizer, loaded_device=None):
    """Initialize all recipe services with model and tokenizer"""
    global recipe_service, model, tokenizer, device
    recipe_service = RecipeAnalysisService(loaded_model, loaded_tokenizer)
    model = loaded_model
    tokenizer = loaded_tokenizer
    device = loaded_device

@router.post("/analyze")
async def analyze_recipe(request: RecipeAnalysisRequest):
    """
    Analyze a recipe to determine estimated cooking time and difficulty level
    """
    try:
        logger.info(f"Analyzing recipe: {request.title}")
        
        if recipe_service is None:
            raise HTTPException(status_code=503, detail="Recipe service not initialized")
        
        # Prepare recipe data in the format expected by the service
        recipe_data = {
            "title": request.title,
            "description": request.description or "",
            "ingredients": request.ingredients,
            "instructions": request.instructions
        }
        
        # Analyze the recipe
        analysis_result = await recipe_service.analyze_recipe(recipe_data)
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error analyzing recipe: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recipe analysis failed: {str(e)}")

async def _run_import_job(job_id: str, url: str) -> None:
    """Background task: run import and update job state."""
    global _import_jobs
    job = _import_jobs.get(job_id)
    if not job:
        return
    try:
        job["status"] = "processing"
        job["started_at"] = time.time()
        job["step"] = "starting"
        job["updated_at"] = time.time()
        logger.info(f"[import] Job {job_id} processing URL: {url}")
        sig = inspect.signature(import_recipe_from_url)
        if "progress" in sig.parameters:
            result = await import_recipe_from_url(url, model, tokenizer, device, progress=job)
        else:
            result = await import_recipe_from_url(url, model, tokenizer, device)
        job["status"] = "completed"
        job["step"] = "completed"
        job["updated_at"] = time.time()
        job["result"] = result.model_dump() if hasattr(result, "model_dump") else result.dict()
        logger.info(f"[import] Job {job_id} completed (title: {result.title[:50]!r}...)")
    except Exception as e:
        logger.error(f"[import] Job {job_id} failed: {str(e)}", exc_info=True)
        job["status"] = "failed"
        job["step"] = "failed"
        job["updated_at"] = time.time()
        job["error"] = str(e)


async def _run_video_import_job(job_id: str, url: str) -> None:
    """Background task: run video import and update job state."""
    global _video_import_jobs
    job = _video_import_jobs.get(job_id)
    if not job:
        return
    try:
        job["status"] = "processing"
        job["started_at"] = time.time()
        job["step"] = "starting"
        job["updated_at"] = time.time()
        logger.info(f"[import-video] Job {job_id} processing URL: {url}")
        result = await import_recipe_from_video_url(job_id, url, model, tokenizer, device, progress=job)
        job["status"] = "completed"
        job["step"] = "completed"
        job["updated_at"] = time.time()
        job["result"] = result.model_dump() if hasattr(result, "model_dump") else result.dict()
        logger.info(f"[import-video] Job {job_id} completed (title: {result.title[:50]!r}...)")
    except Exception as e:
        logger.error(f"[import-video] Job {job_id} failed: {str(e)}", exc_info=True)
        job["status"] = "failed"
        job["step"] = "failed"
        job["updated_at"] = time.time()
        job["error"] = str(e)


@router.post("/import", dependencies=[Depends(verify_internal_access)])
async def import_recipe(body: ImportRecipeRequest = Body(...)):
    """
    Start an async import from an external URL. Returns jobId immediately.
    Poll GET /recipe/import/status/{jobId} for status and result.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    job_id = str(uuid.uuid4())
    _import_jobs[job_id] = {
        "status": "pending",
        "url": body.url,
        "created_at": time.time(),
        "step": "pending",
    }
    asyncio.create_task(_run_import_job(job_id, body.url))
    logger.info(f"[import] Started job {job_id} for URL: {body.url}")
    return {"jobId": job_id, "status": "pending"}


@router.get("/import/status/{job_id}", dependencies=[Depends(verify_internal_access)])
async def get_import_status(job_id: str):
    """Get status of an import job. Returns status, step, timestamps, and result/error when done."""
    if job_id not in _import_jobs:
        raise HTTPException(status_code=404, detail="Import job not found")
    job = _import_jobs[job_id]
    out = {
        "status": job["status"],
        "url": job.get("url"),
        "step": job.get("step", "unknown"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "updated_at": job.get("updated_at"),
    }
    if job.get("started_at") and job.get("updated_at"):
        out["elapsed_seconds"] = round(job["updated_at"] - job["started_at"], 1)
    if job.get("result") is not None:
        out["result"] = job["result"]
    if job.get("error") is not None:
        out["error"] = job["error"]
    return out


@router.post("/import-video", dependencies=[Depends(verify_internal_access)])
async def import_recipe_video(body: ImportRecipeRequest = Body(...)):
    """
    Start an async import from a video URL (YouTube, Instagram, TikTok, etc.).
    Downloads description and subtitle; if no subtitle, downloads audio and transcribes with Whisper.
    Returns jobId immediately. Poll GET /recipe/import-video/status/{jobId} for status and result.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    job_id = str(uuid.uuid4())
    _video_import_jobs[job_id] = {
        "status": "pending",
        "url": body.url,
        "created_at": time.time(),
        "step": "pending",
    }
    asyncio.create_task(_run_video_import_job(job_id, body.url))
    logger.info(f"[import-video] Started job {job_id} for URL: {body.url}")
    return {"jobId": job_id, "status": "pending"}


@router.get("/import-video/status/{job_id}", dependencies=[Depends(verify_internal_access)])
async def get_video_import_status(job_id: str):
    """Get status of a video import job. Returns status, step, timestamps, and result/error when done."""
    if job_id not in _video_import_jobs:
        raise HTTPException(status_code=404, detail="Video import job not found")
    job = _video_import_jobs[job_id]
    out = {
        "status": job["status"],
        "url": job.get("url"),
        "step": job.get("step", "unknown"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "updated_at": job.get("updated_at"),
    }
    if job.get("started_at") and job.get("updated_at"):
        out["elapsed_seconds"] = round(job["updated_at"] - job["started_at"], 1)
    if job.get("result") is not None:
        out["result"] = job["result"]
    if job.get("error") is not None:
        out["error"] = job["error"]
    return out


@router.post("/auto-category")
async def auto_category(request: AutoCategoryRequest):
    """
    Automatically categorize a recipe based on its content
    """
    try:
        logger.info("Auto-categorizing recipe")
        
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        # Create a prompt for auto-categorization
        prompt = f"""Analyze this recipe and suggest appropriate categories:

Title: {request.title or 'N/A'}
Description: {request.description or 'N/A'}
Ingredients: {request.ingredients or 'N/A'}
Instructions: {request.instructions or 'N/A'}

Please suggest 3-5 relevant categories for this recipe. Return the response as a JSON object with a "categories" field containing an array of category names.

Example response:
{{"categories": ["Main Course", "Italian", "Pasta", "Vegetarian"]}}"""
        
        # Use the chat service to generate response (not async)
        response = generate_chat_response(prompt, model, tokenizer, device)
        
        # Parse the response
        import json
        from utils.json_parser import extract_json_from_markdown
        
        json_str = extract_json_from_markdown(response)
        if json_str:
            category_result = json.loads(json_str)
        else:
            # Fallback if JSON parsing fails
            category_result = {"categories": ["General"]}
        
        return category_result
        
    except Exception as e:
        logger.error(f"Error auto-categorizing recipe: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Auto-categorization failed: {str(e)}")

@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat with the AI about recipes
    """
    try:
        logger.info(f"Chat request: {request.question[:50]}...")
        
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        # Get chat response using the existing function (not async)
        chat_response = generate_chat_response(request.question, model, tokenizer, device)
        
        return {"response": chat_response}
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@router.post("/debug-prompt", dependencies=[Depends(verify_internal_access)])
async def debug_prompt(body: DebugPromptRequest):
    """
    Send a raw prompt to the model and get the exact decoded output.
    Use this to test prompts and see what the model actually returns (including any instruction echo).
    The model returns the full sequence (prompt + generated tokens); there is no separate reasoning field.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    device_val = device if device is not None else (next(model.parameters()).device)

    def _generate(prompt: str, max_new_tokens: int):
        inputs = tokenizer(prompt, return_tensors="pt").to(device_val)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        full_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_decoded

    try:
        raw_output = await run_llm_in_thread(_generate, body.prompt, body.max_new_tokens)
        # What we'd normally treat as "the model's answer" (content after the prompt)
        if raw_output.startswith(body.prompt):
            generated_only = raw_output[len(body.prompt):].strip()
        else:
            generated_only = raw_output
        return {
            "raw_output": raw_output,
            "generated_only": generated_only,
            "prompt_length": len(body.prompt),
            "raw_output_length": len(raw_output),
            "note": "Full sequence is prompt + generated. No separate reasoning field; any reasoning is in the text.",
        }
    except Exception as e:
        logger.exception("debug-prompt failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def recipe_health():
    """Check recipe services health"""
    try:
        services_status = {
            "recipe_analysis": recipe_service is not None,
            "import": model is not None and tokenizer is not None,
            "chat": model is not None and tokenizer is not None
        }
        
        all_healthy = all(services_status.values())
        
        return {
            "status": "OK" if all_healthy else "PARTIAL",
            "service": "Recipe Services",
            "details": services_status
        }
    except Exception as e:
        logger.error(f"Recipe services health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recipe services unhealthy: {str(e)}")
