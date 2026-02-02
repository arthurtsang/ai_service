"""
AI Service FastAPI Application
Provides AI-powered services including recipe analysis and tournament scheduling
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn

# Import API routers
from api.recipe_api import router as recipe_router, initialize_services as initialize_recipe_services
from api.kungfu_api import router as kungfu_router, initialize_service as initialize_kungfu_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Service",
    description="AI-powered services for recipe analysis and tournament scheduling",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer (will be loaded from existing recipe service)
model = None
tokenizer = None

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "OK", "service": "AI Service"}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "AI Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recipe_analysis": "/recipe/analyze",
            "recipe_import": "/recipe/import", 
            "recipe_auto_category": "/recipe/auto-category",
            "recipe_chat": "/recipe/chat",
            "recipe_health": "/recipe/health",
            "tournament_scheduling": "/kungfu/schedule",
            "tournament_health": "/kungfu/health"
        }
    }

# Legacy endpoints for backward compatibility
@app.post("/analyze-recipe")
async def legacy_analyze_recipe(request):
    """Legacy endpoint - redirects to new recipe API"""
    from fastapi import Request
    from fastapi.responses import RedirectResponse
    
    # This will be handled by the recipe router
    return await recipe_router.routes[0].endpoint(request)

@app.post("/import-recipe")
async def legacy_import_recipe(request: Request):
    """Legacy endpoint - start async import, return jobId. Poll GET /import-recipe/status/{jobId} for result."""
    body = await request.json()
    from api.recipe_api import ImportRecipeRequest
    parsed = ImportRecipeRequest(**body)
    return await recipe_router.routes[1].endpoint(parsed)


@app.get("/import-recipe/status/{job_id}")
async def legacy_import_status(job_id: str):
    """Legacy endpoint - get import job status (pending|processing|completed|failed)."""
    from api.recipe_api import get_import_status
    return await get_import_status(job_id)

@app.post("/auto-category")
async def legacy_auto_category(request):
    """Legacy endpoint - redirects to new recipe API"""
    from fastapi import Request
    from fastapi.responses import RedirectResponse
    
    # This will be handled by the recipe router
    return await recipe_router.routes[2].endpoint(request)

@app.post("/chat")
async def legacy_chat(request):
    """Legacy endpoint - redirects to new recipe API"""
    from fastapi import Request
    from fastapi.responses import RedirectResponse
    
    # This will be handled by the recipe router
    return await recipe_router.routes[3].endpoint(request)

# Include routers
app.include_router(recipe_router)
app.include_router(kungfu_router)

# Startup event to initialize services
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        logger.info("Starting AI Service...")
        
        # Try to load model and tokenizer from existing recipe service
        try:
            from models.model_loader import load_model_and_tokenizer
            model, tokenizer = load_model_and_tokenizer()
            logger.info("Successfully loaded model and tokenizer")
            
            # Get device info
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Initialize all services
            initialize_recipe_services(model, tokenizer, device)
            initialize_kungfu_service(model, tokenizer)
            
            logger.info("All services initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not load model and tokenizer: {e}")
            logger.warning("Services will be initialized without model (mock responses)")
            
            # Initialize services without model (will use mock responses)
            initialize_recipe_services(None, None, None)
            initialize_kungfu_service(None, None)
            
    except Exception as e:
        logger.error(f"Error during startup: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "app_new:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
