"""
Recipe API endpoints for AI Service
Handles recipe analysis, import, categorization, and chat
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging

from services.recipe_analysis_service import RecipeAnalysisService
from services.import_service import extract_recipe_with_llm, ImportRecipeRequest, ImportRecipeResponse
from services.chat_service import generate_chat_response, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

# Create router for recipe endpoints
router = APIRouter(prefix="/recipe", tags=["recipe"])

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

@router.post("/import")
async def import_recipe(request: ImportRecipeRequest):
    """
    Import a recipe from an external URL
    """
    try:
        logger.info(f"Importing recipe from: {request.url}")
        
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        # Import the recipe using the existing function
        imported_data = await extract_recipe_with_llm(request.url, model, tokenizer, device)
        
        return imported_data
        
    except Exception as e:
        logger.error(f"Error importing recipe: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recipe import failed: {str(e)}")

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
