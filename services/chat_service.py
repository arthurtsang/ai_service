"""
Chat service for handling recipe-related conversations.
"""
import torch
from pydantic import BaseModel
from typing import List, Any
import re
from utils.recipe_api import fetch_relevant_recipes


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    recipes: List[Any] = []


def extract_keywords_from_question(question: str) -> List[str]:
    """Extract relevant keywords from the user's question for recipe search."""
    # Convert to lowercase and remove punctuation
    question_lower = re.sub(r'[^\w\s]', ' ', question.lower())
    
    # Common cooking-related keywords to look for
    cooking_keywords = [
        'recipe', 'cook', 'make', 'prepare', 'dish', 'meal', 'food',
        'breakfast', 'lunch', 'dinner', 'snack', 'dessert', 'appetizer',
        'soup', 'salad', 'pasta', 'rice', 'chicken', 'beef', 'fish',
        'vegetarian', 'vegan', 'gluten-free', 'quick', 'easy', 'healthy',
        'spicy', 'sweet', 'savory', 'bake', 'grill', 'fry', 'steam'
    ]
    
    # Extract words that match cooking keywords or are likely ingredients
    words = question_lower.split()
    keywords = [word for word in words if word in cooking_keywords or len(word) > 3]
    
    # If no specific keywords found, use the most common words
    if not keywords:
        keywords = [word for word in words if len(word) > 3][:5]
    
    return keywords[:5]  # Limit to 5 keywords


def generate_chat_response(question: str, model, tokenizer, device) -> str:
    """Generate a chat response using the LLM."""
    print(f"[chat] Generating response for: {question}")
    
    # Create a prompt for recipe assistance with instructions to query recipes
    prompt = (
        "You are a helpful cooking assistant. Answer questions about recipes, "
        "cooking techniques, ingredients, and meal planning. "
        "Be friendly, informative, and concise.\n\n"
        "IMPORTANT: If the user is asking about specific recipes, ingredients, or meal suggestions, "
        "you should query the recipe database to find relevant recipes. "
        "When you find relevant recipes, mention them in your response and provide brief details.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            do_sample=True
        )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response_text[len(prompt):].strip()
    
    print(f"[chat] Generated answer: {answer[:100]}...")
    
    return answer


async def retrieve_relevant_recipes(question: str) -> List[Any]:
    """Retrieve relevant recipes based on the question."""
    try:
        print(f"[chat] Retrieving recipes for: {question}")
        
        # Extract keywords from the question
        keywords = extract_keywords_from_question(question)
        print(f"[chat] Extracted keywords: {keywords}")
        
        if not keywords:
            print("[chat] No relevant keywords found, skipping recipe search")
            return []
        
        # Query the recipe database
        recipes = await fetch_relevant_recipes(keywords)
        print(f"[chat] Found {len(recipes)} relevant recipes")
        
        return recipes
        
    except Exception as e:
        print(f"[chat] Error retrieving recipes: {e}")
        return []


async def process_chat_request(request: ChatRequest, model, tokenizer, device) -> ChatResponse:
    """Process a chat request and return response."""
    try:
        # Retrieve relevant recipes (if any)
        recipes = await retrieve_relevant_recipes(request.question)
        
        # Generate response
        answer = generate_chat_response(request.question, model, tokenizer, device)
        
        # If we found recipes, enhance the response to mention them
        if recipes:
            recipe_mentions = []
            for i, recipe in enumerate(recipes[:3]):  # Limit to top 3 recipes
                recipe_mentions.append(f"- {recipe.get('title', 'Unknown Recipe')}")
                if recipe.get('description'):
                    recipe_mentions.append(f"  {recipe['description'][:100]}...")
            
            if recipe_mentions:
                answer += f"\n\nI found some relevant recipes in our database:\n" + "\n".join(recipe_mentions)
        
        return ChatResponse(answer=answer, recipes=recipes)
        
    except Exception as e:
        print(f"[chat] Error processing request: {e}")
        import traceback
        traceback.print_exc()
        
        return ChatResponse(
            answer="I'm sorry, I encountered an error while processing your request. Please try again.",
            recipes=[]
        ) 