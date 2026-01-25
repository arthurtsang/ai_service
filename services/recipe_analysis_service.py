import asyncio
import json
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.json_parser import extract_json_from_markdown

class RecipeAnalysisService:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    async def analyze_recipe(self, recipe_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a recipe to determine estimated cooking time and difficulty level.
        Returns a dictionary with 'estimatedTime', 'difficulty', 'timeReasoning', 'difficultyReasoning', and 'description' fields.
        """
        try:
            # Prepare the recipe data for analysis
            title = recipe_data.get('title', '')
            ingredients = recipe_data.get('ingredients', '')
            instructions = recipe_data.get('instructions', '')
            description = recipe_data.get('description', '')
            estimated_time = recipe_data.get('estimatedTime', '')
            difficulty = recipe_data.get('difficulty', '')

            # Skip analysis if time and difficulty are already set
            if estimated_time and estimated_time != 'Pending...' and difficulty and difficulty != 'Undetermined':
                print(f"[recipe-analysis] Skipping analysis - recipe already has time ({estimated_time}) and difficulty ({difficulty})")
                return {
                    'estimatedTime': estimated_time,
                    'difficulty': difficulty,
                    'timeReasoning': 'Already set by user or import',
                    'difficultyReasoning': 'Already set by user or import',
                    'description': description
                }

            # Create the analysis prompt
            prompt = self._create_analysis_prompt(title, description, ingredients, instructions)
            
            # Get LLM response
            response = await self._get_llm_response(prompt)
            
            # Parse the response
            analysis = self._parse_analysis_response(response)
            
            # Generate description if empty
            if not description or description.strip() == '':
                description_prompt = self._create_description_prompt(title, ingredients, instructions)
                description_response = await self._get_llm_response(description_prompt)
                analysis['description'] = self._parse_description_response(description_response)
            
            return analysis
            
        except Exception as e:
            print(f"[recipe-analysis] Error analyzing recipe: {e}")
            return {
                'estimatedTime': '30',
                'difficulty': 'Medium',
                'timeReasoning': '',
                'difficultyReasoning': ''
            }

    def _create_description_prompt(self, title: str, ingredients: str, instructions: str) -> str:
        """
        Create a prompt for generating recipe descriptions.
        """
        prompt = f"""You are an expert food writer. Write a brief, appetizing description for this recipe.

Recipe Title: {title}

Ingredients:
{ingredients}

Instructions:
{instructions}

Write a concise, engaging description (2-3 sentences) that captures the essence of this dish. Focus on:
- What makes this recipe special
- Key flavors or techniques
- What type of meal it is (breakfast, lunch, dinner, dessert, etc.)
- Any notable characteristics

Provide only the description text, no additional formatting.

IMPORTANT: End your response with '---END---' to indicate completion:"""

        return prompt

    def _parse_description_response(self, response: str) -> str:
        """
        Parse the LLM response to extract the description.
        """
        try:
            # Clean up the response
            description = response.strip()
            # Remove any markdown formatting
            description = description.replace('**', '').replace('*', '')
            # Limit length
            if len(description) > 500:
                description = description[:497] + '...'
            return description
        except Exception as e:
            print(f"[recipe-analysis] Error parsing description response: {e}")
            return ''

    def _create_analysis_prompt(self, title: str, description: str, ingredients: str, instructions: str) -> str:
        """
        Create a comprehensive prompt for recipe analysis.
        """
        prompt = f"""You are an expert cooking instructor and recipe analyst. Analyze the following recipe and provide consistent, accurate assessments.

Recipe Title: {title}
Description: {description}

Ingredients:
{ingredients}

Instructions:
{instructions}

Please analyze this recipe and provide your assessment in the following JSON format:

{{
  "estimatedTime": "time-range",
  "difficulty": "difficulty-level",
  "timeReasoning": "detailed explanation of time estimation",
  "difficultyReasoning": "detailed explanation of difficulty assessment"
}}

For estimatedTime, provide a single approximate time in minutes:
- Return only the number of minutes as a string (e.g., "25", "45", "120")
- Consider total time including prep, cooking, and any waiting time
- Examples: "15" (very quick), "30" (simple), "45" (moderate), "90" (complex), "180" (slow cooker)
- Do not include "mins" or "minutes" - just the number

Consider:
- Oven preheating time (usually 10-15 minutes)
- Parallel tasks (chopping while oven heats, etc.)
- Actual cooking time vs. total time
- Resting/cooling time if mentioned

For difficulty, assess based on required cooking skills:
- "Easy": Basic skills (boiling, simple chopping, following basic instructions)
- "Medium": Intermediate skills (proper knife work, timing, temperature control)
- "Advanced": Expert skills (complex techniques, precision, multi-tasking)

Consider:
- Knife skills required
- Temperature control needs
- Timing sensitivity
- Multi-tasking requirements
- Technique complexity
- Equipment needs

Provide your analysis in valid JSON format.

IMPORTANT: End your response with '---END---' to indicate completion:"""

        return prompt

    async def _get_llm_response(self, prompt: str) -> str:
        """
        Get response from the LLM model.
        """
        try:
            # Tokenize the input
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
            
            # Move inputs to the same device as the model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=2048,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            response = response[len(prompt):].strip()
            
            print(f"[recipe-analysis] LLM Response: {response}")
            return response
            
        except Exception as e:
            print(f"[recipe-analysis] Error getting LLM response: {e}")
            raise

    def _parse_analysis_response(self, response: str) -> Dict[str, str]:
        """
        Parse the LLM response to extract estimated time and difficulty.
        """
        try:
            # Try to extract JSON from the response
            json_data = extract_json_from_markdown(response)
            
            if json_data:
                estimated_time = json_data.get('estimatedTime', '30-45 min')
                difficulty = json_data.get('difficulty', 'Medium')
                
                # Validate the values - check if it's a valid number of minutes
                def is_valid_time(time_str):
                    if not time_str or not isinstance(time_str, str):
                        return False
                    try:
                        minutes = int(time_str)
                        # Check if it's a reasonable number of minutes (1-480 = 1 min to 8 hours)
                        return 1 <= minutes <= 480
                    except:
                        return False
                
                valid_times = is_valid_time
                valid_difficulties = ['Easy', 'Medium', 'Advanced']
                
                if not valid_times(estimated_time):
                    print(f"[recipe-analysis] Invalid estimatedTime: {estimated_time}, using default")
                    estimated_time = '30'
                
                if difficulty not in valid_difficulties:
                    difficulty = 'Medium'
                
                print(f"[recipe-analysis] Successfully parsed: {estimated_time}, {difficulty}")
                return {
                    'estimatedTime': estimated_time,
                    'difficulty': difficulty,
                    'timeReasoning': json_data.get('timeReasoning', ''),
                    'difficultyReasoning': json_data.get('difficultyReasoning', '')
                }
            
            # Fallback if JSON parsing fails - try manual extraction
            print(f"[recipe-analysis] JSON parsing failed, trying manual extraction from: {response}")
            
            # Look for estimatedTime and difficulty in the response
            import re
            
            # Extract estimatedTime
            time_match = re.search(r'"estimatedTime":\s*"([^"]+)"', response)
            estimated_time = time_match.group(1) if time_match else '30'
            
            # Extract difficulty
            difficulty_match = re.search(r'"difficulty":\s*"([^"]+)"', response)
            difficulty = difficulty_match.group(1) if difficulty_match else 'Medium'
            
            # Validate the values - check if it's a valid number of minutes
            def is_valid_time(time_str):
                if not time_str or not isinstance(time_str, str):
                    return False
                try:
                    minutes = int(time_str)
                    # Check if it's a reasonable number of minutes (1-480 = 1 min to 8 hours)
                    return 1 <= minutes <= 480
                except:
                    return False
            
            valid_times = is_valid_time
            valid_difficulties = ['Easy', 'Medium', 'Advanced']
            
            if not valid_times(estimated_time):
                print(f"[recipe-analysis] Invalid estimatedTime: {estimated_time}, using default")
                estimated_time = '30'
            
            if difficulty not in valid_difficulties:
                difficulty = 'Medium'
            
            print(f"[recipe-analysis] Manual extraction successful: {estimated_time}, {difficulty}")
            return {
                'estimatedTime': estimated_time,
                'difficulty': difficulty,
                'timeReasoning': '',
                'difficultyReasoning': ''
            }
            
        except Exception as e:
            print(f"[recipe-analysis] Error parsing analysis response: {e}")
            return {
                'estimatedTime': '30',
                'difficulty': 'Medium',
                'timeReasoning': '',
                'difficultyReasoning': ''
            }

# Global instance
recipe_analysis_service = None

def get_recipe_analysis_service(model, tokenizer) -> RecipeAnalysisService:
    global recipe_analysis_service
    if recipe_analysis_service is None:
        recipe_analysis_service = RecipeAnalysisService(model, tokenizer)
    return recipe_analysis_service 