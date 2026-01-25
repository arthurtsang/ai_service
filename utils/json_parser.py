"""
JSON parsing utilities for handling LLM responses with various parsing strategies.
"""
import json
import torch


def parse_llm_response(response_text, model, tokenizer, device):
    """
    Parse LLM response with multiple fallback strategies.
    
    Returns:
        dict: Parsed data, empty dict if all parsing methods fail
    """
    # Try to parse the entire response as JSON first
    try:
        data = json.loads(response_text)
        print(f"[json-parser] Parsed data directly: {data}")
        return data
    except json.JSONDecodeError as e:
        print(f"[json-parser] JSON decode error: {e}")
        return _parse_with_fallbacks(response_text, model, tokenizer, device)


def _parse_with_fallbacks(response_text, model, tokenizer, device):
    """Apply cascading fallback parsing strategies."""
    # First try to extract JSON by finding balanced braces (handles extra content after JSON)
    data = _extract_balanced_json(response_text)
    
    # If balanced braces didn't work, try LLM completion
    if not data and response_text.strip().startswith('{'):
        data = _llm_json_completion(response_text, model, tokenizer, device)
    
    # If still no data, log the failure
    if not data:
        print(f"[json-parser] All JSON parsing methods failed")
    
    return data


def _extract_balanced_json(response_text):
    """Extract JSON by finding balanced braces to handle extra content after JSON."""
    json_start = response_text.find('{')
    
    if json_start == -1:
        print(f"[json-parser] No JSON structure found in response")
        return {}
    
    brace_count = 0
    json_end = json_start
    
    for i, char in enumerate(response_text[json_start:], json_start):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                json_end = i
                break
    
    if brace_count == 0:  # Found balanced JSON
        json_str = response_text[json_start:json_end + 1]
        try:
            data = json.loads(json_str)
            print(f"[json-parser] Parsed data with balanced braces: {data}")
            return data
        except json.JSONDecodeError as e:
            print(f"[json-parser] Balanced JSON still invalid: {e}")
            return {}
    else:
        print(f"[json-parser] Unbalanced braces in response")
        return {}


def _llm_json_completion(response_text, model, tokenizer, device):
    """Use LLM to fix and complete malformed JSON."""
    print(f"[json-parser] Attempting to complete/fix JSON with LLM...")
    
    completion_prompt = (
        "Fix and complete this JSON to make it valid. "
        "Return only the corrected JSON:\n\n"
        f"{response_text}\n\n"
        "Fixed JSON:"
    )
    
    completion_inputs = tokenizer(completion_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        completion_outputs = model.generate(
            **completion_inputs, 
            max_new_tokens=200, 
            pad_token_id=tokenizer.eos_token_id
        )
    completion_text = tokenizer.decode(completion_outputs[0], skip_special_tokens=True)
    completion_response = completion_text[len(completion_prompt):].strip()
    
    print(f"[json-parser] LLM fix response: {completion_response}")
    
    # Try to extract JSON from the completion response using balanced braces
    return _extract_balanced_json(completion_response)


def extract_json_from_markdown(markdown_text):
    """Extract structured data from markdown format as final fallback."""
    import re
    
    data = {}
    
    # Extract title
    title_match = re.search(r'# (.+?)(?:\n|$)', markdown_text)
    if title_match:
        data["title"] = title_match.group(1).strip()
    
    # Extract description
    description_match = re.search(r'## Description\n(.+?)(?:\n## |$)', markdown_text, re.DOTALL)
    if description_match:
        data["description"] = description_match.group(1).strip()
    
    # Extract ingredients (more flexible pattern)
    ingredients_match = re.search(r'## Ingredients\n(.+?)(?:\n## |$)', markdown_text, re.DOTALL)
    if not ingredients_match:
        # Try alternative patterns
        ingredients_match = re.search(r'Ingredients:\n(.+?)(?:\nInstructions:|$)', markdown_text, re.DOTALL)
    if not ingredients_match:
        ingredients_match = re.search(r'Ingredients\n(.+?)(?:\nInstructions|$)', markdown_text, re.DOTALL)
    
    if ingredients_match:
        ingredients_text = ingredients_match.group(1).strip()
        ingredients_list = [
            line.strip('- ').strip() 
            for line in ingredients_text.split('\n') 
            if line.strip().startswith('-')
        ]
        data["ingredients"] = ingredients_list
    
    # Extract instructions (more flexible pattern)
    instructions_match = re.search(r'## Instructions\n(.+?)(?:\n## |$)', markdown_text, re.DOTALL)
    if not instructions_match:
        # Try alternative patterns
        instructions_match = re.search(r'Instructions:\n(.+?)(?:\n## |$)', markdown_text, re.DOTALL)
    if not instructions_match:
        instructions_match = re.search(r'Instructions\n(.+?)(?:\n## |$)', markdown_text, re.DOTALL)
    
    if instructions_match:
        instructions_text = instructions_match.group(1).strip()
        instructions_list = []
        for line in instructions_text.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line):
                instructions_list.append(re.sub(r'^\d+\.\s*', '', line))
        data["instructions"] = instructions_list
    
    # Extract image URL
    image_match = re.search(r'## Image\n(.+?)(?:\n|$)', markdown_text)
    if image_match:
        data["imageUrl"] = image_match.group(1).strip()
    
    return data 