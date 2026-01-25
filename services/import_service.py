"""
Recipe import service for extracting recipe data from web pages.
"""
import asyncio
import json
import re
import time
import torch
from typing import Dict, Any, List
from bs4 import BeautifulSoup, Comment
from playwright.async_api import async_playwright
from pydantic import BaseModel
from utils.json_parser import parse_llm_response, extract_json_from_markdown


class ImportRecipeRequest(BaseModel):
    url: str


class ImportRecipeResponse(BaseModel):
    title: str
    description: str
    ingredients: str
    instructions: str
    imageUrl: str = ""
    tags: List[str] = []
    cookTime: str = "Pending..."
    difficulty: str = "Undetermined"
    timeReasoning: str = ""
    difficultyReasoning: str = ""


def clean_html_for_recipe_extraction(soup: BeautifulSoup) -> str:
    """
    Clean HTML to focus on recipe content by extracting text with image URLs.
    Simple and fast approach that preserves important recipe information.
    """
    # Create a copy to avoid modifying the original
    soup_clean = BeautifulSoup(str(soup), 'html.parser')
    
    # Remove script and style elements
    for element in soup_clean(["script", "style", "noscript"]):
        element.decompose()
    
    # Remove HTML comments
    for comment in soup_clean.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    # Remove obvious navigation and advertising elements
    remove_selectors = [
        'nav', 'header', 'footer',
        '.nav', '.navigation', '.header', '.footer',
        '.breadcrumb', '.breadcrumbs', '.pagination', '.pager',
        '.advertisement', '.ads', '.ad',
        '.social-share', '.share-buttons',
        '.comments-section', '.user-reviews-section',
        '.related-recipes', '.more-recipes',
        '.newsletter', '.popup', '.modal',
        'iframe', 'svg', 'defs', 'symbol'
    ]
    
    for selector in remove_selectors:
        for element in soup_clean.select(selector):
            element.decompose()
    
    # Extract image URLs and add them to the text
    image_texts = []
    for img in soup_clean.find_all('img'):
        src = img.get('src', '')
        alt = img.get('alt', '')
        if src:
            image_text = f"Image: {src}"
            if alt:
                image_text += f" (alt: {alt})"
            image_texts.append(image_text)
    
    # Extract text content
    text = soup_clean.get_text(separator='\n', strip=True)
    
    # Add image information to the text
    if image_texts:
        text = "Images:\n" + "\n".join(image_texts) + "\n\n" + text
    
    # Clean up the text
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and len(line) > 2:  # Skip very short lines
            # Skip lines that are just repeated navigation text
            if line.lower() in ['home', 'recipes', 'search', 'login', 'sign up', 'subscribe']:
                continue
            cleaned_lines.append(line)
    
    # Join lines and remove excessive whitespace
    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)  # Remove excessive line breaks
    
    # Safety check: if we removed too much content, fall back to original text
    if len(cleaned_text) < 100:  # If cleaned text is too short
        print(f"[import-recipe] Warning: Cleaned text too short ({len(cleaned_text)} chars), using original text")
        original_text = soup.get_text(separator="\n", strip=True)
        # Just do basic cleaning on original text
        lines = original_text.split('\n')
        basic_cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:
                basic_cleaned_lines.append(line)
        cleaned_text = '\n'.join(basic_cleaned_lines)
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    
    # Limit text length to speed up LLM processing
    if len(cleaned_text) > 8000:
        print(f"[import-recipe] Text too long ({len(cleaned_text)} chars), truncating to 8000 chars")
        cleaned_text = cleaned_text[:8000] + "\n\n[Content truncated for processing speed]"
    
    print(f"[import-recipe] Cleaned text length: {len(cleaned_text)} characters")
    print(f"[import-recipe] First 500 chars of cleaned text: {cleaned_text[:500]}")
    
    return cleaned_text


async def fetch_html_with_playwright(url: str) -> str:
    """Fetch HTML using Playwright headless browser to get fully rendered content."""
    try:
        async with async_playwright() as p:
            # Launch browser in headless mode with SSL certificate validation disabled
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--ignore-ssl-errors=yes',
                    '--ignore-certificate-errors=yes',
                    '--ignore-certificate-errors-spki-list=yes',
                    '--ignore-ssl-errors-skip-list=yes',
                    '--disable-web-security',
                    '--allow-running-insecure-content'
                ]
            )
            context = await browser.new_context(ignore_https_errors=True)
            page = await context.new_page()
            
            # Set a realistic user agent to avoid detection
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            })
            
            # Navigate to page with faster timeouts
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=10000)
                # Try to wait for network to be mostly idle, but don't fail if it times out
                try:
                    await page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    print("[import-recipe] Network idle timeout, continuing anyway")
                    pass
            except Exception as e:
                print(f"[import-recipe] Page load timeout, trying to get content anyway: {e}")
                # If even domcontentloaded fails, try a basic load
                await page.goto(url, wait_until="load", timeout=8000)
            
            # Wait shorter for any lazy-loaded images and JavaScript to execute
            print("[import-recipe] Waiting for JavaScript and lazy-loaded images...")
            await page.wait_for_timeout(3000)  # Reduced from 8000ms to 3000ms
            
            # Try to trigger any lazy loading by scrolling
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1000)  # Reduced from 2000ms to 1000ms
            
            # Get the fully rendered HTML
            html = await page.content()
            print(f"[import-recipe] HTML length: {len(html)} characters")
            
            await browser.close()
            return html
            
    except Exception as e:
        print(f"[import-recipe] Playwright error: {e}")
        # If Playwright fails, we could fallback to requests here if needed
        raise e


def extract_images_from_html(soup, base_url):
    """Extract and process image URLs from HTML."""
    from urllib.parse import urlparse, urljoin
    
    image_urls = []
    
    # Debug: Print all img tags found
    all_imgs = soup.find_all('img')
    print(f"[import-recipe] Found {len(all_imgs)} img tags in HTML")
    for i, img in enumerate(all_imgs[:5]):  # Show first 5 for debugging
        print(f"[import-recipe] img[{i}]: src='{img.get('src')}', data-src='{img.get('data-src')}', class='{img.get('class')}'")
    
    for img in soup.find_all('img'):
        src = img.get('src')
        srcset = img.get('srcset')
        data_src = img.get('data-src')
        data_original_src = img.get('data-original-src')
        data_lazy_src = img.get('data-lazy-src')
        data_pin_media = img.get('data-pin-media')
        
        # Try different image source attributes (now including more modern lazy-loading attributes)
        img_src = src or data_src or data_original_src or data_lazy_src or data_pin_media
        if img_src:
            print(f"[import-recipe] Processing img_src: '{img_src}'")
            
            # Convert relative URLs to absolute
            if img_src.startswith('/'):
                parsed = urlparse(base_url)
                base_domain = f"{parsed.scheme}://{parsed.netloc}"
                img_src = base_domain + img_src
                print(f"[import-recipe] Converted to absolute: '{img_src}'")
            elif img_src.startswith('http'):
                print(f"[import-recipe] Already absolute: '{img_src}'")
                pass  # Already absolute
            else:
                # Relative URL, prepend base URL
                img_src = urljoin(base_url, img_src)
                print(f"[import-recipe] Joined with base: '{img_src}'")
            
            # Skip obvious non-recipe images - be more precise about matching
            skip_keywords = ['logo', 'icon', 'avatar', 'social', 'ad', 'advertisement']
            url_lower = img_src.lower()
            
            # Check if any skip keyword appears as a standalone word or at word boundaries
            should_skip = False
            for keyword in skip_keywords:
                # Check for keyword as standalone word (surrounded by non-letters)
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, url_lower):
                    should_skip = True
                    print(f"[import-recipe] Skip keyword '{keyword}' found in URL")
                    break
            
            print(f"[import-recipe] Skip check - URL: '{img_src}', should_skip: {should_skip}")
            
            if not should_skip:
                image_urls.append(img_src)
                print(f"[import-recipe] Added image: {img_src}")
            else:
                print(f"[import-recipe] Skipped image: {img_src}")
        else:
            print(f"[import-recipe] No img_src found for this img tag")
        
        # Also check srcset for higher quality images
        if srcset:
            # Parse srcset format: "url1 1x, url2 2x, ..."
            srcset_urls = []
            for src_desc in srcset.split(','):
                src_desc = src_desc.strip()
                if ' ' in src_desc:
                    src_url = src_desc.split(' ')[0]
                    if src_url.startswith('http') or src_url.startswith('/'):
                        if src_url.startswith('/'):
                            parsed = urlparse(base_url)
                            base_domain = f"{parsed.scheme}://{parsed.netloc}"
                            src_url = base_domain + src_url
                        srcset_urls.append(src_url)
            
            # Add the highest resolution image from srcset
            if srcset_urls:
                image_urls.extend(srcset_urls)
    
    return image_urls


def extract_cook_time_from_text(text):
    """Extract cook time from text using regex patterns."""
    import re
    
    # Look for common time patterns and convert to total minutes
    patterns = [
        # Total Time patterns (preferred)
        r'Total Time:\s*(\d+)\s*(?:hrs?|hours?)\s*(\d+)\s*(?:mins?|minutes?)',
        r'Total Time:\n\s*(\d+)\s*(?:hrs?|hours?)\s*(\d+)\s*(?:mins?|minutes?)',
        r'Total Time:\s*(\d+)\s*(?:mins?|minutes?)',
        r'Total Time:\n\s*(\d+)\s*(?:mins?|minutes?)',
        
        # Prep + Cook Time patterns
        r'Prep Time:\s*(\d+)\s*(?:mins?|minutes?)\s*\+\s*Cook Time:\s*(\d+)\s*(?:hrs?|hours?)\s*(\d+)\s*(?:mins?|minutes?)',
        r'Prep Time:\n\s*(\d+)\s*(?:mins?|minutes?)\s*\+\s*Cook Time:\n\s*(\d+)\s*(?:hrs?|hours?)\s*(\d+)\s*(?:mins?|minutes?)',
        r'Prep Time:\s*(\d+)\s*(?:mins?|minutes?)\s*\+\s*Cook Time:\s*(\d+)\s*(?:mins?|minutes?)',
        r'Prep Time:\n\s*(\d+)\s*(?:mins?|minutes?)\s*\+\s*Cook Time:\n\s*(\d+)\s*(?:mins?|minutes?)',
        
        # Cook Time patterns
        r'Cook Time:\s*(\d+)\s*(?:hrs?|hours?)\s*(\d+)\s*(?:mins?|minutes?)',
        r'Cook Time:\n\s*(\d+)\s*(?:hrs?|hours?)\s*(\d+)\s*(?:mins?|minutes?)',
        r'Cook Time:\s*(\d+)\s*(?:mins?|minutes?)',
        r'Cook Time:\n\s*(\d+)\s*(?:mins?|minutes?)',
        
        # General Time patterns
        r'Time:\s*(\d+)\s*(?:hrs?|hours?)\s*(\d+)\s*(?:mins?|minutes?)',
        r'Time:\n\s*(\d+)\s*(?:hrs?|hours?)\s*(\d+)\s*(?:mins?|minutes?)',
        r'Time:\s*(\d+)\s*(?:mins?|minutes?)',
        r'Time:\n\s*(\d+)\s*(?:mins?|minutes?)',
        
        # Duration patterns
        r'Duration:\s*(\d+)\s*(?:hrs?|hours?)\s*(\d+)\s*(?:mins?|minutes?)',
        r'Duration:\n\s*(\d+)\s*(?:hrs?|hours?)\s*(\d+)\s*(?:mins?|minutes?)',
        r'Duration:\s*(\d+)\s*(?:mins?|minutes?)',
        r'Duration:\n\s*(\d+)\s*(?:mins?|minutes?)',
        
        # Other patterns
        r'(\d+)\s*(?:hrs?|hours?)\s*(\d+)\s*(?:mins?|minutes?)\s*Total',
        r'(\d+)\s*(?:mins?|minutes?)\s*Total',
        r'Total:\s*(\d+)\s*(?:hrs?|hours?)\s*(\d+)\s*(?:mins?|minutes?)',
        r'Total:\s*(\d+)\s*(?:mins?|minutes?)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 3:  # Hours + minutes (e.g., "3 hrs 5 mins")
                hours = int(groups[0])
                minutes = int(groups[1])
                total_minutes = (hours * 60) + minutes
                return str(total_minutes)  # Return just the number as string
            elif len(groups) == 2:  # Prep + Cook time or just minutes
                if 'Prep Time' in pattern and 'Cook Time' in pattern:  # Prep + Cook time
                    prep_time = int(groups[0])
                    cook_time = int(groups[1])
                    total_time = prep_time + cook_time
                    return str(total_time)  # Return just the number as string
                else:  # Just minutes
                    time_value = int(groups[0])
                    return str(time_value)  # Return just the number as string
            elif len(groups) == 1:  # Single time value
                time_value = int(groups[0])
                return str(time_value)  # Return just the number as string
    
    return None

def select_best_image(image_urls):
    """Select the best image from available URLs based on size indicators."""
    if not image_urls:
        return ""
    
    best_image = image_urls[0]  # fallback to first
    
    # Look for AllRecipes main recipe images first (they have specific patterns)
    for img_url in image_urls:
        # AllRecipes main recipe images often have these patterns
        if '/thmb/' in img_url and any(size in img_url for size in ['750x0', '800x', '1200x', '1500x']):
            # Check if it's not a small thumbnail or author photo
            if not any(skip in img_url.lower() for skip in ['40x0', '58x0', '76x0', 'headshot', 'avatar']):
                best_image = img_url
                break
    
    # If no AllRecipes pattern found, use general logic
    if best_image == image_urls[0]:
        for img_url in image_urls:
            # Prefer larger images (look for keywords in URL that indicate larger size)
            if any(keyword in img_url.lower() for keyword in ['1500x', '1200x', '800x', '750x', 'large', 'original']):
                best_image = img_url
                break
            # Avoid small thumbnails
            elif any(keyword in img_url.lower() for keyword in ['75x75', '100x100', '150x150', '40x0', '58x0']):
                continue
            else:
                best_image = img_url
    
    return best_image


def create_recipe_extraction_prompt(visible_text):
    """Create the prompt for recipe extraction (basic info only)."""
    return (
        "Extract basic recipe information from the provided web page text. "
        "Return a JSON object with the following structure:\n"
        "{\n"
        '  "title": "Recipe title",\n'
        '  "description": "Brief description or summary",\n'
        '  "ingredients": ["ingredient 1", "ingredient 2", ...],\n'
        '  "instructions": ["step 1", "step 2", ...]\n'
        "}\n\n"
        "Guidelines:\n"
        "- **Title**: Use the most prominent or repeated recipe title.\n"
        "- **Description**: Extract the first sentence or paragraph summarizing the recipe, verbatim if possible.\n"
        "- **Ingredients**: Capture each ingredient exactly as listed, including quantities, units, and optional phrases (e.g., '1 cup sugar, or more to taste').\n"
        "- **Instructions**: List instructions as numbered steps, preserving original wording and order.\n"
        "- **General**:\n"
        "  - If data is repeated, use the first clear instance.\n"
        "  - End with '---END---' to indicate completion.\n\n"
        f"Page text:\n{visible_text}\n\n"
        "JSON:"
    )


def create_recipe_extraction_prompt_org(visible_text):
    """Create the prompt for recipe extraction."""
    return (
        "Extract recipe information from this web page text. "
        "Return a JSON object with the following structure:\n"
        "{\n"
        '  "title": "Recipe title",\n'
        '  "description": "Brief description or summary",\n'
        '  "ingredients": ["ingredient 1", "ingredient 2", ...],\n'
        '  "instructions": ["step 1", "step 2", ...],\n'
        '  "cookTime": "time-period or time-range or null if not found",\n'
        '  "difficulty": "difficulty-level or null if not found"\n'
        "}\n\n"
        "Guidelines:\n"
        "- Extract the main recipe title\n"
        "- List all ingredients with quantities\n"
        "- Break down cooking instructions into numbered steps\n"
        "- If multiple images are found, use the most relevant one for the recipe\n"
        "- Usually the larger image and/or the alt test is related to the title\n"
        "- CAREFULLY search for cooking time information in the text\n"
        "- Look for time information in tables, lists, and text\n"
        "- Search for: 'Prep Time', 'Cook Time', 'Total Time', 'Time', 'Duration', 'mins', 'minutes'\n"
        "- For cookTime: If 'Total Time' is found, use that. If only 'Prep Time' and 'Cook Time' are found, add them together.\n"
        "- Convert any time found to 5-minute increment ranges (e.g., 5-10 min, 10-15 min, 15-20 min, 20-25 min, etc.)\n"
        "- cookTime could also be a single time period (e.g., 10 minutes, 1 hour, 2 hours, etc.)\n"
        "- Look for: 'Difficulty', 'Level', 'Easy', 'Medium', 'Hard', 'Beginner', 'Advanced'\n"
        "- For difficulty, use: Easy, Medium, Advanced\n"
        "- If time/difficulty not found, use null\n"
        "- Be thorough in searching the entire text for cook time and difficulty information\n"
        "- IMPORTANT: End your response with '---END---' to indicate completion\n\n"
        f"Page text:\n{visible_text}\n\n"
        "JSON:"
    )


def create_cook_time_difficulty_prompt(visible_text):
    """Create the prompt for cook time and difficulty extraction only."""
    return (
        "You are a recipe data extractor. Extract cook time and difficulty from the web page text below.\n\n"
        "IMPORTANT: Return ONLY a JSON object, nothing else. No explanations, no code examples.\n\n"
        "JSON format:\n"
        "{\n"
        '  "cookTime": "number-of-minutes",\n'
        '  "difficulty": "Easy|Medium|Advanced"\n'
        "}\n\n"
        "Rules:\n"
        "- cookTime: Find 'Total Time' and convert to total minutes (e.g., '3 hrs 5 mins' = '185')\n"
        "- difficulty: Find explicit terms or infer from recipe complexity\n"
        "- Return only the JSON object\n\n"
        f"Text to analyze:\n{visible_text}\n\n"
        "JSON:"
    )

def create_markdown_extraction_prompt(visible_text):
    """Create the prompt for markdown extraction as fallback."""
    return (
        "Extract recipe information from this web page text. "
        "Return the information in this exact markdown format:\n\n"
        "# Recipe Title\n"
        "## Description\n"
        "Brief description here\n\n"
        "## Ingredients\n"
        "- ingredient 1\n"
        "- ingredient 2\n"
        "- ingredient 3\n\n"
        "## Instructions\n"
        "1. step 1\n"
        "2. step 2\n"
        "3. step 3\n\n"
        "## Image\n"
        "image_url_here\n\n"
        "---END---\n\n"
        f"Page text:\n{visible_text}\n\n"
        "Markdown:"
    )


def extract_recipe_with_llm(visible_text, model, tokenizer, device):
    """Extract recipe data using LLM with two separate calls for better focus."""
    print(f"[import-recipe] Sending prompts to LLM (length: {len(visible_text) + 600})")
    
    # First call: Extract basic recipe information (title, description, ingredients, instructions)
    print(f"[import-recipe] Making first LLM call for basic recipe info...")
    basic_prompt = create_recipe_extraction_prompt(visible_text)
    basic_inputs = tokenizer(basic_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        basic_outputs = model.generate(**basic_inputs, max_new_tokens=4096, pad_token_id=tokenizer.eos_token_id)
    basic_text = tokenizer.decode(basic_outputs[0], skip_special_tokens=True)
    basic_response = basic_text[len(basic_prompt):].strip()
    
    print(f"[import-recipe] Basic info LLM call complete.")
    print(f"[import-recipe] Basic response length: {len(basic_response)}")
    print(f"[import-recipe] Basic response (first 500 chars): {basic_response[:500]}")
    
    # Remove ---END--- marker before parsing
    basic_response = basic_response.replace('---END---', '').strip()
    
    # Parse the basic JSON response
    data = parse_llm_response(basic_response, model, tokenizer, device)
    
    # If basic JSON parsing failed completely, try markdown extraction
    if not data or (not data.get("title") and not data.get("ingredients") and not data.get("instructions")):
        print(f"[import-recipe] Basic JSON parsing failed, attempting markdown extraction...")
        
        markdown_prompt = create_markdown_extraction_prompt(visible_text)
        markdown_inputs = tokenizer(markdown_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            markdown_outputs = model.generate(**markdown_inputs, max_new_tokens=4096, pad_token_id=tokenizer.eos_token_id)
        markdown_text = tokenizer.decode(markdown_outputs[0], skip_special_tokens=True)
        markdown_response = markdown_text[len(markdown_prompt):].strip()
        
        print(f"[import-recipe] Markdown response: {markdown_response}")
        
        # Remove ---END--- marker before parsing
        markdown_response = markdown_response.replace('---END---', '').strip()
        
        # Parse markdown sections
        data = extract_json_from_markdown(markdown_response)
        print(f"[import-recipe] Extracted from markdown: {data}")
    
    # Second call: Extract cook time and difficulty only
    print(f"[import-recipe] Making second LLM call for cook time and difficulty...")
    time_diff_prompt = create_cook_time_difficulty_prompt(visible_text)
    time_diff_inputs = tokenizer(time_diff_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        time_diff_outputs = model.generate(**time_diff_inputs, max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id)
    time_diff_text = tokenizer.decode(time_diff_outputs[0], skip_special_tokens=True)
    time_diff_response = time_diff_text[len(time_diff_prompt):].strip()
    
    print(f"[import-recipe] Cook time/difficulty LLM call complete.")
    print(f"[import-recipe] Time/diff response: {time_diff_response}")
    
    # Remove ---END--- marker before parsing
    time_diff_response = time_diff_response.replace('---END---', '').strip()
    
    # Parse the time/difficulty JSON response
    time_diff_data = parse_llm_response(time_diff_response, model, tokenizer, device)
    
    # Merge the data
    if time_diff_data:
        data.update(time_diff_data)
    
    return data


async def import_recipe_from_url(url: str, model, tokenizer, device) -> ImportRecipeResponse:
    """Import recipe from a given URL."""
    try:
        print(f"[import-recipe] Fetching URL: {url}")
        html = await fetch_html_with_playwright(url)
        print(f"[import-recipe] Fetched URL, status: {200}") # Playwright doesn't return status code directly here
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Extract image URLs
        image_urls = extract_images_from_html(soup, url)
        print(f"[import-recipe] Found {len(image_urls)} image URLs:")
        for i, img_url in enumerate(image_urls[:10]):  # Show first 10 images
            print(f"[import-recipe]   {i+1}: {img_url}")
        if len(image_urls) > 10:
            print(f"[import-recipe]   ... and {len(image_urls) - 10} more images")
        
        # Clean HTML and extract focused recipe text
        visible_text = clean_html_for_recipe_extraction(soup)
        print(f"[import-recipe] Extracted cleaned text, length: {len(visible_text)}")
        print(f"[import-recipe] First 500 chars of cleaned text: {visible_text[:500]}")
        
        # Write cleaned text to file for debugging
        import os
        debug_dir = "/tmp/recipe_debug"
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, f"cleaned_text_{int(time.time())}.txt")
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(f"URL: {url}\n")
            f.write(f"Length: {len(visible_text)}\n")
            f.write("="*50 + "\n")
            f.write(visible_text)
        print(f"[import-recipe] Cleaned text saved to: {debug_file}")

        # Extract recipe data using LLM
        data = extract_recipe_with_llm(visible_text, model, tokenizer, device)
        
        # Always prioritize our found images over LLM hallucination
        best_image = ""
        if image_urls:
            best_image = select_best_image(image_urls)
            
        # If LLM provided an image and it's one of our found images, use it
        llm_image = data.get("imageUrl", "")
        if llm_image and llm_image in image_urls:
            best_image = llm_image
            print(f"[import-recipe] LLM selected valid image: {llm_image}")
        elif llm_image:
            print(f"[import-recipe] LLM selected image not in found images: {llm_image}")
            # Don't use LLM hallucinated images, stick with our found ones
            
        # Set the final image
        data["imageUrl"] = best_image
        if best_image:
            print(f"[import-recipe] Final selected image: {best_image}")
        else:
            print(f"[import-recipe] No suitable image found")
        
        # Log all images that look like recipe images (for debugging)
        recipe_like_images = [img for img in image_urls if any(keyword in img.lower() for keyword in ['1500x', '750x', '800x', '1200x', 'recipe', 'food', 'dish']) and not any(skip in img.lower() for skip in ['headshot', 'avatar', 'author'])]
        if recipe_like_images:
            print(f"[import-recipe] Recipe-like images found:")
            for img in recipe_like_images[:5]:
                print(f"[import-recipe]   - {img}")
        
        # Set reasoning for imported values
        cook_time = data.get("cookTime")
        difficulty = data.get("difficulty")
        
        # If LLM didn't find cook time, try to extract it from the cleaned text
        if not cook_time or cook_time == "null" or cook_time == "Pending...":
            cook_time = extract_cook_time_from_text(visible_text)
            time_reasoning = "Extracted from page text" if cook_time else ""
        else:
            time_reasoning = "Imported from recipe"
            
        difficulty_reasoning = "Imported from recipe" if difficulty and difficulty != "Undetermined" else ""
        
        return ImportRecipeResponse(
            title=data.get("title", "Imported Recipe"),
            description=data.get("description", ""),
            ingredients="\n".join(data.get("ingredients", [])) if isinstance(data.get("ingredients"), list) else str(data.get("ingredients", "")),
            instructions="\n".join(data.get("instructions", [])) if isinstance(data.get("instructions"), list) else str(data.get("instructions", "")),
            imageUrl=data.get("imageUrl", ""),
            tags=["imported"],
            cookTime=cook_time or "Pending...",
            difficulty=difficulty or "Undetermined",
            timeReasoning=time_reasoning,
            difficultyReasoning=difficulty_reasoning
        )
        
    except Exception as e:
        print(f"[import-recipe] Error: {e}")
        import traceback
        traceback.print_exc()
        return ImportRecipeResponse(
            title="Imported Recipe",
            description=f"Error: {str(e)}",
            ingredients="",
            instructions="",
            imageUrl="",
            tags=["imported", "error"],
            cookTime="Pending...",
            difficulty="Undetermined",
            timeReasoning="",
            difficultyReasoning=""
        ) 