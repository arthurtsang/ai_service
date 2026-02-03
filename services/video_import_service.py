"""
Video recipe import: download description/subtitle (or transcribe with Whisper), then extract recipe with LLM.
Supports YouTube, Instagram, TikTok, and other sites yt-dlp supports.
"""
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from services.import_service import ImportRecipeResponse
from utils.json_parser import parse_llm_response, extract_json_from_markdown
from utils.llm_thread import run_llm_in_thread

logger = logging.getLogger(__name__)

# Minimum words in description to use "summarize"; below this we ask model to generate from transcript
DESCRIPTION_WORD_THRESHOLD = 50
DESCRIPTION_MAX_WORDS = 220


def _word_count(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(text.strip().split())


def _srt_to_text(srt_path: Path) -> str:
    """Extract plain text from SRT file, stripping timestamps."""
    text = srt_path.read_text(encoding="utf-8", errors="replace")
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.isdigit():
            continue
        if line in ("WEBVTT",) or line.startswith("NOTE ") or line.startswith("X-TIMESTAMP"):
            continue
        if re.match(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}", line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _transcribe_audio_with_whisper(audio_path: Path, language: Optional[str] = None) -> str:
    """Transcribe audio using faster-whisper. No language hint = auto-detect."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning("faster-whisper not installed; skipping transcription")
        return ""
    compute_type = "int8"  # CPU-friendly
    model = WhisperModel("base", device="cpu", compute_type=compute_type)
    segments, _ = model.transcribe(str(audio_path), language=language)
    lines = [seg.text.strip() for seg in segments if seg.text.strip()]
    return "\n".join(lines).strip()


# Directory for thumbnails returned to recipe backend (local path in imageUrl)
THUMBNAIL_TMP_DIR = Path("/tmp/ai-service-thumbnails")


def _download_video_content(url: str, work_dir: Path) -> Tuple[str, str, str, Optional[Path]]:
    """
    Use yt-dlp to download description, subtitle (or audio for transcription), and thumbnail.
    Returns (title, description, transcript, thumbnail_path or None).
    """
    # Download metadata + subtitles + thumbnail; if no subs, download audio
    cmd = [
        "yt-dlp",
        "--no-overwrites",
        "--write-info-json",
        "--write-description",
        "--write-thumbnail",
        "--convert-thumbnails", "jpg",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en,en-US,en-GB",
        "--convert-subs", "srt",
        "--restrict-filenames",
        "-x",
        "--audio-format", "m4a",
        "-o", str(work_dir / "%(title)s [%(id)s].%(ext)s"),
        "--ignore-errors",
        url,
    ]
    logger.info("[video-import] Running yt-dlp: %s", " ".join(cmd[:12]) + " ...")
    result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        logger.warning("[video-import] yt-dlp stderr: %s", result.stderr[:500] if result.stderr else "")

    title = ""
    description = ""
    transcript = ""

    # Find info.json (any)
    info_files = list(work_dir.glob("*.info.json"))
    if info_files:
        try:
            data = json.loads(info_files[0].read_text(encoding="utf-8"))
            title = (data.get("title") or "").strip() or "Untitled"
        except Exception as e:
            logger.warning("[video-import] Failed to read info.json: %s", e)

    # Find .description file (any)
    desc_files = list(work_dir.glob("*.description"))
    if desc_files:
        try:
            description = desc_files[0].read_text(encoding="utf-8", errors="replace").strip()
        except Exception as e:
            logger.warning("[video-import] Failed to read description: %s", e)

    # Prefer subtitle over transcription
    srt_files = list(work_dir.glob("*.srt"))
    audio_files = list(work_dir.glob("*.m4a"))
    stem_for_srt = None
    if audio_files:
        stem_for_srt = audio_files[0].stem
    if not stem_for_srt and info_files:
        try:
            idata = json.loads(info_files[0].read_text(encoding="utf-8"))
            tid = idata.get("id", "")
            ttitle = (idata.get("title") or "").strip()
            stem_for_srt = f"{ttitle} [{tid}]" if ttitle and tid else None
        except Exception:
            pass

    if stem_for_srt and srt_files:
        # Match SRT to same base (e.g. "Title [id].en-US.srt"); [ ] are special in glob
        stem_escaped = glob.escape(stem_for_srt)
        for pat in (f"{stem_escaped}.*.srt", f"{stem_escaped}.srt"):
            try:
                matches = sorted(work_dir.glob(pat))
            except Exception:
                matches = []
            if matches:
                transcript = _srt_to_text(matches[0])
                logger.info("[video-import] Using subtitle, %d chars", len(transcript))
                break

    if not transcript and audio_files:
        logger.info("[video-import] No subtitle; transcribing with Whisper (no language hint)")
        transcript = _transcribe_audio_with_whisper(audio_files[0], language=None)
        logger.info("[video-import] Transcript length: %d chars", len(transcript))

    if not title and audio_files:
        title = audio_files[0].stem[:80] or "Untitled"

    # Thumbnail: yt-dlp may write e.g. "Title [id].jpg" (with --convert-thumbnails jpg)
    thumbnail_path: Optional[Path] = None
    for ext in ("*.jpg", "*.webp", "*.png"):
        thumbs = list(work_dir.glob(ext))
        if thumbs:
            thumbnail_path = thumbs[0]
            break

    return title, description, transcript, thumbnail_path


def _instructions_to_numbered_steps(instructions: str) -> str:
    """If instructions are one concatenated paragraph, split by sentence into numbered steps."""
    if not instructions or not instructions.strip():
        return instructions
    s = instructions.strip()
    # Already numbered steps (starts with "1." or "1)" or has "2." on a new line)
    if re.search(r"^\s*1[\.\)]\s", s) or "\n2\." in s or "\n2)" in s:
        return s
    # Split on period + space when followed by capital (sentence boundary)
    parts = re.split(r"\.\s+(?=[A-Z])", s)
    if len(parts) <= 1:
        return s
    steps = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not p.endswith("."):
            p += "."
        steps.append(p)
    if len(steps) <= 1:
        return s
    return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))


def _create_video_recipe_prompt(title: str, description: str, transcript: str) -> str:
    desc_words = _word_count(description)
    if desc_words >= DESCRIPTION_WORD_THRESHOLD:
        instruction = (
            "Summarize the recipe below into a structured format. "
            "The description and transcript contain ingredients and instructions; "
            "write a single description of the recipe in at most 220 words. "
            "Extract title, ingredients list, and step-by-step instructions. "
            "Infer cookTime and difficulty if possible."
        )
    else:
        instruction = (
            "The video description is short or missing. Use the transcript to create a recipe. "
            "Write a short description of the dish (up to 220 words). "
            "Extract title, ingredients, and instructions from the transcript. "
            "Infer cookTime and difficulty if possible."
        )

    return (
        "You are a recipe data extractor. From the following video recipe (title, description, and transcript), "
        "return ONLY a JSON object with no other text.\n\n"
        f"Instructions: {instruction}\n\n"
        "JSON format:\n"
        "{\n"
        '  "title": "Recipe title",\n'
        '  "description": "Short description (max 220 words)",\n'
        '  "ingredients": "markdown or newline-separated list",\n'
        '  "instructions": "Numbered steps, ONE step per line: 1. First step.\\n2. Second step.\\n3. ... (do not put all steps in one line)",\n'
        '  "cookTime": "e.g. 30 min or Pending...",\n'
        '  "difficulty": "Easy or Medium or Advanced or Undetermined"\n'
        "}\n\n"
        f"Title: {title}\n\n"
        f"Description:\n{description or '(none)'}\n\n"
        f"Transcript:\n{transcript or '(none)'}\n\n"
        "JSON:"
    )


def _extract_recipe_from_video_llm(
    title: str, description: str, transcript: str, model, tokenizer, device
) -> Dict[str, Any]:
    """Call LLM to extract/summarize recipe from title + description + transcript."""
    prompt = _create_video_recipe_prompt(title, description, transcript)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = text[len(prompt):].strip().replace("---END---", "").strip()
    data = parse_llm_response(response, model, tokenizer, device)
    if not data or (not data.get("title") and not data.get("ingredients") and not data.get("instructions")):
        data = extract_json_from_markdown(response)
    if not data:
        data = {}
    # Normalize to strings
    raw_ing = data.get("ingredients")
    raw_instr = data.get("instructions")
    ingredients_str = "\n".join(str(x).strip() for x in raw_ing) if isinstance(raw_ing, list) else str(raw_ing or "").strip()
    instructions_str = "\n".join(str(x).strip() for x in raw_instr) if isinstance(raw_instr, list) else str(raw_instr or "").strip()
    # If LLM returned one long paragraph, split into numbered steps by sentence
    instructions_str = _instructions_to_numbered_steps(instructions_str)
    data["ingredients"] = ingredients_str
    data["instructions"] = instructions_str
    data["description"] = (data.get("description") or "").strip()
    if len(data["description"]) > 220 * 6:  # rough 220 words
        data["description"] = data["description"][: 220 * 6].rsplit(" ", 1)[0] + "..."
    data["cookTime"] = data.get("cookTime") or "Pending..."
    data["difficulty"] = data.get("difficulty") or "Undetermined"
    return data


async def import_recipe_from_video_url(
    job_id: str,
    url: str,
    model,
    tokenizer,
    device,
    progress: Optional[Dict[str, Any]] = None,
) -> ImportRecipeResponse:
    """
    Download video metadata/subtitle (or transcribe), then extract recipe with LLM.
    Uses same response shape as web import (ImportRecipeResponse).
    Thumbnail is saved to THUMBNAIL_TMP_DIR and its path returned in imageUrl for the recipe backend to copy.
    """
    def update(s: str) -> None:
        if progress is not None:
            progress["step"] = s
            progress["updated_at"] = time.time()

    update("starting")
    work_dir = Path(tempfile.mkdtemp(prefix="video_import_"))
    try:
        update("downloading")
        title, description, transcript, thumbnail_path = _download_video_content(url, work_dir)
        if not title and not transcript and not description:
            raise ValueError("Could not get title, description, or transcript from video URL")

        image_url = ""
        if thumbnail_path and thumbnail_path.exists():
            THUMBNAIL_TMP_DIR.mkdir(parents=True, exist_ok=True)
            dest = THUMBNAIL_TMP_DIR / f"{job_id}.jpg"
            try:
                shutil.copy2(thumbnail_path, dest)
                image_url = str(dest)
                logger.info("[video-import] Thumbnail saved to %s", image_url)
            except Exception as e:
                logger.warning("[video-import] Failed to copy thumbnail: %s", e)

        update("extracting_recipe")
        data = await run_llm_in_thread(
            _extract_recipe_from_video_llm,
            title, description, transcript,
            model, tokenizer, device,
        )
        update("completed")
        return ImportRecipeResponse(
            title=data.get("title") or title or "Untitled",
            description=data.get("description") or "",
            ingredients=data.get("ingredients") or "",
            instructions=data.get("instructions") or "",
            imageUrl=image_url,
            tags=["imported"],
            cookTime=data.get("cookTime") or "Pending...",
            difficulty=data.get("difficulty") or "Undetermined",
            timeReasoning="",
            difficultyReasoning="",
        )
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception as e:
            logger.warning("[video-import] Cleanup temp dir: %s", e)
