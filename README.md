# AI Service Module

This is the AI service module for the recipe application, now deployed as a separate module at the same level as `recipe` and `KungFuTournament`.

## Features

- Zephyr model with 4-bit quantization
- FastAPI-based REST API
- Systemd service for automatic startup
- CUDA support for GPU acceleration

## Structure

```
ai_service/
├── app.py                 # Main FastAPI application
├── ai-server.service     # Systemd service file
├── manage-service.sh     # Service management script
├── setup-env.sh         # Environment setup script
├── install-ai-service.sh # Installation script
├── models/              # AI model loading and inference
├── services/            # Business logic services
├── utils/               # Utility functions
├── pyproject.toml       # Python dependencies
└── .venv/              # Python virtual environment
```

## Dependencies (yt-dlp)

Video recipe import uses **yt-dlp** to fetch metadata, subtitles, thumbnails, and comments. It is declared in `pyproject.toml` as `yt-dlp (>=2024.1.0)`.

- **Where it runs from**: The systemd service sets `PATH=.venv/bin:...`. If `yt-dlp` is installed in the project venv (`.venv/bin/yt-dlp`), that is used; otherwise the system binary (e.g. `/usr/local/bin/yt-dlp`) is used.
- **If another agent installed it**: If `yt-dlp` is not in `.venv/bin`, it was likely installed system-wide (e.g. `pip install yt-dlp`, `sudo apt`, or another script). To use the **project’s** version, install deps into the venv: from the project root run `poetry lock` (to resolve yt-dlp into the lockfile if needed) and `poetry install`. Then restart the service.
- **Comment limit**: yt-dlp has no global `--max-comments` flag. The code uses `--extractor-args "youtube:max-comments=100"` to limit YouTube comments; other sites use default behavior.

To see which binary and version the service uses:
```bash
PATH=/home/tsangc1/Projects/ai_service/.venv/bin:/usr/local/bin:/usr/bin:/bin which yt-dlp
PATH=/home/tsangc1/Projects/ai_service/.venv/bin:/usr/local/bin:/usr/bin:/bin yt-dlp --version
```

## Installation

### Quick Install (as root)
```bash
cd /home/tsangc1/Projects/ai_service
sudo ./install-ai-service.sh
```

### Manual Install
1. Copy the service file:
   ```bash
   sudo cp ai-server.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable ai-server
   ```

2. Set up environment variables:
   ```bash
   ./setup-env.sh
   ```

## Service Management

Use the `manage-service.sh` script for common operations:

```bash
./manage-service.sh start      # Start the service
./manage-service.sh stop       # Stop the service
./manage-service.sh restart    # Restart the service
./manage-service.sh status     # Check service status
./manage-service.sh logs       # View service logs
./manage-service.sh enable     # Enable on boot
./manage-service.sh disable    # Disable on boot
```

## Configuration

The service runs on port 8001 and requires:
- Hugging Face Hub token (set via `setup-env.sh`)
- CUDA-compatible GPU (optional, for acceleration)
- Python virtual environment with dependencies

## Deployment

This module is deployed via:
1. **Manual installation**: Using `install-ai-service.sh`
2. **Ansible automation**: Via the System role `app_deployment`

## Dependencies

- Python 3.12+
- PyTorch with CUDA support
- Transformers library
- FastAPI and Uvicorn
- Poetry for dependency management

## Environment Variables

- `HUGGINGFACE_HUB_TOKEN`: Required for model downloads
- `PYTORCH_CUDA_ALLOC_CONF`: CUDA memory allocation settings
- `TRANSFORMERS_CACHE`: Hugging Face model cache directory
- `HF_HOME`: Hugging Face home directory
- `AI_SERVICE_MAX_MEMORY_GB`: Optional. Comma-separated per-GPU limits in GB when using multiple GPUs with different VRAM (e.g. `11,7` for 12GB + 8GB). Leave unset for single-GPU or equal cards.