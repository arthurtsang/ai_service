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
