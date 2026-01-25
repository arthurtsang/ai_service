#!/bin/bash

# AI Service Installation Script
# This script installs the AI service as a systemd service

set -e

echo "=========================================="
echo "AI Service Installation"
echo "=========================================="
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

echo "Installing AI Service..."
echo ""

# Install AI Server
echo "1. Installing AI Server..."
cd /home/tsangc1/Projects/ai_service
sudo cp ai-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-server
echo "✅ AI Server installed"
echo ""

echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Services installed:"
echo "  • AI Server (ai-server) - Port 8001"
echo ""
echo "To start the service:"
echo "  cd /home/tsangc1/Projects/ai_service"
echo "  ./manage-service.sh start"
echo ""
echo "To check status:"
echo "  ./manage-service.sh status"
echo ""
echo "To view logs:"
echo "  ./manage-service.sh logs"
echo ""
echo "To enable on boot:"
echo "  ./manage-service.sh enable"
