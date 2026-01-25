#!/bin/bash

# AI Server Environment Setup Script
# This script helps you set up environment variables for the AI server

ENV_FILE=".env"
SERVICE_FILE="/etc/systemd/system/ai-server.service"

echo "AI Server Environment Setup"
echo "=========================="
echo ""

# Check if .env file exists
if [[ -f "$ENV_FILE" ]]; then
    echo "Found existing .env file. Current contents:"
    echo "----------------------------------------"
    cat "$ENV_FILE"
    echo "----------------------------------------"
    echo ""
    read -p "Do you want to update it? (y/n): " update_env
else
    update_env="y"
fi

if [[ "$update_env" == "y" || "$update_env" == "Y" ]]; then
    echo ""
    echo "Setting up environment variables..."
    echo ""
    
    # Get Hugging Face token
    echo "Hugging Face Token Setup:"
    echo "1. Go to https://huggingface.co/settings/tokens"
    echo "2. Create a new token with 'read' permissions"
    echo "3. Copy the token (starts with 'hf_')"
    echo ""
    read -p "Enter your Hugging Face token: " hf_token
    
    # Create or update .env file
    cat > "$ENV_FILE" << EOF
# Hugging Face Hub Token
HUGGINGFACE_HUB_TOKEN=$hf_token

# Other environment variables can be added here
# RECIPE_API_URL=http://localhost:3000/api
EOF
    
    echo ""
    echo "✅ .env file created/updated successfully!"
    echo ""
    
    # Update systemd service file
    echo "Updating systemd service file..."
    if [[ -f "$SERVICE_FILE" ]]; then
        # Remove the Environment line if it exists
        sudo sed -i '/^Environment=HUGGINGFACE_HUB_TOKEN=/d' "$SERVICE_FILE"
        
        # Reload systemd
        sudo systemctl daemon-reload
        
        echo "✅ Systemd service updated!"
        echo ""
        echo "To restart the service with new environment variables:"
        echo "  ./manage-service.sh restart"
    else
        echo "⚠️  Systemd service file not found. Run install-service.sh first."
    fi
else
    echo "Environment setup skipped."
fi

echo ""
echo "Setup complete! Your environment variables are ready." 