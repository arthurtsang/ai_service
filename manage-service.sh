#!/bin/bash

# AI Server Service Manager
# This script provides easy management of the AI server systemd service

SERVICE_NAME="ai-server"

show_usage() {
    echo "Usage: $0 {start|stop|restart|status|logs|enable|disable}"
    echo ""
    echo "Commands:"
    echo "  start   - Start the AI server service"
    echo "  stop    - Stop the AI server service"
    echo "  restart - Restart the AI server service"
    echo "  status  - Show service status"
    echo "  logs    - Show service logs (follow mode)"
    echo "  enable  - Enable service to start on boot"
    echo "  disable - Disable service from starting on boot"
    echo ""
}

case "$1" in
    start)
        echo "Starting AI server service..."
        sudo systemctl start "$SERVICE_NAME"
        echo "Service started. Check status with: $0 status"
        ;;
    stop)
        echo "Stopping AI server service..."
        sudo systemctl stop "$SERVICE_NAME"
        echo "Service stopped."
        ;;
    restart)
        echo "Restarting AI server service..."
        sudo systemctl restart "$SERVICE_NAME"
        echo "Service restarted. Check status with: $0 status"
        ;;
    status)
        echo "AI server service status:"
        sudo systemctl status "$SERVICE_NAME" --no-pager
        ;;
    logs)
        echo "Showing AI server logs (press Ctrl+C to exit):"
        sudo journalctl -u "$SERVICE_NAME" -f
        ;;
    enable)
        echo "Enabling AI server service to start on boot..."
        sudo systemctl enable "$SERVICE_NAME"
        echo "Service enabled."
        ;;
    disable)
        echo "Disabling AI server service from starting on boot..."
        sudo systemctl disable "$SERVICE_NAME"
        echo "Service disabled."
        ;;
    *)
        show_usage
        exit 1
        ;;
esac 