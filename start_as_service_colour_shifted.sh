#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="/etc/systemd/system/uno-q-inference.service"
USERNAME=$(whoami)

cat > /tmp/uno-q-inference.service <<EOF
[Unit]
Description=uno-q-inference FastAPI server
After=network.target

[Service]
User=$USERNAME
WorkingDirectory=$SCRIPT_DIR
ExecStart=NODE_NAME=colour_shifted $SCRIPT_DIR/.venv/bin/fastapi run web_app.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo mv /tmp/uno-q-inference.service "$SERVICE_FILE"
sudo systemctl daemon-reload
sudo systemctl enable uno-q-inference.service
sudo systemctl start uno-q-inference.service

echo "Service installed and started. Check status with:"
echo "  sudo systemctl status uno-q-inference"
