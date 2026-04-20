#!/bin/bash
# Stops and disables the uno-q-inference systemd service.
# Run this before re-deploying or switching node profiles.
# Usage: bash stop_service.sh
set -e

sudo systemctl stop uno-q-inference.service     # stop the running process
sudo systemctl disable uno-q-inference.service  # remove auto-start on boot
echo "Service stopped."
