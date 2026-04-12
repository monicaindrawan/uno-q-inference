#!/bin/bash
set -e

sudo systemctl stop uno-q-inference.service
sudo systemctl disable uno-q-inference.service
echo "Service stopped."
