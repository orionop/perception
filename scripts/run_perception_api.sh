#!/bin/bash
# Run the Perception Engine API server.
# Install deps: pip install ".[api]" or: pip install fastapi uvicorn python-multipart
cd "$(dirname "$0")/.."
uvicorn perception_engine.api.server:app --reload --port 8000
