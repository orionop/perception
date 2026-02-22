# Perception Lab API

The Perception Lab in the UI runs real inference via a FastAPI backend that uses the DINOv2 model and full pipeline.

## Setup

1. **Install API dependencies:**
   ```bash
   pip install ".[api]"
   # Or: pip install fastapi uvicorn python-multipart
   ```

2. **Add model weights and prior** (place in `weights/` directory or set env vars):
   - `weights/best_model_v6.pth` — DINOv2 + ConvNeXt segmentation weights
   - `weights/joint_histograms.pkl` — 4D Bayesian prior (optional)
   
   Or set environment variables when starting the server:
   ```bash
   export PERCEPTION_WEIGHTS=weights/best_model_v6.pth
   export PERCEPTION_PRIOR=weights/joint_histograms.pkl
   ```

3. **Run the API server:**
   ```bash
   uvicorn perception_engine.api.server:app --reload --port 8000
   ```

4. **Run the frontend** (in `ui/`):
   ```bash
   npm run dev
   ```

5. Open http://localhost:3000/perception and upload a terrain image. Click **Run Pipeline** to get real segmentation, cost map, and safe path.

## API Endpoint

- **POST** `/api/infer` — accepts `multipart/form-data` with `image` file
- Returns JSON with base64-encoded PNGs: `raw`, `segmentation`, `costmap`, `path`, plus `path_found`, `path_cost`, `path_length`, `inference_time_ms`

## Frontend Config

Set `NEXT_PUBLIC_PERCEPTION_API_URL` (e.g. in `.env.local`) to point to the API if not at `http://localhost:8000`.
