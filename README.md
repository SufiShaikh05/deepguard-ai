# DeepGuard AI

Explainable deepfake & synthetic image detection - a dual-stream
(spatial + frequency) PyTorch model, served through a FastAPI backend
with Grad-CAM explainability, shown in a React dashboard.

## Structure
- `ml_core/`  — model architecture, training, Grad-CAM, ONNX export
- `backend/`  — FastAPI inference service (two endpoints: full explainable, and fast ONNX-only)
- `frontend/` — React + Tailwind dashboard

## Local setup
1. `ml_core/`: create a venv, `pip install -r requirements.txt`, organize your
   dataset as `data/processed/{train,val,test}/{real,fake}/*.jpg`, then
   `python train.py`.
2. Export the trained model: `python -m export.export_onnx` (from `ml_core/`).
3. `backend/`: create a venv, `pip install -r requirements.txt`,
   copy `.env.example` to `.env`, then `fastapi dev app/main.py`.
4. `frontend/`: `npm install`, copy `.env.example` to `.env`, then `npm run dev`.
5. Visit `http://localhost:5173`.

## Testing
- `ml_core/`: `pytest`
- `backend/`: `pytest`
- `frontend/`: `npm run test`

## Docker
`docker compose up --build` from the repo root runs the full stack.
