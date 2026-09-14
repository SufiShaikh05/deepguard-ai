FROM python:3.11-slim
WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Only what's needed to RUN inference, not the training infrastructure
COPY backend/ ./backend/
COPY ml_core/models ./ml_core/models
COPY ml_core/explainability ./ml_core/explainability
COPY ml_core/utils ./ml_core/utils
COPY ml_core/configs ./ml_core/configs
COPY ml_core/checkpoints/best_model.pth ./ml_core/checkpoints/best_model.pth

WORKDIR /app/backend
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
