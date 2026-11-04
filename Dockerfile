FROM python:3.13.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/model

COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/

COPY docker_model/ ./model/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
