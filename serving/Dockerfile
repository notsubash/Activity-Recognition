FROM python:3.13-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# Full `pip install .` pulls mlflow, which requires pyarrow<20 while the package
# pins pyarrow==23.0.1. The API only needs these runtime packages.
RUN pip install --no-cache-dir \
        "numpy==2.2.6" \
        "fastapi==0.115.14" \
        "uvicorn==0.34.3" \
        "onnxruntime==1.20.1" \
    && pip install --no-cache-dir --no-deps .

EXPOSE 8000
ENV HAR_MODEL_PATH=/models/model.onnx
CMD ["uvicorn", "har.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
