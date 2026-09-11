FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install PyG extensions (CPU)
RUN pip install --no-cache-dir \
    torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.6.0+cpu.html

# Everything else
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

RUN mkdir -p feedback/submissions

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/api/health || exit 1

ENTRYPOINT ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
