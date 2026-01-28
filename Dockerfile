FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

WORKDIR /app

# System deps for audio/video processing + Python runtime
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     python3 \
     python3-pip \
     python3-venv \
     python-is-python3 \
     ffmpeg \
     libsm6 \
     libxext6 \
  && rm -rf /var/lib/apt/lists/*

# Install uv (use python3 explicitly)
RUN python3 -m pip install --no-cache-dir uv

# Install Python dependencies first (better layer caching)
COPY pyproject.toml uv.lock ./
# Pull CUDA-enabled PyTorch wheels from the official index
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu129
RUN uv sync --frozen --no-dev

# Copy the rest of the project
COPY . .

ENV GRADIO_SERVER_NAME=0.0.0.0 \
    PORT=7860 \
    MMAUDIO_CACHE_DIR=/data/mmaudio_cache

EXPOSE 7860

CMD ["./start.sh"]
