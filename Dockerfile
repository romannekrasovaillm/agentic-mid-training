FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git curl wget build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Flash Attention 2 (for GigaChat3 MLA acceleration)
RUN pip install --no-cache-dir flash-attn --no-build-isolation 2>/dev/null || \
    echo "flash-attn build failed — will fall back to sdpa attention"

# Project source
COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY tests/ tests/

# Install project
RUN pip install --no-cache-dir -e .

# Create output dirs
RUN mkdir -p outputs logs

# NLTK data (needed for some metrics)
RUN python -c "import nltk; nltk.download('punkt', quiet=True)" 2>/dev/null || true

# Pre-download GigaChat3 model (optional — comment out to download at runtime)
# RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
#     AutoTokenizer.from_pretrained('ai-sage/GigaChat3-10B-A1.8B', trust_remote_code=True); \
#     AutoModelForCausalLM.from_pretrained('ai-sage/GigaChat3-10B-A1.8B', trust_remote_code=True)"

ENTRYPOINT ["python", "scripts/run_experiment.py"]
CMD ["--config", "configs/base.yaml"]
