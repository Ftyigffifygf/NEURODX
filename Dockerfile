# Multi-stage build for NeuroDx-MultiModal system
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgdcm-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Create virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/images data/wearable data/genomics models/checkpoints models/cache logs

# Set permissions
RUN chmod +x scripts/setup_genomics_workflow.py

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Default command
CMD ["python", "main.py", "--mode", "api"]

# Development stage
FROM base as development
RUN pip install --no-cache-dir pytest pytest-asyncio black flake8 mypy
CMD ["python", "main.py", "--mode", "test"]

# Production stage
FROM base as production
# Remove development dependencies and optimize
RUN pip uninstall -y pytest pytest-asyncio black flake8 mypy || true
# Set production environment
ENV FLASK_ENV=production
ENV PYTHONOPTIMIZE=1
CMD ["python", "main.py", "--mode", "api"]