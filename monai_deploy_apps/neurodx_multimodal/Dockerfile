# NeuroDx-MultiModal MONAI Deploy Application Dockerfile
# Production-ready container for clinical deployment

FROM nvcr.io/nvidia/pytorch:23.08-py3

# Metadata
LABEL maintainer="NeuroDx Development Team"
LABEL version="1.0.0"
LABEL description="NeuroDx-MultiModal MONAI Deploy Application"
LABEL medical.specialty="neurology"
LABEL ai.framework="monai"
LABEL compliance.hipaa="true"

# Set working directory
WORKDIR /opt/neurodx

# Set environment variables
ENV PYTHONPATH=/opt/neurodx:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0
ENV MONAI_LOG_LEVEL=INFO
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install MONAI Deploy SDK
RUN pip install --no-cache-dir monai-deploy-app-sdk

# Install additional medical imaging dependencies
RUN pip install --no-cache-dir \
    nibabel \
    pydicom \
    SimpleITK \
    scikit-image \
    matplotlib \
    plotly \
    streamlit

# Copy application code
COPY app.py .
COPY app.yaml .
COPY models/ ./models/
COPY config/ ./config/

# Copy NeuroDx source code
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p /input /output /tmp/monai

# Set permissions
RUN chmod +x app.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('Health check passed')" || exit 1

# Expose ports (if needed for web interface)
EXPOSE 8080

# Set user (for security)
RUN useradd -m -u 1000 neurodx && chown -R neurodx:neurodx /opt/neurodx
USER neurodx

# Entry point
ENTRYPOINT ["python", "app.py"]