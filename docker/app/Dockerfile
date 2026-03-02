FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ARG PORT=8000
ENV PORT=${PORT}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgfortran5 \
    libopenblas0 \
    ca-certificates \
    curl \
    ccache \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Remove blinker package (to avoid conflicts)
RUN apt-get update && apt-get remove -y python3-blinker

# Set Python 3.11 as default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
 && ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip and set environment variables
RUN python3 -m pip install --upgrade pip
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1

# Install PaddlePaddle-GPU
RUN python3 -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Install PaddleDetection
WORKDIR /app/PaddleDetection
RUN python3 setup.py install
# RUN python3 ppdet/modeling/tests/test_architectures.py

WORKDIR /app

# Expose port and run application
EXPOSE ${PORT}
CMD ["python3", "cmd/api.py"]
