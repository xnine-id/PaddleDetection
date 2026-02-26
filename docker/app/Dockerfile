FROM python:3.12.3-slim

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libcuda1 \
    ca-certificates \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# RUN ln -s /usr/bin/python3.10 /usr/bin/python

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /app/PaddleDetection

RUN python setup.py install
RUN python ppdet/modeling/tests/test_architectures.py

WORKDIR /app

EXPOSE ${PORT}
CMD ["python", "cmd/main.py"]
