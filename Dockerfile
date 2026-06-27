# 1. Sử dụng Base Image CUDA 12.8 theo yêu cầu của BTC
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /code

# Cấu hình biến môi trường
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTORCH_ALLOC_CONF="expandable_segments:True"

# 2. Cài đặt Python, Git và dọn dẹp bộ nhớ đệm hệ thống
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Tạo alias cho python
RUN ln -s /usr/bin/python3 /usr/bin/python

# 3. Nâng cấp pip và cài đặt PyTorch phiên bản hỗ trợ CUDA 12.x từ nguồn chính thức
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# 4. Cài đặt các thư viện Python theo layer cache của bạn
COPY requirements.txt .
RUN pip install --no-cache-dir git+https://github.com/huggingface/transformers.git && \
    pip install --no-cache-dir -r requirements.txt

# Copy file dữ liệu và toàn bộ mã nguồn
COPY private_test.json .
COPY src/ ./src/

ENTRYPOINT ["python", "src/predict.py"]