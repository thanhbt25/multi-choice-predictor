# ==========================================
# STAGE 1: Build Stage (Cài đặt và chuẩn bị thư viện)
# ==========================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS builder

WORKDIR /code

ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt công cụ cần thiết để build/tải thư viện (git, dev tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Nâng cấp pip và cài đặt wheel
RUN pip install --no-cache-dir -U pip wheel

# Cài đặt PyTorch và các thư viện vào một thư mục riêng (/install) thay vì hệ thống
RUN pip install --no-cache-dir --prefix=/install torch --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==========================================
# STAGE 2: Final Stage (Image gọn nhẹ để chạy)
# ==========================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /code

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    PATH="/install/bin:${PATH}" \
    PYTHONPATH="/install/lib/python3.10/site-packages:${PYTHONPATH}"

# Chỉ cài python3-distutils hoặc python3-minimal vừa đủ để chạy (Không cài git, không cài python3-dev)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# COPY toàn bộ thư viện đã cài sạch sẽ từ Stage 1 sang, bỏ lại git và rác build
COPY --from=builder /install /install

# Copy mã nguồn logic vào container
COPY src/ ./src/

ENTRYPOINT ["python", "src/predict.py"]