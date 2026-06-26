# Base Image PyTorch hỗ trợ CUDA 12.1
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /code

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTORCH_ALLOC_CONF="expandable_segments:True"

# Cài đặt git và dọn dẹp cache để giảm dung lượng image
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt các thư viện Python trước để tối ưu hóa Docker Layer Cache
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir git+https://github.com/huggingface/transformers.git && \
    pip install --no-cache-dir -r requirements.txt

# Copy file dữ liệu public_test.json vào thẳng thư mục /code
COPY public_test.json .

# Copy toàn bộ mã nguồn trong thư mục src vào /code/src
COPY src/ ./src/

ENTRYPOINT ["python", "src/main.py"]