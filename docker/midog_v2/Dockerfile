FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 LC_ALL=C.UTF-8 LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libgl1 libglib2.0-0 libjpeg-turbo-progs libtiff5 libopenjp2-7 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python -m pip install --upgrade pip

WORKDIR /opt/algorithm

# Torch 1.9.0 + cu111
RUN python -m pip install --upgrade "pip<24.1" && \
    pip install --no-cache-dir \
      torch==1.9.0+cu111\
      -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.inference.txt .
RUN pip install --no-cache-dir -r requirements.inference.txt

# Copy only what you need
COPY algorithm /opt/algorithm/algorithm
COPY inference.py /opt/algorithm/inference.py

# Copy just the YAMLs (do NOT bake large .ckpt here)
COPY resources/patch_config.yaml /opt/algorithm/resources/patch_config.yaml
COPY resources/model_config.yaml /opt/algorithm/resources/model_config.yaml
COPY resources/inference_config.yaml /opt/algorithm/resources/inference_config.yaml

# Entrypoint
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Non-root user (Grand-Challenge friendly)
RUN useradd -m algo && chown -R algo:algo /opt/algorithm
USER algo

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
