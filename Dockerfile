FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 设置环境变量
ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 设置工作目录
WORKDIR /app

# 统一安装系统依赖
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        software-properties-common \
        curl \
        zip \
        unzip \
        git-lfs \
        awscli \
        libssl-dev \
        openssh-server \
        vim \
        net-tools \
        iputils-ping \
        iproute2 \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# 安装Python
RUN add-apt-repository -y 'ppa:deadsnakes/ppa' \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3.10-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && wget -qO- https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# 复制并安装Python依赖
COPY requirements.txt /app/
RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir onnxruntime-gpu==1.17.0 \
        --index-url=https://pkgs.dev.azure.com/onnxruntime/onnxruntime/_packaging/onnxruntime-cuda-12/pypi/simple \
        --force-reinstall --no-deps \
    && pip3 uninstall -y Pillow \
    && pip3 install --no-cache-dir pillow

# 创建应用目录结构
RUN mkdir -p /app/models/{tokenizer,tts,llm} \
    /app/config \
    /app/data/{output,cache} \
    /app/logs

# 配置卷和环境变量
VOLUME ["/app/models", "/app/config", "/app/data", "/app/logs"]

ENV MODELS_DIR=/app/models \
    CONFIG_DIR=/app/config \
    DATA_DIR=/app/data \
    LOGS_DIR=/app/logs

# 配置IPv6
RUN echo "net.ipv6.conf.all.disable_ipv6 = 0" >> /etc/sysctl.conf \
    && echo "net.ipv6.conf.default.disable_ipv6 = 0" >> /etc/sysctl.conf \
    && echo "net.ipv6.conf.lo.disable_ipv6 = 0" >> /etc/sysctl.conf

# 复制应用代码（放在最后以利用缓存）
COPY . /app/

# 设置安全令牌（放在运行时环境变量中更安全）
ENV STEP_AUDIO_TOKEN="step_audio_778899"

EXPOSE 7860

# 使用非root用户运行应用（提高安全性）
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

CMD ["python", "app.py", "--model-path", "/app/models", "--server-name", "::"]
