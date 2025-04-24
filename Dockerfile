FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# 添加DEBIAN_FRONTEND=noninteractive避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get install -y software-properties-common curl zip unzip git-lfs awscli libssl-dev openssh-server vim \
    && apt-get install -y net-tools iputils-ping iproute2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get install --reinstall ca-certificates && update-ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y 'ppa:deadsnakes/ppa' && apt update
RUN apt install python3.10 python3.10-dev python3.10-distutils python3.10-venv -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://bootstrap.pypa.io/get-pip.py | python3.10
RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN pip uninstall -y Pillow && pip install pillow

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install onnxruntime-gpu==1.17.0  --index-url=https://pkgs.dev.azure.com/onnxruntime/onnxruntime/_packaging/onnxruntime-cuda-12/pypi/simple --force-reinstall --no-deps

WORKDIR /app

RUN mkdir -p /app/models/tokenizer \
    && mkdir -p /app/models/tts \
    && mkdir -p /app/models/llm \
    && mkdir -p /app/config \
    && mkdir -p /app/data/output \
    && mkdir -p /app/data/cache \
    && mkdir -p /app/logs

VOLUME ["/app/models", "/app/config", "/app/data", "/app/logs"]

ENV MODELS_DIR=/app/models \
    CONFIG_DIR=/app/config \
    DATA_DIR=/app/data \
    LOGS_DIR=/app/logs \
    STEP_AUDIO_TOKEN="step_audio_778899"

RUN echo "net.ipv6.conf.all.disable_ipv6 = 0" >> /etc/sysctl.conf \
    && echo "net.ipv6.conf.default.disable_ipv6 = 0" >> /etc/sysctl.conf \
    && echo "net.ipv6.conf.lo.disable_ipv6 = 0" >> /etc/sysctl.conf

COPY . /app/

EXPOSE 7860

CMD ["python", "app.py", "--model-path", "/app/models", "--server-name", "::"]
