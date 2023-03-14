FROM python:3.9-slim

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN apt update -y && apt install -y ffmpeg \
libsm6 \
libxext6 \
net-tools \
iputils-ping

RUN python -m pip install --upgrade pip && \
pip install --default-timeout=100 -r requirements.txt

RUN printf "\nalias ls='ls --color=auto'\n" >> ~/.bashrc
RUN printf "\nalias ll='ls -alF'\n" >> ~/.bashrc

ENV PACKAGE_PATH="/workspace/src/package"
