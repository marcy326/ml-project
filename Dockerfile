# ベースイメージ
FROM python:3.10-slim-bullseye

# install essential libraries like gcc and make
RUN apt-get update && apt-get install -y build-essential git libgomp1

# コンテナ内の作業ディレクトリ
WORKDIR /app

# ソースコードをコピー
COPY . /app

# install project requirements
RUN pip install --upgrade pip && \
    pip install --no-cache -r /app/requirements.txt && \
    rm -f /app/requirements.txt
