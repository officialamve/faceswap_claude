FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
COPY scripts ./scripts
RUN useradd --create-home --uid 10001 bot && mkdir -p /srv/models /tmp/faceswap-bot \
    && chown -R bot /srv /tmp/faceswap-bot
USER bot
ENV MODEL_DIR=/srv/models
EXPOSE 10000
CMD ["python", "-m", "app.main"]
