# CPU image for Linux servers and cloud batch jobs.
#   docker build -t flusight-fm .
#   docker run --rm \
#     -v "$PWD/forecasts:/app/forecasts" -v "$PWD/cache:/app/cache" -v hf:/root/.cache/huggingface \
#     flusight-fm forecast
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY flusight_fm ./flusight_fm
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -e ".[all]"
COPY model-metadata ./model-metadata
ENTRYPOINT ["flusight"]
