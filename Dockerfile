FROM python:3.11-slim

WORKDIR /app
ENV MPLBACKEND=Agg PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[parquet]"

COPY config.yaml ./
COPY data ./data

# Default: train. Override the command to run `predict` or `drift` instead.
ENTRYPOINT ["mlpipeline"]
CMD ["train"]
