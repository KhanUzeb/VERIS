FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY veris ./veris
RUN pip install --no-cache-dir .

COPY . .

# Trend/cache DBs live in a mounted volume so history persists across
# container runs — mount /app/data to preserve trend history
# wiring on the host side.
VOLUME ["/app/data"]
ENV TREND_DB_PATH=/app/data/veris_trends.db

ENTRYPOINT ["python", "-m", "veris.cli"]
CMD ["--help"]
