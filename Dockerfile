FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Trend/cache DBs live in a mounted volume so history persists across
# container runs — see .github/workflows/veris-gate.yml for Actions cache
# wiring on the host side.
VOLUME ["/app/data"]
ENV TREND_DB_PATH=/app/data/veris_trends.db

ENTRYPOINT ["python", "-m", "veris.cli"]
CMD ["run", "--target", "veris.targets:my_agent_fn"]
