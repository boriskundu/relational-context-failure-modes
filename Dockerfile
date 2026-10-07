FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests
COPY docs ./docs
RUN pip install --no-cache-dir -e ".[dev]"

# Default: the offline test suite (no API keys, no network).
CMD ["python", "-m", "pytest", "tests/", "-q"]
