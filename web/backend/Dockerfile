# Backend Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv for faster dependency management
RUN pip install uv

# Copy holdem-core first (dependency)
COPY holdem-core /holdem-core

# Copy backend code
COPY web/backend/pyproject.toml .
COPY web/backend/app ./app

# Install dependencies
RUN uv pip install --system -e /holdem-core -e .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
