# Multi-stage Dockerfile for EasyML with Azure Blob Storage
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/models /app/datasets /app/logs /app/dvc_storage

# Set permissions
RUN chmod +x /app/scripts/init_database.py
RUN chmod +x /app/check_system_status.py

# Production stage
FROM base as production

# Copy environment template
COPY .env.container /app/.env.template

# Create a non-root user
RUN groupadd -r easyml && useradd -r -g easyml easyml
RUN chown -R easyml:easyml /app
USER easyml

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start script
COPY --chown=easyml:easyml docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
