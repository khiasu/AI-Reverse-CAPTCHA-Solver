# Production Docker configuration for RVR AI CAPTCHA SOLVER
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser

# Copy requirements first for better caching
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads model static/uploads \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080
ENV WORKERS=1
ENV THREADS=8

# Run the application
CMD ["sh", "-c", "gunicorn app_prod:app --bind 0.0.0.0:$PORT --workers $WORKERS --threads $THREADS --timeout 120 --keep-alive 2 --max-requests 1000 --max-requests-jitter 50"]
