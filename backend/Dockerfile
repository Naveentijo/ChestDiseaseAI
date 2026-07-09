# Base slim python image
FROM python:3.10-slim

# Set environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies for OpenCV library
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source directories
COPY ml/ /app/ml/
COPY backend/ /app/backend/

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
