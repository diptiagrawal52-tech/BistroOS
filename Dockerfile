# Stage 1: Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY main.py .
COPY index.html .

# Expose the port Render will assign via $PORT
EXPOSE 8000

# Start the FastAPI server on 0.0.0.0 so Render's reverse proxy can reach it
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
