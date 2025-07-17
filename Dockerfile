# Test Dockerfile for Whisper on Render
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy test application
COPY test_app.py app.py

# Expose port
EXPOSE 5000

# Run the test app
CMD ["python", "app.py"]
