# 1. Base Image: Lightweight Python 3.11
FROM python:3.11-slim

# 2. Environment Variables
# PYTHONUNBUFFERED=1:   Forces logs to output immediately
# PYTHONDONTWRITEBYTECODE=1: Prevents .pyc file creation
# GRADIO_SERVER_NAME:   Tells Gradio to listen on all interfaces
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_SERVER_NAME="0.0.0.0"

# 3. Work Directory
WORKDIR /app

# 4. System Dependencies
# Install ffmpeg for Gradio Audio processing (microphone input)
# Clean up apt lists afterwards to keep image size small
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 5. Python Dependencies (Cached Layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Application Code
COPY . .

# 7. Networking
EXPOSE 7860

# 8. Entrypoint
CMD ["python", "app.py"]
