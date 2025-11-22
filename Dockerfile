FROM python:3.9-slim

# Environment settings for predictable, unbuffered logging
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Document exposed port
EXPOSE 8000

# Production-style command (omit --reload for stability inside container)
CMD ["uvicorn", "demo:app", "--host", "0.0.0.0", "--port", "8000"]