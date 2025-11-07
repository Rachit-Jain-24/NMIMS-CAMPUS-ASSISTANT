# Use a slim Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by 'openai-whisper' (ffmpeg)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# This includes app.py, rag_backend.py, static/, and templates/
COPY . .

# Set the port environment variable
# This matches the port 8086 in your app.py
ENV PORT=8086

# Expose the port
EXPOSE 8086

# Run the application with Gunicorn (as mentioned in requirements.txt)
CMD ["gunicorn", "--bind", "0.0.0.0:8086", "app:app"]
