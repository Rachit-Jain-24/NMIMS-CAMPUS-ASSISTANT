# Use a slim Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by 'unstructured' (libmagic)
RUN apt-get update && apt-get install -y \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# This includes app.py, backend_processor.py, static/, and templates/
COPY . .

# Set the port environment variable
# This matches the port 5000 in your app.py
ENV PORT=5000

# Expose the port
EXPOSE 5000

# Run the application with Gunicorn (as mentioned in requirements.txt)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
