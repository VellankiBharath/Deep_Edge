# Use a Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy necessary folders and files to the container
COPY ./client /app/client
COPY ./model /app/model
COPY ./server /app/server

# Install system dependencies required for OpenCV and others
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install NumPy first to ensure compatibility
RUN pip install --no-cache-dir numpy==1.24.3

# Install PyTorch, torchvision, and other dependencies with SSL verification disabled
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org \
    flask \
    torch==2.0.0 \
    torchvision==0.15.1 \
    diffusers \
    opencv-python-headless \
    pillow \
    matplotlib \
    openai-whisper \
    transformers \
    segment-anything

# Expose the port for Flask (5000 by default)
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "server/app.py"]
