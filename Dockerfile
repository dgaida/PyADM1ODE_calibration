# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

# Install system dependencies including Mono for .NET components
RUN apt-get update && apt-get install -y --no-install-recommends \
    mono-complete \
    git \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the package and its dependencies
RUN pip install --upgrade pip && \
    pip install .[dev,docs]

# Default command
CMD ["bash"]
