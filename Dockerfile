# Dockerfile (Final Ubuntu-based Version)

# 1. Start from a stable Ubuntu base image
FROM ubuntu:22.04

# 2. Set environment variables to prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. Install Python 3.10, pip, and essential build tools + media libraries
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    build-essential \
    cmake \
    ffmpeg \
    libgl1 \
    libx264-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory in the container
WORKDIR /app

# 5. Copy the dependencies file
COPY requirements.txt .

# 6. Install Python dependencies using pip3
#    We can go back to using the pre-compiled wheel now
#    because the Ubuntu environment is more standard.
RUN pip3 install --no-cache-dir -r requirements.txt

# 7. Copy the rest of the application's code
COPY . .

# 8. Expose the port the app runs on
EXPOSE 8000

# 9. Define the command to run your app using python3
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]