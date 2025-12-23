# Base Image: Lightweight Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (Required for OpenCV/cv2)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Requirements first (for caching)
COPY production/requirements.txt /app/requirements.txt

# Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project logic
# We copy 'production' specifically to keep the image clean, 
# but if the code relies on '../data' or '../models' mounts, 
# those will be provided by volume mapping in docker-compose.
COPY production /app/production

# Expose Streamlit Port
EXPOSE 8501

# Set the working directory to where app.py is relative to logic
WORKDIR /app/production

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]