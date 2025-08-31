# Base Image: Use a comprehensive, NVIDIA-optimized PyTorch container
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set the working directory inside the container
WORKDIR /workspace/project

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the Python requirements file
COPY requirements.txt .

# Upgrade pip and install all general Python libraries
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch Geometric using the official method for our CUDA/PyTorch version
RUN pip install torch_geometric \
  --extra-index-url https://data.pyg.org/whl/torch-2.2.0+cu121.html

# Copy the rest of the project's source code
COPY . .

# Set up Jupyter Lab as the default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''"]