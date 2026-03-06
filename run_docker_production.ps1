# PowerShell Script to Launch Production App in Docker
# Image: ecg-project

$ImageName = "ecg_classification-cardioscan"
$CurrentDir = Get-Location

Write-Host "--- Launching CardioScan AI in Docker Container ---"
Write-Host "Image: $ImageName"
Write-Host "Mounting: $CurrentDir -> /workspace"

# Docker Command Breakdown:
# --gpus all: Access to GPU (if available/configured in Docker Desktop)
# -it: Interactive mode
# --rm: Remove container after exit
# -p 8501:8501: Forward Streamlit port
# -v: Mount volume
# -w /workspace: Set working directory
# COMMAND: Install specific production reqs (fast) then launch app

docker run --gpus all -it --rm `
    -p 8501:8501 `
    -v "${CurrentDir}:/workspace" `
    -w /workspace `
    $ImageName `
    /bin/bash -c "pip install -r production/requirements.txt && streamlit run production/app.py"

Write-Host "--- Container Stopped ---"
