# PowerShell Script to Organize Production Models
# Run this AFTER you have downloaded the models from the server to project_root/models/

$SourceModels = "models"
$DestModels = "production/models"
$SourceData = "data"
$DestData = "production/data"
$RemoteIP = "103.97.164.116"
$RemotePort = "2222"
$RemoteUser = "ayush"

Write-Host "--- Starting Production Model Organization ---"

# Ensure Destinations Exist
if (-not (Test-Path -Path $DestModels)) { New-Item -ItemType Directory -Path $DestModels -Force }
if (-not (Test-Path -Path $DestData)) { New-Item -ItemType Directory -Path $DestData -Force }

# 1. Copy Models
$Models = @(
    "hierarchy_stage2_router.pth",
    "hierarchy_stage3_rhythm.pth",
    "hierarchy_stage3_structure.pth",
    "Engine_A_ResNet-50.pth"
)

foreach ($model in $Models) {
    $SrcPath = Join-Path $SourceModels $model
    $DestPath = Join-Path $DestModels $model
    
    if (Test-Path $SrcPath) {
        Copy-Item -Path $SrcPath -Destination $DestPath -Force
        Write-Host "✅ Copied $model to Production"
    } else {
        Write-Host "❌ MISSING: $model. Please download it from the server first." -ForegroundColor Red
    }
}

# 2. Handle Data (CSV)
$CSVName = "train_labels.csv"
$SrcCSV = Join-Path $SourceData $CSVName
$DestCSV = Join-Path $DestData $CSVName

if (-not (Test-Path $SrcCSV)) {
    Write-Host "⚠️ Local CSV missing. Attempting to download from server..."
    # Ensure local data dir exists
    if (-not (Test-Path $SourceData)) { New-Item -ItemType Directory -Path $SourceData -Force }
    
    # Try SCP download
    $ScpCmd = "scp -P $RemotePort ${RemoteUser}@${RemoteIP}:~/ecg-classification-project/data/$CSVName $SourceData\"
    Write-Host "Running: $ScpCmd"
    Invoke-Expression $ScpCmd
}

if (Test-Path $SrcCSV) {
    Copy-Item -Path $SrcCSV -Destination $DestCSV -Force
    Write-Host "✅ Copied $CSVName to Production"
} else {
    Write-Host "❌ FAILED: Could not find or download $CSVName. Visual Classifier will not work." -ForegroundColor Red
}

Write-Host "--- Organization Complete ---"
