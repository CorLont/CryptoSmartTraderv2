@echo off
echo CryptoSmartTrader V2 - Dependency Installation
echo =============================================

echo Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Installing optional GPU monitoring...
pip install GPUtil pynvml

echo Creating necessary directories...
mkdir logs\daily 2>nul
mkdir models\backup 2>nul
mkdir cache\temp 2>nul
mkdir data\raw 2>nul

echo Configuring Windows Defender exclusions...
powershell -Command "Add-MpPreference -ExclusionPath '%CD%'"
powershell -Command "Add-MpPreference -ExclusionProcess 'python.exe'"

echo Dependencies installed successfully!
pause
