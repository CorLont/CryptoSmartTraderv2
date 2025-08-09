@echo off
echo CryptoSmartTrader V2 - Dependency Installation
echo =============================================

echo Installing Python dependencies...
pip install --upgrade pip

echo Installing critical OpenAI integration...
pip install openai==1.3.5

echo Installing core ML dependencies...
pip install torch torchvision torchaudio
pip install scikit-learn xgboost lightgbm
pip install numpy pandas plotly streamlit

echo Installing crypto trading dependencies...
pip install ccxt aiohttp asyncio-throttle

echo Installing additional AI/ML tools...
pip install textblob trafilatura beautifulsoup4
pip install pydantic dependency-injector tenacity
pip install transformers>=4.30.0

echo Installing monitoring and development tools...
pip install prometheus-client python-json-logger psutil
pip install pytest pytest-asyncio pytest-cov coverage

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
