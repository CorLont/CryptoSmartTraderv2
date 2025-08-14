# CryptoSmartTrader V2 - Workstation Deployment Guide

## Perfect Workstation Setup - 3 Simple Steps

### 🔧 Step 1: Install All Dependencies
```bash
1_install_all_dependencies.bat
```
**Wat dit doet:**
- Installeert alle Python dependencies (Streamlit, PyTorch, CCXT, etc.)
- Configureert directory structuur
- Controleert GPU ondersteuning
- Maakt configuratie bestanden aan
- Test alle installaties

### ⚡ Step 2: Start Background Services
```bash
2_start_background_services.bat
```
**Wat dit start:**
- Data Collection Service (market data van exchanges)
- ML Prediction Service (LSTM/GRU/Transformer predictions)
- Sentiment Analysis Service (social media & news sentiment)
- Whale Detection Service (grote transactie monitoring)
- Technical Analysis Service (RSI, MACD, Bollinger Bands)
- Risk Management Service (VaR, portfolio risk)
- Portfolio Optimization Service (AI-driven allocations)
- System Health Monitor (CPU, memory, component status)

### 🚀 Step 3: Launch Dashboard
```bash
3_start_dashboard.bat
```
**Wat dit opent:**
- Hoofddashboard op http://localhost:5000
- 8 gespecialiseerde AI/ML dashboards
- Real-time market analysis
- Volledig interactieve interface

## 🎯 Available Dashboards

### Core Dashboards
- **🏠 Main Dashboard** - Systeem overzicht en snelle acties
- **📊 Comprehensive Market** - Real-time marktanalyse van 1458+ crypto's
- **🧠 Causal Inference** - Ontdek WAAROM markten bewegen
- **🤖 RL Portfolio Allocation** - AI-gedreven portfolio optimalisatie

### Advanced AI/ML Dashboards
- **🔧 Self-Healing System** - Autonome systeembescherming
- **🎲 Synthetic Data Augmentation** - Edge case training & stress testing
- **👤 Human-in-the-Loop** - Expert feedback integratie
- **📊 Shadow Trading** - Risicovrije strategy validatie

## 📋 System Requirements

### Minimum Requirements
- Windows 10/11
- Python 3.11+
- 8GB RAM
- 10GB vrije schijfruimte
- Internetverbinding

### Recommended Requirements
- Windows 11
- Python 3.11+
- 16GB RAM
- NVIDIA GPU met CUDA (voor ML acceleration)
- SSD storage
- Stabiele internetverbinding

## 🔑 API Keys Setup

Edit `.env` bestand met jouw API keys:
```bash
OPENAI_API_KEY=your_openai_key_here
KRAKEN_API_KEY=your_kraken_key
KRAKEN_SECRET=your_kraken_secret
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET=your_binance_secret
```

## 🛠️ Troubleshooting

### Dependencies Installation Issues
```bash
# Als pip installation faalt
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel

# Voor GPU problemen
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Background Services Issues
```bash
# Check of services draaien
tasklist | findstr python

# Stop alle services
taskkill /F /IM python.exe /FI "WINDOWTITLE eq CryptoTrader*"

# Restart services
2_start_background_services.bat
```

### Dashboard Issues
```bash
# Check port 5000
netstat -ano | find ":5000"

# Kill proces op port 5000
for /f "tokens=5" %a in ('netstat -ano ^| find ":5000"') do taskkill /PID %a /F

# Restart dashboard
3_start_dashboard.bat
```

## 📊 Performance Optimization

### GPU Acceleration
- Systeem detecteert automatisch CUDA GPU
- CuPy wordt geïnstalleerd voor GPU versnelling
- PyTorch gebruikt automatisch GPU als beschikbaar

### Memory Management
- Intelligent cache management
- Memory-aware data processing
- Automatic cleanup van oude data

### CPU Optimization
- Multi-threading voor alle agents
- Async processing waar mogelijk
- Automatic load balancing

## 🔄 Daily Operations

### Morning Routine
1. Start background services: `2_start_background_services.bat`
2. Launch dashboard: `3_start_dashboard.bat`
3. Check system health in dashboard
4. Review overnight predictions and alerts

### Evening Routine
1. Check portfolio performance
2. Review trade recommendations
3. Backup important data
4. Optional: Stop services voor nacht

## 📈 Monitoring & Alerts

### System Health
- Real-time CPU/Memory monitoring
- Component status tracking
- Automatic alerts bij problemen
- Performance metrics logging

### Trading Alerts
- High confidence prediction alerts
- Whale activity detection
- Market regime changes
- Risk threshold breaches

## 🎯 Advanced Features

### Machine Learning
- LSTM/GRU/Transformer models voor price prediction
- Bayesian uncertainty quantification
- Ensemble model predictions
- Automated feature engineering

### Risk Management
- Real-time VaR calculation
- Portfolio stress testing
- Black swan detection
- Automatic position sizing

### AI Integration
- OpenAI-powered market analysis
- Sentiment analysis van social media
- Causal inference voor market relationships
- Human-in-the-loop feedback

## 🚀 Getting Maximum Performance

1. **Use SSD storage** voor fastest data access
2. **Enable GPU acceleration** voor ML models
3. **Allocate 16GB+ RAM** voor optimal performance
4. **Stable internet** voor real-time data feeds
5. **Regular updates** van dependencies
6. **Monitor system health** daily

Het systeem is nu geoptimaliseerd voor perfect draaien op jouw workstation met maximale performance en stabiliteit!