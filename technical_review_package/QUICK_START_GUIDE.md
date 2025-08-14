# QUICK START GUIDE - CryptoSmartTrader V2

## 🚀 DIRECTE INSTALLATIE (Windows)

### 1. Extract & Setup
```cmd
# Extract technical_review_package.zip naar gewenste locatie
# Open Command Prompt als Administrator in project directory
```

### 2. Een-Klik Installatie
```cmd
1_install_all_dependencies.bat
```
**Wacht tot "✅ INSTALLATION COMPLETE!" verschijnt**

### 3. Validatie
```cmd  
workstation_validation.bat
```
**Verwacht resultaat**: `✅ VALIDATION PASSED - FULLY OPERATIONAL`

### 4. API Configuratie
Create `.env` bestand:
```env
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_SECRET=your_kraken_secret
OPENAI_API_KEY=your_openai_api_key
```

### 5. Start Systeem
```cmd
# Terminal 1:
2_start_background_services.bat

# Terminal 2:  
3_start_dashboard.bat
```

### 6. Toegang Dashboard
Open browser naar: `http://localhost:5000`

---

## ⚡ DIRECTE TOEGANG (Replit/Cloud)

### 1. Upload Files
Upload alle bestanden naar project root

### 2. Install Dependencies  
```bash
pip install -e .
```

### 3. Start Dashboard
```bash
streamlit run app_trading_analysis_dashboard.py --server.port 5000
```

---

## 🔍 TROUBLESHOOTING

### ❌ Python niet gevonden
```cmd
# Download Python 3.11+ van python.org
# Zorg voor PATH configuratie tijdens installatie
```

### ❌ Permission errors
```cmd
# Run Command Prompt als Administrator
# Voor optimale performance: Windows Defender exclusions
```

### ❌ Dependencies missing
```cmd
# Re-run installation:
1_install_all_dependencies.bat
```

### ❌ Port conflicts
```cmd
# Check ports:
netstat -an | findstr :5000

# Kill conflicting processes:
taskkill /F /PID [PID_NUMBER]
```

---

## ✅ SUCCESS INDICATORS

### Installation Success:
- ✅ Python 3.11+ detected
- ✅ Virtual environment created
- ✅ All dependencies installed  
- ✅ Project structure created
- ✅ Windows optimizations applied

### Validation Success:
- ✅ 8/8 mandatory dependencies available
- ✅ Mandatory Execution Gateway operational
- ✅ Project structure complete
- ✅ Main applications found
- ✅ API configuration detected

### Runtime Success:
- ✅ Dashboard loads binnen 5-10 seconden
- ✅ Real-time crypto data visible
- ✅ Security gates active (geen bypass mogelijk)
- ✅ Background services running
- ✅ Prometheus metrics op port 8090

---

## 🎯 EERSTE GEBRUIK

### 1. Dashboard Exploratie
- **Overzicht Tab**: Systeemstatus en live marktdata
- **Trading Analysis**: AI-powered crypto analyse  
- **Risk Management**: Position limits en veiligheidscontroles
- **Performance**: Real-time metrics en monitoring

### 2. Security Verificatie
```python
# Test in Python console:
from src.cryptosmarttrader.core.mandatory_execution_gateway import MANDATORY_GATEWAY
print("Gateway operational:", MANDATORY_GATEWAY is not None)
```

### 3. Live Trading Setup
- Configure API keys in `.env`  
- Verify connection in dashboard
- Start met kleine position sizes
- Monitor via Prometheus metrics (port 8090)

---

**SUPPORT**: Alle logs beschikbaar in `logs/` directory  
**MONITORING**: http://localhost:8090 voor Prometheus metrics  
**DOCUMENTATION**: Volledige docs in technical review package