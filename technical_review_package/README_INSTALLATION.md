# CryptoSmartTrader V2 - Installatie Handleiding

## 🚀 Snelle Installatie (3 Stappen)

### Stap 1: Installeer Dependencies
```cmd
1_install_all_dependencies.bat
```
**Wat dit doet:**
- Controleert Python installatie (vereist 3.10+)
- Maakt virtuele omgeving aan (.venv)
- Installeert alle benodigde Python packages
- Configureert project structuur
- Optimaliseert Windows instellingen
- Controleert GPU ondersteuning

### Stap 2: Start Achtergrond Services
```cmd
2_start_background_services.bat
```
**Wat dit doet:**
- Start Prometheus metrics server (poort 8090)
- Start system health monitor
- Start production orchestrator
- Configureert alle achtergrond processen

### Stap 3: Start Dashboard
```cmd
3_start_dashboard.bat
```
**Wat dit doet:**
- Start hoofd dashboard op http://localhost:5000
- Configureert hoge prestatie modus
- Toont live cryptocurrency analyse
- 471+ cryptocurrencies met ML predictions

## 📋 Vereisten

- **Windows 10/11**
- **Python 3.10 of hoger**
- **8GB+ RAM** (16GB aanbevolen)
- **NVIDIA GPU** (optioneel, voor CUDA)
- **Internetverbinding**

## 🔑 API Keys Configuratie

Maak een `.env` bestand in de hoofdmap:
```env
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_SECRET=your_kraken_secret
OPENAI_API_KEY=your_openai_api_key
```

## 🎯 Features

- **Real-time Analyse:** 471+ cryptocurrencies
- **ML Predictions:** 80% confidence gate systeem
- **Sentiment Analyse:** News en social media
- **Whale Detection:** Grote transactie monitoring
- **Technical Analysis:** RSI, MACD, Bollinger Bands
- **Dutch Language:** Volledige Nederlandse interface
- **Risk Management:** Geautomatiseerd risico assessment

## 📊 Dashboard Overzicht

Na het starten van het dashboard:
1. Ga naar **http://localhost:5000**
2. Klik op **"START ANALYSE"**
3. Bekijk live cryptocurrency data
4. Filter op high-confidence predictions
5. Analyseer trading opportunities

## 🔧 Troubleshooting

### Python Niet Gevonden
- Download Python van https://python.org
- Zorg ervoor dat Python in PATH staat

### Virtual Environment Fout
- Verwijder `.venv` map
- Run `1_install_all_dependencies.bat` opnieuw

### Port 5000 Bezet
- Stop andere applicaties op poort 5000
- Of wijzig poort in `3_start_dashboard.bat`

### GPU Niet Werkend
- Installeer NVIDIA drivers
- CUDA is optioneel, systeem werkt ook zonder

## 📁 Project Structuur

```
CryptoSmartTrader/
├── 1_install_all_dependencies.bat    # Installatie script
├── 2_start_background_services.bat   # Achtergrond services
├── 3_start_dashboard.bat             # Dashboard starter
├── app_fixed_all_issues.py           # Hoofd dashboard
├── core/                             # Kern functionaliteit
├── agents/                           # Trading agents
├── data/                             # Cryptocurrency data
├── exports/                          # Analyse resultaten
├── logs/                             # System logs
└── .env                              # API configuratie
```

## 🎮 Voor Gevorderden

### Custom Configuratie
- Bewerk `core/config_manager.py`
- Pas trading parameters aan
- Configureer extra exchanges

### Development Mode
```cmd
pip install -e .[dev]
pytest tests/
```

### Monitoring
- Prometheus metrics: http://localhost:8090
- Logs: `logs/` directory
- Health status: Zichtbaar in dashboard

## 📞 Support

Voor vragen of problemen:
1. Check de logs in `logs/` directory
2. Bekijk `CLEANUP_REPORT.md` voor project status
3. Controleer `.env` configuratie
4. Herstart services indien nodig

---

**🚀 Veel succes met cryptocurrency trading!**