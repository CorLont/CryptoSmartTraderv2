# CryptoSmartTrader V2 - Setup Guide

## Snelle Start vanaf je Workstation

### Vereisten
- Python 3.11 of hoger
- Git (voor code download)
- Minimaal 8GB RAM (16GB aanbevolen)
- Internet verbinding voor cryptocurrency data

### Stap 1: Code Downloaden
```bash
# Download de code vanaf Replit of GitHub
git clone [repository-url]
cd CryptoSmartTrader-V2
```

### Stap 2: Dependencies Installeren
```bash
# Installeer alle benodigde packages
pip install -r requirements.txt

# Of gebruik pipenv/poetry indien gewenst
pipenv install
# of
poetry install
```

### Stap 3: Configuratie
```bash
# Kopieer configuratie template
cp .env.example .env

# Bewerk .env bestand met je voorkeuren:
# - OPENAI_API_KEY=jouw_openai_key (optioneel, voor geavanceerde sentiment analyse)
# - EXCHANGE_API_KEYS (optioneel, voor live trading)
```

### Stap 4: Systeem Starten
```bash
# Start het complete systeem
streamlit run app.py --server.port 5000

# Of gebruik de productie configuratie
streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 5000
```

### Stap 5: Dashboard Toegang
- Open je browser naar: http://localhost:5000
- Selecteer gewenste dashboard uit het menu
- Systeem start automatisch alle analyses

## Dashboard Overzicht

### 🎯 Main Dashboard
**Gebruik:** Snelle overzicht en real-time monitoring
- Systeem status en health metrics
- Top alpha opportunities 
- Agent performance indicators
- Real-time trading opportunities

### 🌍 Comprehensive Market Dashboard
**Gebruik:** Volledige marktanalyse
- 1457+ trading pairs coverage
- Multi-timeframe analyse (1m tot 1M)
- Filtering op confidence levels
- Sorteer op verwachte returns

### 🧠 Advanced Analytics Dashboard
**Gebruik:** Diepgaande technische analyse
- 50+ technische indicatoren
- Backtest resultaten
- Performance metrics
- Risk assessment tools

### 🤖 AI/ML Engine Dashboard
**Gebruik:** Machine learning systemen
- Multi-horizon predictions (1H, 24H, 7D, 30D)
- Model performance tracking
- Feature engineering insights
- Confidence scoring

### 🚀 ML/AI Differentiators Dashboard
**Gebruik:** Geavanceerde AI features
- Deep learning modellen (LSTM, Transformers)
- Multi-modal feature fusion
- Explainable AI (SHAP explanations)
- AutoML optimization

### 🧠 Crypto AI System Dashboard
**Gebruik:** Complete AI pipeline management
- Daily analysis scheduling
- Data quality monitoring
- System coordination
- Alpha seeking pipeline

## Productie Tips

### Performance Optimalisatie
```bash
# Voor betere performance op je workstation:
export CUDA_VISIBLE_DEVICES=0  # Als je GPU hebt
export NUMBA_NUM_THREADS=4     # CPU threads optimalisatie
```

### Monitoring & Logs
```bash
# Bekijk real-time logs
tail -f logs/cryptotrader.json

# Check system health
python scripts/production_health_check.py

# Fix eventuele issues
python scripts/fix_production_errors.py
```

### Data Storage
- Cache wordt opgeslagen in `/cache/` directory
- Logs in `/logs/` directory  
- Models in `/models/` directory
- Backup configuraties in `/config/`

## Gebruik voor Alpha Seeking (500%+ Returns)

### Dagelijkse Routine
1. **Start systeem** met `streamlit run app.py`
2. **Check Comprehensive Market** voor nieuwe opportunities
3. **Gebruik ML/AI Differentiators** voor deep analysis
4. **Monitor via Crypto AI System** voor pipeline status
5. **Review Advanced Analytics** voor risk assessment

### Key Filters
- **Confidence Level:** Minimaal 80%
- **Expected Return:** Minimaal 100% (30-day)
- **Data Quality:** Alleen real data, geen synthetic
- **Timeframes:** Focus op 1H, 24H, 7D, 30D predictions

### Alert Setup
Het systeem stuurt automatisch alerts bij:
- High-confidence opportunities (≥80%)
- Significant price movements
- Whale activity detection
- System health issues

## Troubleshooting

### Veelvoorkomende Problemen
1. **Port in gebruik:** Wijzig port in command naar `--server.port 5001`
2. **Memory errors:** Verhoog swap space of gebruik minder threads
3. **API rate limits:** Systeem heeft ingebouwde rate limiting
4. **Exchange connectivity:** Binance kan blocked zijn in sommige regio's

### Support
- Check logs in `/logs/` directory
- Run health check script
- Review configuration in `config.json`
- Documentatie in `/docs/` directory

## Production Ready Features
- Automatic error recovery
- Graceful shutdown handling
- Data validation (zero dummy data tolerance)
- Multi-exchange failover
- Comprehensive monitoring
- Real-time opportunity detection

Het systeem is volledig production-ready en analyseert continu alle beschikbare cryptocurrencies voor alpha opportunities.