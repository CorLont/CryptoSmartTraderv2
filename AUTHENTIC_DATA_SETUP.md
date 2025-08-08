# CryptoSmartTrader V2 - Authentic Data Setup

## Current Status: DEMO MODE

Het systeem draait momenteel in demo modus met voorbeelddata. Voor echte trading en accurate voorspellingen zijn de volgende stappen nodig:

## Vereiste API Keys voor Live Data

### 1. Cryptocurrency Exchange APIs
```bash
# Kraken API (primaire exchange)
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_SECRET=your_kraken_secret

# Binance API (secundaire)
BINANCE_API_KEY=your_binance_api_key  
BINANCE_SECRET=your_binance_secret
```

### 2. AI/ML Services
```bash
# OpenAI voor sentiment analyse
OPENAI_API_KEY=your_openai_api_key

# Alternative ML services
ANTHROPIC_API_KEY=your_anthropic_key (optioneel)
```

### 3. Market Data Providers
```bash
# CoinGecko Pro API
COINGECKO_API_KEY=your_coingecko_key

# CryptoCompare API  
CRYPTOCOMPARE_API_KEY=your_cryptocompare_key
```

## Model Training Vereisten

### Machine Learning Pipeline
1. **Data Collection**: Minimaal 90 dagen historische data per coin
2. **Feature Engineering**: Technical indicators, sentiment scores, whale metrics
3. **Model Training**: LSTM, GRU, Transformer models trainen op echte data
4. **Backtesting**: Minimum 30 dagen out-of-sample testing
5. **Validation**: Live paper trading voor 2 weken voor echte deployment

### Confidence Score Criteria
- **Hoog (80-95%)**: Model getraind op >90 dagen data, backtest accuracy >85%
- **Gemiddeld (60-79%)**: Model getraind op >30 dagen data, backtest accuracy >70%  
- **Laag (<60%)**: Onvoldoende data of model niet gevalideerd

## Demo vs Live Mode

### Demo Mode (Huidige Status)
- ❌ Geen live marktdata
- ❌ Geen getrainde ML modellen
- ❌ Fictional confidence scores
- ✅ UI/UX testing mogelijk
- ✅ System architecture validation

### Live Mode (Na API Setup)
- ✅ Real-time marktdata via exchanges
- ✅ Getrainde ML modellen met validated accuracy
- ✅ Authentic confidence scores gebaseerd op backtesting
- ✅ Echte whale detection en sentiment analyse
- ✅ Paper trading en risk management

## Setup Instructies

1. **API Keys verkrijgen** van bovenstaande providers
2. **Environment variables** instellen in `.env` file
3. **Data collection** starten voor minimaal 2 weken
4. **Model training** uitvoeren op verzamelde data
5. **Backtesting** en validatie van model performance
6. **Live mode** activeren na succesvolle validatie

## Waarschuwing

⚠️ **BELANGRIJKE DISCLAIMER**: 
- Demo data is NIET geschikt voor echte trading beslissingen
- Alle confidence scores in demo mode zijn fictional
- Gebruik alleen live mode met gevalideerde modellen voor echte trades
- Investeer nooit meer dan je kunt verliezen