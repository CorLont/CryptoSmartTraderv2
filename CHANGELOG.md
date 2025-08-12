# Changelog

Alle belangrijke wijzigingen aan dit project worden gedocumenteerd in dit bestand.

Het format is gebaseerd op [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
en dit project volgt [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Toegevoegd
- Enterprise src/ package layout met proper import structuur
- Matrix CI testing voor Python 3.11 en 3.12
- Comprehensive coverage reporting (XML/HTML artifacts)  
- CODEOWNERS bestand voor code review governance
- SECURITY.md met vulnerability reporting procedures
- Pull request template met uitgebreide checklist
- UV dependency caching voor CI performance
- Hard CI enforcement (geen || true fallbacks meer)

### Gewijzigd
- Migratie naar src/cryptosmarttrader/ package structuur
- Coverage scope aangepast naar src/ layout
- pyproject.toml bijgewerkt voor setuptools build system
- CI workflow uitgebreid met performance optimalisaties

### Gerepareerd
- Import shadowing issues door src/ layout
- Build system configuratie voor proper packaging
- LSP diagnostics voor clean type checking

## [2.0.0] - 2025-01-12

### Toegevoegd
- **Enterprise Architecture**: Complete multi-service deployment architecture
  - Dashboard service (poort 5000)
  - API service (poort 8001) 
  - Metrics service (poort 8000)
- **GitHub Actions CI/CD**: Comprehensive workflow met UV package management
  - Ruff/Black/MyPy linting met hard enforcement
  - pytest met strict settings en security scanning
  - Concurrency controls en efficiency optimalisaties
- **Configuration Hardening**: Moderne Pydantic v2 settings
  - Centralized type validation en fail-fast startup
  - Dutch language support throughout system
  - Clean logging opties en main entry points
- **API Contract**: FastAPI app factory pattern
  - Health endpoints met structured monitoring
  - E2E testing met httpx.AsyncClient
  - Dependency injection en CORS middleware
- **Repository Infrastructure**: Enterprise-grade development setup
  - Comprehensive .gitignore met artifact exclusions
  - Strict pytest.ini met marker system en coverage targets
  - Multi-service Replit configuration met process isolation

### Gewijzigd
- **Clean Architecture**: Enterprise src/ layout geïmplementeerd
  - Domain interfaces (ports) en swappable adapters
  - Package structuur voorkomt import shadowing
  - Dependency inversion voor testable, maintainable code
- **Multi-Agent System**: Distributed architecture met 8 specialized agents
  - Data Collector, Sentiment Analyzer, Technical Analyzer, ML Predictor
  - Whale Detector, Risk Manager, Portfolio Optimizer, Health Monitor
  - Circuit breakers, exponential backoff, automatic restart capabilities

### Technische Details
- **Data Pipeline**: Real-time analysis van 1457+ cryptocurrencies via Kraken API
- **ML/AI Stack**: PyTorch LSTM/GRU/Transformer modellen, scikit-learn, XGBoost
- **Risk Management**: Strict 80% confidence gate, shadow trading, portfolio optimization
- **Monitoring**: Enterprise JSON logging, Prometheus metrics, health grading system
- **Security**: Comprehensive audit systeem, secrets management, vulnerability scanning

## [1.0.0] - Legacy
### Opmerking
Versie 1.x.x is deprecated en wordt niet langer ondersteund. Alle development focus ligt op 2.x.x versies.

---

## Versioning Guidelines

Dit project gebruikt [Semantic Versioning](https://semver.org/):

- **MAJOR** versie voor incompatible API wijzigingen
- **MINOR** versie voor backward-compatible functionaliteit
- **PATCH** versie voor backward-compatible bug fixes

### Versie Branches
- `main`: Stable release branch
- `develop`: Development integration branch  
- `feature/*`: Feature development branches
- `hotfix/*`: Critical production fixes

### Release Process
1. Create release branch from develop
2. Update version in pyproject.toml
3. Update CHANGELOG.md met release notes
4. PR naar main met required reviews
5. Tag release met git tag v2.x.x
6. Deploy via Replit deployments
7. Merge back naar develop