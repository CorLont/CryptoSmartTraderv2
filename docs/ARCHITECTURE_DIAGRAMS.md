# CryptoSmartTrader V2 - Architecture Diagrams

## System Overview Diagram

```mermaid
graph TB
    subgraph "External Interfaces"
        K[Kraken API]
        O[OpenAI API]
        P[Prometheus]
    end
    
    subgraph "Service Layer - Port Management"
        D[Dashboard Service<br/>Port 5000<br/>Streamlit]
        A[API Service<br/>Port 8001<br/>FastAPI]
        M[Metrics Service<br/>Port 8000<br/>Prometheus]
    end
    
    subgraph "Agent Layer - Process Isolation"
        SA[Sentiment Agent]
        TA[Technical Agent]
        ML[ML Predictor Agent]
        WD[Whale Detector Agent]
        RM[Risk Manager Agent]
        PO[Portfolio Optimizer Agent]
        HM[Health Monitor Agent]
        DC[Data Collector Agent]
    end
    
    subgraph "Core Infrastructure"
        DO[Distributed Orchestrator]
        DM[Data Manager]
        MM[Model Manager]
        LM[Logging Manager]
        CM[Config Manager]
    end
    
    subgraph "Data Storage"
        MD[Market Data<br/>JSON/CSV]
        ML_MOD[ML Models<br/>PKL Files]
        LOGS[Structured Logs<br/>Rotation]
        CACHE[Redis Cache<br/>Optional]
    end
    
    %% External connections
    K --> DC
    O --> SA
    M --> P
    
    %% Service layer connections
    D --> A
    A --> DO
    M --> HM
    
    %% Agent orchestration
    DO --> SA
    DO --> TA
    DO --> ML
    DO --> WD
    DO --> RM
    DO --> PO
    DO --> HM
    DO --> DC
    
    %% Core infrastructure
    SA --> DM
    TA --> DM
    ML --> MM
    DC --> DM
    HM --> LM
    DO --> CM
    
    %% Data flow
    DM --> MD
    MM --> ML_MOD
    LM --> LOGS
    DM --> CACHE
    
    style D fill:#e1f5fe
    style A fill:#f3e5f5
    style M fill:#e8f5e8
    style DO fill:#fff3e0
    style SA fill:#fce4ec
    style TA fill:#e3f2fd
    style ML fill:#f1f8e9
```

## Multi-Agent Communication Flow

```mermaid
sequenceDiagram
    participant U as User/Dashboard
    participant DO as Distributed Orchestrator
    participant DC as Data Collector
    participant SA as Sentiment Agent
    participant TA as Technical Agent
    participant ML as ML Predictor
    participant RM as Risk Manager
    participant PO as Portfolio Optimizer
    
    U->>DO: Request Market Analysis
    DO->>DC: Fetch Market Data
    DC-->>DO: Raw Market Data
    
    par Parallel Agent Processing
        DO->>SA: Analyze Sentiment
        DO->>TA: Calculate Technical Indicators
        DO->>ML: Generate Price Predictions
    end
    
    SA-->>DO: Sentiment Score + Confidence
    TA-->>DO: Technical Indicators + Signals
    ML-->>DO: Price Predictions + Uncertainty
    
    DO->>RM: Assess Risk (All Signals)
    RM-->>DO: Risk Assessment + Position Size
    
    DO->>PO: Optimize Portfolio
    PO-->>DO: Optimized Allocation
    
    DO-->>U: Consolidated Analysis + Recommendations
    
    Note over DO: 80% Confidence Gate Applied
    Note over RM: Circuit Breakers Active
    Note over PO: Kelly Criterion + Uncertainty Aware
```

## Data Flow Architecture

```mermaid
flowchart LR
    subgraph "Data Sources"
        K[Kraken API<br/>Real-time Market Data]
        N[News APIs<br/>Sentiment Data]
        S[Social Media<br/>Sentiment Signals]
    end
    
    subgraph "Data Pipeline"
        I[Data Ingestion<br/>Rate Limited]
        V[Data Validation<br/>Zero Tolerance Policy]
        T[Temporal Alignment<br/>UTC Sync]
        F[Feature Engineering<br/>Technical Indicators]
    end
    
    subgraph "ML Pipeline"
        FS[Feature Selection<br/>Leakage Detection]
        CV[Cross Validation<br/>Time Series Safe]
        TR[Model Training<br/>Multiple Horizons]
        EN[Ensemble Models<br/>Uncertainty Quantification]
    end
    
    subgraph "Decision Engine"
        CG[Confidence Gate<br/>80% Threshold]
        RR[Regime Router<br/>Market State Aware]
        RS[Risk Sizing<br/>Kelly + Uncertainty]
        EX[Execution<br/>Paper/Live Trading]
    end
    
    subgraph "Monitoring"
        DD[Drift Detection<br/>Statistical Tests]
        PM[Performance Monitoring<br/>Real-time Metrics]
        AL[Alerting<br/>GO/NO-GO Gates]
        LG[Logging<br/>Structured JSON]
    end
    
    K --> I
    N --> I
    S --> I
    
    I --> V
    V --> T
    T --> F
    
    F --> FS
    FS --> CV
    CV --> TR
    TR --> EN
    
    EN --> CG
    CG --> RR
    RR --> RS
    RS --> EX
    
    EN --> DD
    EX --> PM
    PM --> AL
    AL --> LG
    
    style V fill:#ffebee
    style CG fill:#e8f5e8
    style DD fill:#fff3e0
    style EX fill:#e3f2fd
```

## Service Deployment Architecture

```mermaid
graph TB
    subgraph "Replit Environment"
        subgraph "Port 5000 - Public Web Access"
            ST[Streamlit Dashboard<br/>--server.headless true<br/>--server.address 0.0.0.0]
        end
        
        subgraph "Port 8001 - API Services"
            FA[FastAPI Health Service<br/>uvicorn + async]
            HE[/health endpoint]
            HD[/health/detailed endpoint]
            AD[/api/docs - OpenAPI]
        end
        
        subgraph "Port 8000 - Monitoring"
            PM[Prometheus Metrics Server]
            ME[/metrics endpoint]
            MC[Custom Business Metrics]
        end
    end
    
    subgraph "Process Management"
        UV[UV Package Manager<br/>uv sync]
        BG[Background Processes<br/>service1 & service2 & service3]
        WT[Wait Coordination<br/>& wait]
        SH[Signal Handling<br/>Graceful Shutdown]
    end
    
    subgraph "Health Monitoring"
        HS[Health Status Checks]
        AM[Automatic Monitoring]
        RP[Replit Port Panel]
        LR[Log Rotation]
    end
    
    UV --> BG
    BG --> ST
    BG --> FA
    BG --> PM
    BG --> WT
    
    ST --> HS
    FA --> HE
    FA --> HD
    FA --> AD
    PM --> ME
    PM --> MC
    
    HS --> AM
    AM --> RP
    WT --> SH
    SH --> LR
    
    style ST fill:#e1f5fe
    style FA fill:#f3e5f5
    style PM fill:#e8f5e8
    style UV fill:#fff3e0
```

## Agent State Machine

```mermaid
stateDiagram-v2
    [*] --> Initialized
    
    Initialized --> Starting: start()
    Starting --> Running: successful_startup()
    Starting --> Failed: startup_error()
    
    Running --> Processing: new_data()
    Processing --> Running: processing_complete()
    Processing --> Error: processing_error()
    
    Running --> Paused: pause_signal()
    Paused --> Running: resume_signal()
    
    Error --> Running: error_recovered()
    Error --> Failed: error_critical()
    
    Running --> Stopping: stop_signal()
    Paused --> Stopping: stop_signal()
    Error --> Stopping: stop_signal()
    
    Stopping --> Stopped: graceful_shutdown()
    Failed --> Stopped: force_shutdown()
    
    Stopped --> [*]
    
    note right of Processing
        Circuit Breakers Active
        Timeout Protection
        Resource Monitoring
    end note
    
    note right of Error
        Exponential Backoff
        Retry Logic
        Fallback Strategies
    end note
```

## Confidence Gate Decision Tree

```mermaid
flowchart TD
    S[Signal Generated] --> C{Confidence > 80%?}
    
    C -->|Yes| R{Regime Check}
    C -->|No| Block[Block Signal<br/>Log Low Confidence]
    
    R -->|Bull Market| B{Bull Strategy}
    R -->|Bear Market| Bear{Bear Strategy}
    R -->|Sideways| Side{Sideways Strategy}
    
    B -->|Confirmed| Size1[Position Sizing<br/>Kelly + Uncertainty]
    Bear -->|Confirmed| Size2[Position Sizing<br/>Defensive]
    Side -->|Confirmed| Size3[Position Sizing<br/>Range Trading]
    
    B -->|Rejected| Block
    Bear -->|Rejected| Block
    Side -->|Rejected| Block
    
    Size1 --> Risk{Risk Check}
    Size2 --> Risk
    Size3 --> Risk
    
    Risk -->|Pass| Execute[Execute Trade<br/>Paper/Live]
    Risk -->|Fail| Block
    
    Execute --> Monitor[Monitor Performance<br/>Update Metrics]
    Block --> Log[Log Decision<br/>Update Statistics]
    
    Monitor --> Feedback[Model Feedback<br/>Performance Update]
    Log --> Feedback
    
    style C fill:#ffebee
    style Risk fill:#e8f5e8
    style Execute fill:#e3f2fd
    style Block fill:#fff3e0
```

## Error Handling & Recovery Flow

```mermaid
flowchart TD
    E[Error Detected] --> T{Error Type}
    
    T -->|Network| N[Network Error<br/>Retry with Backoff]
    T -->|Data| D[Data Error<br/>Validate & Clean]
    T -->|Model| M[Model Error<br/>Fallback to Previous]
    T -->|System| S[System Error<br/>Circuit Breaker]
    
    N --> R1{Retry Success?}
    D --> R2{Data Recoverable?}
    M --> R3{Fallback Available?}
    S --> R4{System Recoverable?}
    
    R1 -->|Yes| Success[Continue Processing]
    R1 -->|No| Degrade1[Graceful Degradation<br/>Reduce Functionality]
    
    R2 -->|Yes| Success
    R2 -->|No| Alert1[Alert: Data Quality Issue<br/>Switch to Backup Source]
    
    R3 -->|Yes| Success
    R3 -->|No| Alert2[Alert: Model Failure<br/>Switch to Paper Trading]
    
    R4 -->|Yes| Success
    R4 -->|No| Shutdown[Graceful Shutdown<br/>Preserve State]
    
    Degrade1 --> Monitor[Monitor for Recovery]
    Alert1 --> Monitor
    Alert2 --> Monitor
    
    Monitor --> Auto{Auto Recovery?}
    Auto -->|Yes| Success
    Auto -->|No| Manual[Manual Intervention Required]
    
    Shutdown --> Restart[Restart Procedure<br/>Health Checks]
    Restart --> Success
    
    style E fill:#ffebee
    style Success fill:#e8f5e8
    style Shutdown fill:#fff3e0
    style Manual fill:#fce4ec
```

---

## Component Interaction Matrix

| Component | Data Collector | Sentiment Agent | Technical Agent | ML Predictor | Risk Manager | Portfolio Optimizer | Health Monitor |
|-----------|---------------|-----------------|-----------------|--------------|--------------|-------------------|----------------|
| **Data Collector** | - | Market Data | Market Data | Market Data | Market Data | Market Data | Status |
| **Sentiment Agent** | - | - | Sentiment Scores | Sentiment Features | Sentiment Risk | Sentiment Weights | Agent Health |
| **Technical Agent** | - | - | - | Technical Features | Technical Signals | Technical Weights | Agent Health |
| **ML Predictor** | - | Prediction Context | Technical Context | - | Predictions + Uncertainty | Expected Returns | Model Performance |
| **Risk Manager** | Risk Metrics | Sentiment Risk | Technical Risk | Model Risk | - | Risk Constraints | Risk Status |
| **Portfolio Optimizer** | Portfolio Data | Sentiment Alpha | Technical Alpha | Return Predictions | Risk Constraints | - | Optimization Status |
| **Health Monitor** | Data Quality | Agent Status | Agent Status | Model Performance | Risk Metrics | Portfolio Health | - |

---

## Technology Stack Detail

```mermaid
graph TB
    subgraph "Frontend Layer"
        ST[Streamlit 1.28+]
        PL[Plotly 5.17+]
        PD[Pandas 2.1+]
    end
    
    subgraph "API Layer"
        FA[FastAPI 0.104+]
        UV[Uvicorn 0.24+]
        PY[Pydantic 2.5+]
    end
    
    subgraph "ML/AI Layer"
        SK[Scikit-learn 1.3+]
        XG[XGBoost 2.0+]
        NP[NumPy 1.24+]
        OP[OpenAI 1.3+]
    end
    
    subgraph "Data Layer"
        CC[CCXT 4.1+]
        RE[Requests 2.31+]
        AH[aiohttp 3.9+]
    end
    
    subgraph "Infrastructure"
        PR[Prometheus Client 0.19+]
        LO[Python Logging]
        UV_PKG[UV Package Manager]
        TH[Threading/Asyncio]
    end
    
    style ST fill:#e1f5fe
    style FA fill:#f3e5f5  
    style SK fill:#e8f5e8
    style CC fill:#fff3e0
    style PR fill:#fce4ec
```

This architecture ensures:
- **Scalability:** Independent service scaling
- **Reliability:** Circuit breakers and health monitoring
- **Maintainability:** Clear separation of concerns  
- **Performance:** Async processing and caching
- **Observability:** Comprehensive logging and metrics