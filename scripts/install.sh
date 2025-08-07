#!/bin/bash
# CryptoSmartTrader V2 - Linux Installation Script

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.11"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
LOG_DIR="${PROJECT_DIR}/logs"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_system() {
    log_info "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    log_info "Operating system: $OS"
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python 3.11+ is required but not found"
        exit 1
    fi
    
    PYTHON_VERSION_INSTALLED=$($PYTHON_CMD --version | grep -oE '[0-9]+\.[0-9]+')
    log_info "Python version: $PYTHON_VERSION_INSTALLED"
    
    # Check minimum Python version
    if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
        log_error "Python 3.10+ is required"
        exit 1
    fi
    
    # Check pip
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        log_error "pip is required but not found"
        exit 1
    fi
    
    # Check git
    if ! command -v git &> /dev/null; then
        log_warning "Git not found - some features may be limited"
    fi
    
    log_success "System requirements check passed"
}

install_system_dependencies() {
    log_info "Installing system dependencies..."
    
    if [[ "$OS" == "linux" ]]; then
        # Detect package manager
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y \
                python3-venv \
                python3-dev \
                build-essential \
                curl \
                wget \
                git \
                htop \
                jq
        elif command -v yum &> /dev/null; then
            sudo yum install -y \
                python3-venv \
                python3-devel \
                gcc \
                gcc-c++ \
                curl \
                wget \
                git \
                htop \
                jq
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y \
                python3-venv \
                python3-devel \
                gcc \
                gcc-c++ \
                curl \
                wget \
                git \
                htop \
                jq
        else
            log_warning "Unknown package manager - please install dependencies manually"
        fi
    elif [[ "$OS" == "macos" ]]; then
        if command -v brew &> /dev/null; then
            brew install python@3.11 git curl wget jq
        else
            log_warning "Homebrew not found - please install dependencies manually"
        fi
    fi
    
    log_success "System dependencies installed"
}

create_virtual_environment() {
    log_info "Creating Python virtual environment..."
    
    if [[ -d "$VENV_DIR" ]]; then
        log_warning "Virtual environment already exists - removing..."
        rm -rf "$VENV_DIR"
    fi
    
    $PYTHON_CMD -m venv "$VENV_DIR"
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    log_success "Virtual environment created and activated"
}

install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Ensure virtual environment is activated
    source "$VENV_DIR/bin/activate"
    
    # Install core dependencies
    pip install \
        streamlit>=1.28.0 \
        fastapi>=0.104.0 \
        uvicorn>=0.24.0 \
        pydantic>=2.5.0 \
        pydantic-settings>=2.1.0 \
        pandas>=2.1.0 \
        numpy>=1.24.0 \
        plotly>=5.17.0 \
        scikit-learn>=1.3.0 \
        xgboost>=2.0.0 \
        textblob>=0.17.1 \
        ccxt>=4.1.0 \
        dependency-injector>=4.41.0 \
        aiohttp>=3.9.0 \
        tenacity>=8.2.0 \
        psutil>=5.9.0 \
        prometheus-client>=0.19.0 \
        python-json-logger>=2.0.0 \
        hvac>=2.0.0 \
        openai>=1.3.0
    
    # Install development dependencies
    pip install \
        pytest>=7.4.0 \
        pytest-cov>=4.1.0 \
        pytest-asyncio>=0.21.0 \
        black>=23.11.0 \
        isort>=5.12.0 \
        flake8>=6.1.0 \
        mypy>=1.7.0 \
        pre-commit>=3.5.0 \
        bandit>=1.7.0 \
        safety>=2.3.0
    
    log_success "Python dependencies installed"
}

setup_directories() {
    log_info "Setting up directory structure..."
    
    # Create necessary directories
    mkdir -p "$LOG_DIR"
    mkdir -p "${PROJECT_DIR}/data"
    mkdir -p "${PROJECT_DIR}/models"
    mkdir -p "${PROJECT_DIR}/temp"
    mkdir -p "${PROJECT_DIR}/config"
    
    # Set permissions
    chmod 755 "$LOG_DIR"
    chmod 755 "${PROJECT_DIR}/data"
    chmod 755 "${PROJECT_DIR}/models"
    chmod 755 "${PROJECT_DIR}/temp"
    
    log_success "Directory structure created"
}

configure_environment() {
    log_info "Configuring environment..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f "${PROJECT_DIR}/.env" ]]; then
        cat > "${PROJECT_DIR}/.env" << EOF
# CryptoSmartTrader V2 Environment Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
PYTHONPATH=${PROJECT_DIR}

# API Keys (configure as needed)
OPENAI_API_KEY=
KRAKEN_API_KEY=
KRAKEN_SECRET=
BINANCE_API_KEY=
BINANCE_SECRET=

# Database (if using external database)
DATABASE_URL=

# Security
SECRET_KEY=$(openssl rand -hex 32)
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0

# Performance
MAX_WORKERS=4
MEMORY_LIMIT_GB=8
ENABLE_GPU=false
EOF
        log_info "Created .env configuration file"
    else
        log_info ".env file already exists"
    fi
    
    log_success "Environment configured"
}

setup_pre_commit() {
    log_info "Setting up pre-commit hooks..."
    
    source "$VENV_DIR/bin/activate"
    
    # Create pre-commit config if it doesn't exist
    if [[ ! -f "${PROJECT_DIR}/.pre-commit-config.yaml" ]]; then
        cat > "${PROJECT_DIR}/.pre-commit-config.yaml" << EOF
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
EOF
    fi
    
    # Install pre-commit hooks
    pre-commit install
    
    log_success "Pre-commit hooks configured"
}

run_tests() {
    log_info "Running system tests..."
    
    source "$VENV_DIR/bin/activate"
    
    # Basic import test
    $PYTHON_CMD -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}')

try:
    from config.validation import SystemConfiguration
    from config.security import SecurityManager
    from utils.orchestrator import SystemOrchestrator
    print('✓ Core modules imported successfully')
except Exception as e:
    print(f'✗ Import test failed: {e}')
    sys.exit(1)
"
    
    # Configuration validation test
    $PYTHON_CMD -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}')

try:
    from config.validation import get_default_configuration, validate_configuration
    config = get_default_configuration()
    validated = validate_configuration(config)
    print('✓ Configuration validation passed')
except Exception as e:
    print(f'✗ Configuration test failed: {e}')
    sys.exit(1)
"
    
    log_success "System tests passed"
}

create_startup_scripts() {
    log_info "Creating startup scripts..."
    
    # Create start script
    cat > "${PROJECT_DIR}/start.sh" << EOF
#!/bin/bash
# CryptoSmartTrader V2 Startup Script

set -euo pipefail

PROJECT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="\${PROJECT_DIR}/venv"

# Activate virtual environment
source "\${VENV_DIR}/bin/activate"

# Set Python path
export PYTHONPATH="\${PROJECT_DIR}:\${PYTHONPATH:-}"

# Load environment variables
if [[ -f "\${PROJECT_DIR}/.env" ]]; then
    source "\${PROJECT_DIR}/.env"
fi

# Start the application
echo "Starting CryptoSmartTrader V2..."
echo "Access the application at: http://localhost:5000"

streamlit run app.py --server.port 5000 --server.address 0.0.0.0
EOF
    
    # Create API server script
    cat > "${PROJECT_DIR}/start_api.sh" << EOF
#!/bin/bash
# CryptoSmartTrader V2 API Server Script

set -euo pipefail

PROJECT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="\${PROJECT_DIR}/venv"

# Activate virtual environment
source "\${VENV_DIR}/bin/activate"

# Set Python path
export PYTHONPATH="\${PROJECT_DIR}:\${PYTHONPATH:-}"

# Load environment variables
if [[ -f "\${PROJECT_DIR}/.env" ]]; then
    source "\${PROJECT_DIR}/.env"
fi

# Start the API server
echo "Starting CryptoSmartTrader V2 API Server..."
echo "API documentation available at: http://localhost:8001/docs"

uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
EOF
    
    # Make scripts executable
    chmod +x "${PROJECT_DIR}/start.sh"
    chmod +x "${PROJECT_DIR}/start_api.sh"
    
    log_success "Startup scripts created"
}

main() {
    log_info "Starting CryptoSmartTrader V2 installation..."
    
    check_system
    install_system_dependencies
    create_virtual_environment
    install_python_dependencies
    setup_directories
    configure_environment
    setup_pre_commit
    run_tests
    create_startup_scripts
    
    log_success "Installation completed successfully!"
    
    echo
    echo -e "${GREEN}🚀 CryptoSmartTrader V2 is ready!${NC}"
    echo
    echo "To start the application:"
    echo "  ./start.sh"
    echo
    echo "To start the API server:"
    echo "  ./start_api.sh"
    echo
    echo "To run tests:"
    echo "  source venv/bin/activate && pytest"
    echo
    echo "Configuration file: .env"
    echo "Logs directory: logs/"
    echo
}

# Run main function
main "$@"