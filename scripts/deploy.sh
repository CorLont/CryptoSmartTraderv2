#!/bin/bash
# Deployment script for CryptoSmartTrader V2
# Enterprise container deployment with health checks

set -euo pipefail

# Configuration
DOCKER_IMAGE="cryptosmarttrader:v2.0.0"
CONTAINER_NAME="cryptosmarttrader-main"
HEALTH_ENDPOINT="http://localhost:8001/health"
MAX_WAIT=300  # 5 minutes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    if [[ ! -f ".env" ]]; then
        warning ".env file not found, creating from template..."
        cp .env.example .env
        warning "Please configure .env file with your API keys before deployment"
    fi
    
    success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log "Building Docker image..."
    
    docker build -t "${DOCKER_IMAGE}" \
        --target production \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        .
    
    success "Docker image built: ${DOCKER_IMAGE}"
}

# Deploy with Docker Compose
deploy_compose() {
    log "Deploying with Docker Compose..."
    
    # Stop existing services
    docker-compose down --remove-orphans || true
    
    # Start services
    docker-compose up -d
    
    success "Services started with Docker Compose"
}

# Deploy single container
deploy_container() {
    log "Deploying single container..."
    
    # Stop and remove existing container
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
    
    # Run new container
    docker run -d \
        --name "${CONTAINER_NAME}" \
        --restart unless-stopped \
        -p 5000:5000 \
        -p 8001:8001 \
        -p 8000:8000 \
        --env-file .env \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/cache:/app/cache" \
        "${DOCKER_IMAGE}"
    
    success "Container deployed: ${CONTAINER_NAME}"
}

# Wait for service health
wait_for_health() {
    log "Waiting for service to become healthy..."
    
    local waited=0
    while [ $waited -lt $MAX_WAIT ]; do
        if curl -f -s "${HEALTH_ENDPOINT}" > /dev/null 2>&1; then
            success "Service is healthy!"
            return 0
        fi
        
        echo -n "."
        sleep 5
        waited=$((waited + 5))
    done
    
    error "Service health check failed after ${MAX_WAIT} seconds"
    return 1
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check container status
    if docker ps | grep -q "${CONTAINER_NAME}"; then
        success "Container is running"
    else
        error "Container is not running"
        return 1
    fi
    
    # Check health endpoint
    local health_response
    health_response=$(curl -s "${HEALTH_ENDPOINT}" || echo "failed")
    
    if echo "${health_response}" | grep -q "healthy"; then
        success "Health check passed"
    else
        error "Health check failed: ${health_response}"
        return 1
    fi
    
    # Check metrics endpoint
    if curl -f -s "http://localhost:8000/metrics" > /dev/null; then
        success "Metrics endpoint accessible"
    else
        warning "Metrics endpoint not accessible"
    fi
    
    # Check dashboard
    if curl -f -s "http://localhost:5000" > /dev/null; then
        success "Dashboard accessible"
    else
        warning "Dashboard not accessible"
    fi
    
    success "Deployment verification completed"
}

# Show service status
show_status() {
    log "Service Status:"
    echo
    echo "🌐 Dashboard:  http://localhost:5000"
    echo "🏥 Health API: http://localhost:8001/health"
    echo "📊 Metrics:    http://localhost:8000/metrics"
    echo
    echo "Container logs: docker logs ${CONTAINER_NAME}"
    echo "Follow logs:    docker logs -f ${CONTAINER_NAME}"
}

# Main deployment function
main() {
    local deployment_type="${1:-container}"
    
    log "Starting CryptoSmartTrader V2 deployment..."
    log "Deployment type: ${deployment_type}"
    
    check_prerequisites
    build_image
    
    case "${deployment_type}" in
        "compose")
            deploy_compose
            ;;
        "container"|*)
            deploy_container
            ;;
    esac
    
    wait_for_health
    verify_deployment
    show_status
    
    success "Deployment completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    "compose")
        main "compose"
        ;;
    "container"|"")
        main "container"
        ;;
    "build")
        check_prerequisites
        build_image
        ;;
    "status")
        verify_deployment
        show_status
        ;;
    *)
        echo "Usage: $0 [compose|container|build|status]"
        echo "  compose   - Deploy with Docker Compose (includes monitoring)"
        echo "  container - Deploy single container (default)"
        echo "  build     - Build Docker image only"
        echo "  status    - Check deployment status"
        exit 1
        ;;
esac