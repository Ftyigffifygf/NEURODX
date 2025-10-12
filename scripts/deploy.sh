#!/bin/bash

# NeuroDx-MultiModal Kubernetes Deployment Script

set -e

# Configuration
NAMESPACE="neurodx-multimodal"
DOCKER_REGISTRY="neurodx"
VERSION="${VERSION:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check docker
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build main API image
    log_info "Building neurodx-api image..."
    docker build -t ${DOCKER_REGISTRY}/api:${VERSION} \
        --target production \
        -f Dockerfile .
    
    # Build MONAI Label image
    log_info "Building monai-label image..."
    docker build -t ${DOCKER_REGISTRY}/monai-label:${VERSION} \
        -f Dockerfile.monai-label .
    
    log_info "Docker images built successfully"
}

# Push images to registry
push_images() {
    if [ "$PUSH_IMAGES" = "true" ]; then
        log_info "Pushing images to registry..."
        
        docker push ${DOCKER_REGISTRY}/api:${VERSION}
        docker push ${DOCKER_REGISTRY}/monai-label:${VERSION}
        
        log_info "Images pushed successfully"
    else
        log_warn "Skipping image push (set PUSH_IMAGES=true to enable)"
    fi
}

# Create namespace
create_namespace() {
    log_info "Creating namespace..."
    
    kubectl apply -f k8s/namespace.yaml
    
    log_info "Namespace created/updated"
}

# Deploy storage components
deploy_storage() {
    log_info "Deploying storage components..."
    
    kubectl apply -f k8s/persistent-volumes.yaml
    kubectl apply -f k8s/persistent-volume-claims.yaml
    
    # Wait for PVCs to be bound
    log_info "Waiting for PVCs to be bound..."
    kubectl wait --for=condition=Bound pvc --all -n ${NAMESPACE} --timeout=300s
    
    log_info "Storage components deployed"
}

# Deploy configuration
deploy_config() {
    log_info "Deploying configuration..."
    
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secrets.yaml
    
    log_info "Configuration deployed"
}

# Deploy databases
deploy_databases() {
    log_info "Deploying databases..."
    
    kubectl apply -f k8s/postgres-deployment.yaml
    kubectl apply -f k8s/redis-deployment.yaml
    kubectl apply -f k8s/influxdb-deployment.yaml
    kubectl apply -f k8s/minio-deployment.yaml
    
    # Wait for databases to be ready
    log_info "Waiting for databases to be ready..."
    kubectl wait --for=condition=available deployment/postgres -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=available deployment/redis -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=available deployment/influxdb -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=available deployment/minio -n ${NAMESPACE} --timeout=300s
    
    log_info "Databases deployed and ready"
}

# Deploy applications
deploy_applications() {
    log_info "Deploying applications..."
    
    kubectl apply -f k8s/neurodx-api-deployment.yaml
    kubectl apply -f k8s/monai-label-deployment.yaml
    
    # Wait for applications to be ready
    log_info "Waiting for applications to be ready..."
    kubectl wait --for=condition=available deployment/neurodx-api -n ${NAMESPACE} --timeout=600s
    kubectl wait --for=condition=available deployment/monai-label -n ${NAMESPACE} --timeout=600s
    
    log_info "Applications deployed and ready"
}

# Deploy monitoring
deploy_monitoring() {
    log_info "Deploying monitoring..."
    
    kubectl apply -f k8s/prometheus-deployment.yaml
    kubectl apply -f k8s/grafana-deployment.yaml
    
    # Wait for monitoring to be ready
    log_info "Waiting for monitoring to be ready..."
    kubectl wait --for=condition=available deployment/prometheus -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=available deployment/grafana -n ${NAMESPACE} --timeout=300s
    
    log_info "Monitoring deployed and ready"
}

# Deploy ingress
deploy_ingress() {
    log_info "Deploying ingress..."
    
    kubectl apply -f k8s/ingress.yaml
    
    log_info "Ingress deployed"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Wait a bit for services to stabilize
    sleep 30
    
    # Check pod status
    log_info "Checking pod status..."
    kubectl get pods -n ${NAMESPACE}
    
    # Check service status
    log_info "Checking service status..."
    kubectl get services -n ${NAMESPACE}
    
    # Run comprehensive health check
    if kubectl get pod -l app=neurodx-api -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}' &> /dev/null; then
        POD_NAME=$(kubectl get pod -l app=neurodx-api -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}')
        log_info "Running health check script..."
        kubectl exec -n ${NAMESPACE} ${POD_NAME} -- python scripts/health_check.py
    else
        log_warn "Could not find neurodx-api pod for health check"
    fi
    
    log_info "Health checks completed"
}

# Print access information
print_access_info() {
    log_info "Deployment completed successfully!"
    echo
    log_info "Access Information:"
    echo "  Namespace: ${NAMESPACE}"
    echo "  API Service: kubectl port-forward -n ${NAMESPACE} svc/neurodx-api-service 5000:5000"
    echo "  MONAI Label: kubectl port-forward -n ${NAMESPACE} svc/monai-label-service 8000:8000"
    echo "  Grafana: kubectl port-forward -n ${NAMESPACE} svc/grafana-service 3000:3000"
    echo "  Prometheus: kubectl port-forward -n ${NAMESPACE} svc/prometheus-service 9090:9090"
    echo
    log_info "To check status: kubectl get all -n ${NAMESPACE}"
    log_info "To view logs: kubectl logs -f deployment/neurodx-api -n ${NAMESPACE}"
}

# Cleanup function
cleanup() {
    if [ "$1" = "all" ]; then
        log_warn "Cleaning up all resources..."
        kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
        log_info "Cleanup completed"
    else
        log_info "Use '$0 cleanup all' to remove all resources"
    fi
}

# Main deployment function
main() {
    case "$1" in
        "cleanup")
            cleanup $2
            ;;
        "build")
            check_prerequisites
            build_images
            ;;
        "deploy")
            check_prerequisites
            build_images
            push_images
            create_namespace
            deploy_storage
            deploy_config
            deploy_databases
            deploy_applications
            deploy_monitoring
            deploy_ingress
            run_health_checks
            print_access_info
            ;;
        "health")
            run_health_checks
            ;;
        *)
            echo "Usage: $0 {build|deploy|health|cleanup [all]}"
            echo
            echo "Commands:"
            echo "  build    - Build Docker images only"
            echo "  deploy   - Full deployment (build, push, deploy)"
            echo "  health   - Run health checks"
            echo "  cleanup  - Remove resources (use 'cleanup all' for complete removal)"
            echo
            echo "Environment variables:"
            echo "  VERSION      - Docker image version (default: latest)"
            echo "  PUSH_IMAGES  - Push images to registry (default: false)"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"