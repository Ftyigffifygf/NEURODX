#!/bin/bash

# NeuroDx-MultiModal Monitoring Setup Script
# This script sets up comprehensive monitoring and alerting for the system

set -e

echo "=========================================="
echo "NeuroDx-MultiModal Monitoring Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
print_status "Creating monitoring directories..."
mkdir -p logs
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/prometheus/data
mkdir -p monitoring/alertmanager/data

# Set proper permissions
print_status "Setting directory permissions..."
chmod 755 logs
chmod -R 755 monitoring/

# Create Grafana provisioning configuration
print_status "Creating Grafana provisioning configuration..."

cat > monitoring/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

cat > monitoring/grafana/provisioning/dashboards/dashboards.yml << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

# Copy dashboard files
print_status "Copying dashboard configurations..."
cp monitoring/grafana/dashboards/*.json monitoring/grafana/provisioning/dashboards/

# Create Prometheus configuration if it doesn't exist
if [ ! -f monitoring/prometheus.yml ]; then
    print_status "Creating default Prometheus configuration..."
    cp monitoring/prometheus.yml.example monitoring/prometheus.yml 2>/dev/null || true
fi

# Create AlertManager configuration if it doesn't exist
if [ ! -f monitoring/alertmanager.yml ]; then
    print_status "Creating default AlertManager configuration..."
    cp monitoring/alertmanager.yml.example monitoring/alertmanager.yml 2>/dev/null || true
fi

# Install Python dependencies for monitoring
print_status "Installing Python monitoring dependencies..."
pip install -r requirements.txt

# Create systemd service file
print_status "Creating systemd service file..."
sudo cp scripts/neurodx-monitoring.service /etc/systemd/system/
sudo systemctl daemon-reload

# Make monitoring script executable
chmod +x scripts/monitoring_service.py
chmod +x scripts/health_check.py

# Create monitoring configuration
print_status "Creating monitoring configuration..."
python3 -c "
from src.config.monitoring_config import DEFAULT_MONITORING_CONFIG, PROMETHEUS_ALERT_RULES
import json
import yaml

# Save monitoring config
with open('monitoring/monitoring_config.json', 'w') as f:
    json.dump(DEFAULT_MONITORING_CONFIG.__dict__, f, indent=2, default=str)

# Save Prometheus alert rules
with open('monitoring/rules/neurodx-alerts.yml', 'w') as f:
    yaml.dump(PROMETHEUS_ALERT_RULES, f, default_flow_style=False)

print('Monitoring configuration created successfully')
"

# Test monitoring components
print_status "Testing monitoring components..."

# Test health check script
print_status "Testing health check script..."
python3 scripts/health_check.py --test || print_warning "Health check test failed (services may not be running)"

# Test system monitor
print_status "Testing system monitor..."
python3 -c "
from src.services.monitoring.system_monitor import system_monitor
import asyncio

async def test_monitor():
    try:
        metrics = await system_monitor.collect_system_metrics()
        print(f'System monitor test successful - CPU: {metrics.cpu_usage:.1f}%')
        return True
    except Exception as e:
        print(f'System monitor test failed: {e}')
        return False

result = asyncio.run(test_monitor())
exit(0 if result else 1)
" || print_warning "System monitor test failed"

# Create monitoring startup script
print_status "Creating monitoring startup script..."
cat > scripts/start_monitoring.sh << 'EOF'
#!/bin/bash

# Start NeuroDx-MultiModal monitoring stack

echo "Starting NeuroDx-MultiModal monitoring stack..."

# Start monitoring services with Docker Compose
docker-compose -f docker-compose.prod.yml up -d prometheus grafana alertmanager

# Wait for services to start
sleep 10

# Start Python monitoring service
sudo systemctl start neurodx-monitoring
sudo systemctl enable neurodx-monitoring

echo "Monitoring stack started successfully!"
echo ""
echo "Access points:"
echo "- Grafana: http://localhost:3000 (admin/admin)"
echo "- Prometheus: http://localhost:9090"
echo "- AlertManager: http://localhost:9093"
echo ""
echo "Check monitoring service status:"
echo "sudo systemctl status neurodx-monitoring"
EOF

chmod +x scripts/start_monitoring.sh

# Create monitoring stop script
print_status "Creating monitoring stop script..."
cat > scripts/stop_monitoring.sh << 'EOF'
#!/bin/bash

# Stop NeuroDx-MultiModal monitoring stack

echo "Stopping NeuroDx-MultiModal monitoring stack..."

# Stop Python monitoring service
sudo systemctl stop neurodx-monitoring

# Stop monitoring services
docker-compose -f docker-compose.prod.yml stop prometheus grafana alertmanager

echo "Monitoring stack stopped successfully!"
EOF

chmod +x scripts/stop_monitoring.sh

# Create monitoring status script
print_status "Creating monitoring status script..."
cat > scripts/monitoring_status.sh << 'EOF'
#!/bin/bash

# Check NeuroDx-MultiModal monitoring status

echo "=========================================="
echo "NeuroDx-MultiModal Monitoring Status"
echo "=========================================="

# Check systemd service
echo "Python Monitoring Service:"
sudo systemctl status neurodx-monitoring --no-pager -l

echo ""
echo "Docker Services:"
docker-compose -f docker-compose.prod.yml ps prometheus grafana alertmanager

echo ""
echo "Recent Logs:"
echo "Monitoring Service Logs (last 10 lines):"
sudo journalctl -u neurodx-monitoring -n 10 --no-pager

echo ""
echo "Health Check:"
python3 scripts/health_check.py
EOF

chmod +x scripts/monitoring_status.sh

print_status "Monitoring setup completed successfully!"
print_status ""
print_status "Next steps:"
print_status "1. Review and customize monitoring/alertmanager.yml for your notification preferences"
print_status "2. Review and customize monitoring/prometheus.yml for your environment"
print_status "3. Start the monitoring stack: ./scripts/start_monitoring.sh"
print_status "4. Check monitoring status: ./scripts/monitoring_status.sh"
print_status "5. Access Grafana at http://localhost:3000 (admin/admin)"
print_status ""
print_status "For production deployment:"
print_status "- Update email/Slack/PagerDuty configurations in alertmanager.yml"
print_status "- Set up proper SSL certificates for Grafana"
print_status "- Configure firewall rules for monitoring ports"
print_status "- Set up log rotation for monitoring logs"