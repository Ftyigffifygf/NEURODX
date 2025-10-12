#!/usr/bin/env python3
"""
NeuroDx-MultiModal Monitoring Service
Runs continuous system monitoring and alerting.
"""

import asyncio
import signal
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.monitoring.system_monitor import system_monitor
from src.utils.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)


class MonitoringService:
    """Main monitoring service orchestrator."""
    
    def __init__(self):
        self.settings = get_settings()
        self.running = False
        self.tasks = []
        
    async def start(self):
        """Start the monitoring service."""
        
        logger.info("Starting NeuroDx-MultiModal Monitoring Service")
        
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Start system monitoring
            monitoring_task = asyncio.create_task(
                system_monitor.start_monitoring(interval=30)
            )
            self.tasks.append(monitoring_task)
            
            # Start health check service
            health_check_task = asyncio.create_task(
                self._run_periodic_health_checks()
            )
            self.tasks.append(health_check_task)
            
            # Start metrics cleanup service
            cleanup_task = asyncio.create_task(
                self._run_metrics_cleanup()
            )
            self.tasks.append(cleanup_task)
            
            # Wait for all tasks
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error in monitoring service: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the monitoring service."""
        
        if not self.running:
            return
            
        logger.info("Stopping NeuroDx-MultiModal Monitoring Service")
        
        self.running = False
        system_monitor.stop_monitoring()
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("Monitoring service stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        
        # Create a task to stop the service
        loop = asyncio.get_event_loop()
        loop.create_task(self.stop())
    
    async def _run_periodic_health_checks(self):
        """Run periodic comprehensive health checks."""
        
        logger.info("Starting periodic health checks")
        
        while self.running:
            try:
                # Run comprehensive health check
                await self._comprehensive_health_check()
                
                # Wait 5 minutes between health checks
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic health check: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _comprehensive_health_check(self):
        """Perform comprehensive health check."""
        
        try:
            # Get current system metrics
            metrics = system_monitor.get_current_metrics()
            active_alerts = system_monitor.get_active_alerts()
            service_health = system_monitor.get_service_health()
            
            # Log health summary
            healthy_services = sum(
                1 for health in service_health.values() 
                if health.get('status') == 'healthy'
            )
            total_services = len(service_health)
            
            logger.info(
                f"Health Check - Services: {healthy_services}/{total_services} healthy, "
                f"Active Alerts: {len(active_alerts)}, "
                f"CPU: {metrics.get('cpu_usage', 0):.1f}%, "
                f"Memory: {metrics.get('memory_usage', 0):.1f}%"
            )
            
            # Check for critical conditions
            critical_alerts = [
                alert for alert in active_alerts 
                if alert.get('severity') == 'critical'
            ]
            
            if critical_alerts:
                logger.error(f"CRITICAL: {len(critical_alerts)} critical alerts active")
                for alert in critical_alerts:
                    logger.error(f"  - {alert.get('service')}: {alert.get('message')}")
            
            # Check for unhealthy services
            unhealthy_services = [
                name for name, health in service_health.items()
                if health.get('status') == 'unhealthy'
            ]
            
            if unhealthy_services:
                logger.warning(f"Unhealthy services: {', '.join(unhealthy_services)}")
            
        except Exception as e:
            logger.error(f"Error in comprehensive health check: {e}")
    
    async def _run_metrics_cleanup(self):
        """Run periodic metrics cleanup."""
        
        logger.info("Starting metrics cleanup service")
        
        while self.running:
            try:
                # Clean up old alert history (keep last 1000 alerts)
                if len(system_monitor.alert_history) > 1000:
                    system_monitor.alert_history = system_monitor.alert_history[-1000:]
                    logger.info("Cleaned up old alert history")
                
                # Wait 1 hour between cleanups
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics cleanup: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error


async def main():
    """Main entry point."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/monitoring.log')
        ]
    )
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("NeuroDx-MultiModal Monitoring Service")
    logger.info(f"Started at: {datetime.utcnow().isoformat()}")
    logger.info("=" * 60)
    
    # Create and start monitoring service
    service = MonitoringService()
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error in monitoring service: {e}")
        sys.exit(1)
    finally:
        logger.info("Monitoring service shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())