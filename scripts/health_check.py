#!/usr/bin/env python3
"""
Comprehensive health check script for NeuroDx-MultiModal system.
"""

import asyncio
import json
import logging
import sys
import time
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

import requests
import psycopg2
import redis
from influxdb_client import InfluxDBClient


@dataclass
class HealthCheckResult:
    """Health check result for a service."""
    service: str
    status: str
    response_time: float
    details: Dict[str, Any]
    timestamp: datetime


class HealthChecker:
    """Comprehensive health checker for all system services."""
    
    def __init__(self):
        self.results: List[HealthCheckResult] = []
        self.logger = logging.getLogger(__name__)
        
    async def check_all_services(self) -> Dict[str, Any]:
        """Check health of all services."""
        
        checks = [
            self.check_api_service(),
            self.check_monai_label(),
            self.check_postgres(),
            self.check_redis(),
            self.check_influxdb(),
            self.check_minio(),
            self.check_prometheus(),
            self.check_grafana(),
        ]
        
        # Run all checks concurrently
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        # Process results
        healthy_count = 0
        total_count = len(results)
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Health check failed: {result}")
                continue
                
            if result and result.status == "healthy":
                healthy_count += 1
                
        overall_status = "healthy" if healthy_count == total_count else "degraded"
        
        return {
            "overall_status": overall_status,
            "healthy_services": healthy_count,
            "total_services": total_count,
            "timestamp": datetime.utcnow().isoformat(),
            "services": [
                {
                    "service": r.service,
                    "status": r.status,
                    "response_time": r.response_time,
                    "details": r.details
                }
                for r in self.results if r
            ]
        }
    
    async def check_api_service(self) -> HealthCheckResult:
        """Check NeuroDx API service health."""
        
        start_time = time.time()
        
        try:
            response = requests.get(
                "http://neurodx-api:5000/api/health",
                timeout=10
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                details = response.json() if response.headers.get('content-type') == 'application/json' else {}
                result = HealthCheckResult(
                    service="neurodx-api",
                    status="healthy",
                    response_time=response_time,
                    details=details,
                    timestamp=datetime.utcnow()
                )
            else:
                result = HealthCheckResult(
                    service="neurodx-api",
                    status="unhealthy",
                    response_time=response_time,
                    details={"status_code": response.status_code},
                    timestamp=datetime.utcnow()
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            result = HealthCheckResult(
                service="neurodx-api",
                status="unhealthy",
                response_time=response_time,
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
            
        self.results.append(result)
        return result
    
    async def check_monai_label(self) -> HealthCheckResult:
        """Check MONAI Label service health."""
        
        start_time = time.time()
        
        try:
            response = requests.get(
                "http://monai-label:8000/info/",
                timeout=10
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                details = response.json()
                result = HealthCheckResult(
                    service="monai-label",
                    status="healthy",
                    response_time=response_time,
                    details=details,
                    timestamp=datetime.utcnow()
                )
            else:
                result = HealthCheckResult(
                    service="monai-label",
                    status="unhealthy",
                    response_time=response_time,
                    details={"status_code": response.status_code},
                    timestamp=datetime.utcnow()
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            result = HealthCheckResult(
                service="monai-label",
                status="unhealthy",
                response_time=response_time,
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
            
        self.results.append(result)
        return result
    
    async def check_postgres(self) -> HealthCheckResult:
        """Check PostgreSQL database health."""
        
        start_time = time.time()
        
        try:
            conn = psycopg2.connect(
                host="postgres",
                port=5432,
                database="neurodx",
                user="neurodx_user",
                password="neurodx_password",
                connect_timeout=10
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'neurodx';")
            table_count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            response_time = time.time() - start_time
            
            result = HealthCheckResult(
                service="postgres",
                status="healthy",
                response_time=response_time,
                details={
                    "version": version,
                    "table_count": table_count
                },
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            result = HealthCheckResult(
                service="postgres",
                status="unhealthy",
                response_time=response_time,
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
            
        self.results.append(result)
        return result
    
    async def check_redis(self) -> HealthCheckResult:
        """Check Redis cache health."""
        
        start_time = time.time()
        
        try:
            r = redis.Redis(host="redis", port=6379, decode_responses=True, socket_timeout=10)
            
            # Test basic operations
            r.ping()
            r.set("health_check", "test", ex=60)
            value = r.get("health_check")
            r.delete("health_check")
            
            info = r.info()
            
            response_time = time.time() - start_time
            
            result = HealthCheckResult(
                service="redis",
                status="healthy",
                response_time=response_time,
                details={
                    "version": info.get("redis_version"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human")
                },
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            result = HealthCheckResult(
                service="redis",
                status="unhealthy",
                response_time=response_time,
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
            
        self.results.append(result)
        return result
    
    async def check_influxdb(self) -> HealthCheckResult:
        """Check InfluxDB health."""
        
        start_time = time.time()
        
        try:
            response = requests.get(
                "http://influxdb:8086/ping",
                timeout=10
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 204:
                result = HealthCheckResult(
                    service="influxdb",
                    status="healthy",
                    response_time=response_time,
                    details={"version": response.headers.get("X-Influxdb-Version", "unknown")},
                    timestamp=datetime.utcnow()
                )
            else:
                result = HealthCheckResult(
                    service="influxdb",
                    status="unhealthy",
                    response_time=response_time,
                    details={"status_code": response.status_code},
                    timestamp=datetime.utcnow()
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            result = HealthCheckResult(
                service="influxdb",
                status="unhealthy",
                response_time=response_time,
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
            
        self.results.append(result)
        return result
    
    async def check_minio(self) -> HealthCheckResult:
        """Check MinIO object storage health."""
        
        start_time = time.time()
        
        try:
            response = requests.get(
                "http://minio:9000/minio/health/live",
                timeout=10
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = HealthCheckResult(
                    service="minio",
                    status="healthy",
                    response_time=response_time,
                    details={},
                    timestamp=datetime.utcnow()
                )
            else:
                result = HealthCheckResult(
                    service="minio",
                    status="unhealthy",
                    response_time=response_time,
                    details={"status_code": response.status_code},
                    timestamp=datetime.utcnow()
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            result = HealthCheckResult(
                service="minio",
                status="unhealthy",
                response_time=response_time,
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
            
        self.results.append(result)
        return result
    
    async def check_prometheus(self) -> HealthCheckResult:
        """Check Prometheus monitoring health."""
        
        start_time = time.time()
        
        try:
            response = requests.get(
                "http://prometheus:9090/-/healthy",
                timeout=10
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = HealthCheckResult(
                    service="prometheus",
                    status="healthy",
                    response_time=response_time,
                    details={},
                    timestamp=datetime.utcnow()
                )
            else:
                result = HealthCheckResult(
                    service="prometheus",
                    status="unhealthy",
                    response_time=response_time,
                    details={"status_code": response.status_code},
                    timestamp=datetime.utcnow()
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            result = HealthCheckResult(
                service="prometheus",
                status="unhealthy",
                response_time=response_time,
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
            
        self.results.append(result)
        return result
    
    async def check_grafana(self) -> HealthCheckResult:
        """Check Grafana dashboard health."""
        
        start_time = time.time()
        
        try:
            response = requests.get(
                "http://grafana:3000/api/health",
                timeout=10
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                details = response.json()
                result = HealthCheckResult(
                    service="grafana",
                    status="healthy",
                    response_time=response_time,
                    details=details,
                    timestamp=datetime.utcnow()
                )
            else:
                result = HealthCheckResult(
                    service="grafana",
                    status="unhealthy",
                    response_time=response_time,
                    details={"status_code": response.status_code},
                    timestamp=datetime.utcnow()
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            result = HealthCheckResult(
                service="grafana",
                status="unhealthy",
                response_time=response_time,
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
            
        self.results.append(result)
        return result


async def main():
    """Main health check function."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    checker = HealthChecker()
    results = await checker.check_all_services()
    
    print(json.dumps(results, indent=2))
    
    # Exit with error code if any service is unhealthy
    if results["overall_status"] != "healthy":
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())