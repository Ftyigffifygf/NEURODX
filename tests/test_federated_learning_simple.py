"""
Simple tests for MONAI Federated Learning Infrastructure

Basic tests that don't require complex MONAI FL imports.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any

from src.services.federated_learning.aggregation_engine import (
    ModelAggregationEngine, ModelUpdate
)
from src.config.federated_learning_config import (
    FederatedServerConfig, FederatedClientConfig
)


class TestModelUpdate:
    """Test ModelUpdate data class"""
    
    def test_model_update_creation(self):
        """Test creating a model update"""
        model_params = {
            'layer.weight': torch.randn(5, 3),
            'layer.bias': torch.randn(5)
        }
        metrics = {'loss': 0.5, 'dice_score': 0.8}
        
        update = ModelUpdate(
            node_id="HOSP_A",
            model_params=model_params,
            metrics=metrics,
            num_samples=100,
            timestamp="2024-01-01T00:00:00"
        )
        
        assert update.node_id == "HOSP_A"
        assert update.num_samples == 100
        assert update.weight == 100.0  # Weight equals num_samples
        assert update.metrics == metrics
    
    def test_weight_calculation(self):
        """Test weight calculation based on samples"""
        update = ModelUpdate(
            node_id="HOSP_A",
            model_params={},
            metrics={},
            num_samples=50,
            timestamp="2024-01-01T00:00:00"
        )
        
        assert update.weight == 50.0
        
        # Test minimum weight
        update_small = ModelUpdate(
            node_id="HOSP_B",
            model_params={},
            metrics={},
            num_samples=0,
            timestamp="2024-01-01T00:00:00"
        )
        
        assert update_small.weight == 1.0  # Minimum weight


class TestModelAggregationEngineSimple:
    """Test ModelAggregationEngine basic functionality"""
    
    @pytest.fixture
    def aggregation_engine(self):
        """Create aggregation engine for testing"""
        return ModelAggregationEngine()
    
    @pytest.fixture
    def mock_model_params(self):
        """Create mock model parameters"""
        return {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
            'layer2.weight': torch.randn(3, 10),
            'layer2.bias': torch.randn(3)
        }
    
    @pytest.mark.asyncio
    async def test_add_model_update(self, aggregation_engine, mock_model_params):
        """Test adding model update"""
        metrics = {'loss': 0.5, 'dice_score': 0.8}
        num_samples = 100
        
        success = await aggregation_engine.add_model_update(
            "HOSP_A", mock_model_params, metrics, num_samples
        )
        
        assert success is True
        assert "HOSP_A" in aggregation_engine.pending_updates
        
        update = aggregation_engine.pending_updates["HOSP_A"]
        assert update.node_id == "HOSP_A"
        assert update.num_samples == num_samples
        assert update.metrics == metrics
    
    @pytest.mark.asyncio
    async def test_weighted_averaging_simple(self, aggregation_engine, mock_model_params):
        """Test weighted averaging with simple case"""
        # Add two model updates with different sample counts
        params1 = {k: v.clone() for k, v in mock_model_params.items()}
        params2 = {k: v.clone() + 1.0 for k, v in mock_model_params.items()}
        
        await aggregation_engine.add_model_update(
            "HOSP_A", params1, {'loss': 0.4}, 100
        )
        await aggregation_engine.add_model_update(
            "HOSP_B", params2, {'loss': 0.6}, 200
        )
        
        # Perform weighted aggregation
        aggregated_params, aggregated_metrics = await aggregation_engine.aggregate_models(
            strategy='weighted_averaging'
        )
        
        # Check that parameters are properly weighted
        # HOSP_B should have 2x weight of HOSP_A (200 vs 100 samples)
        expected_weight_a = 100.0 / 300.0  # 1/3
        expected_weight_b = 200.0 / 300.0  # 2/3
        
        # Check one parameter to verify weighting
        expected_layer1_weight = (expected_weight_a * params1['layer1.weight'] + 
                                expected_weight_b * params2['layer1.weight'])
        
        torch.testing.assert_close(
            aggregated_params['layer1.weight'], 
            expected_layer1_weight,
            rtol=1e-5, atol=1e-5
        )
        
        # Check aggregated metrics
        assert 'loss' in aggregated_metrics
        assert 'num_clients' in aggregated_metrics
        assert 'total_samples' in aggregated_metrics
        assert aggregated_metrics['num_clients'] == 2
        assert aggregated_metrics['total_samples'] == 300
    
    @pytest.mark.asyncio
    async def test_federated_averaging_simple(self, aggregation_engine, mock_model_params):
        """Test federated averaging (simple average)"""
        # Add three identical updates
        for i, node_id in enumerate(["HOSP_A", "HOSP_B", "CLINIC_C"]):
            params = {k: v.clone() + i for k, v in mock_model_params.items()}
            await aggregation_engine.add_model_update(
                node_id, params, {'loss': 0.5}, 100
            )
        
        # Perform federated averaging
        aggregated_params, aggregated_metrics = await aggregation_engine.aggregate_models(
            strategy='federated_averaging'
        )
        
        # Check that parameters are simple average
        # Expected: (params0 + params1 + params2) / 3
        expected_layer1_weight = (mock_model_params['layer1.weight'] + 
                                (mock_model_params['layer1.weight'] + 1) + 
                                (mock_model_params['layer1.weight'] + 2)) / 3
        
        torch.testing.assert_close(
            aggregated_params['layer1.weight'], 
            expected_layer1_weight,
            rtol=1e-5, atol=1e-5
        )
        
        assert aggregated_metrics['num_clients'] == 3
    
    def test_aggregation_stats(self, aggregation_engine):
        """Test aggregation statistics"""
        stats = aggregation_engine.get_aggregation_stats()
        
        expected_keys = [
            'pending_updates', 'total_aggregations', 'supported_strategies',
            'default_strategy', 'pending_nodes'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats['pending_updates'] == 0
        assert stats['total_aggregations'] == 0
        assert 'weighted_averaging' in stats['supported_strategies']
        assert stats['default_strategy'] == 'weighted_averaging'
    
    @pytest.mark.asyncio
    async def test_empty_updates_error(self, aggregation_engine):
        """Test error when no updates to aggregate"""
        with pytest.raises(ValueError, match="No pending model updates"):
            await aggregation_engine.aggregate_models()
    
    @pytest.mark.asyncio
    async def test_clear_pending_updates(self, aggregation_engine, mock_model_params):
        """Test clearing pending updates"""
        # Add an update
        await aggregation_engine.add_model_update(
            "HOSP_A", mock_model_params, {'loss': 0.5}, 100
        )
        
        assert len(aggregation_engine.pending_updates) == 1
        
        # Clear updates
        aggregation_engine.clear_pending_updates()
        
        assert len(aggregation_engine.pending_updates) == 0


class TestFederatedLearningConfig:
    """Test federated learning configuration classes"""
    
    def test_federated_server_config_defaults(self):
        """Test server configuration defaults"""
        config = FederatedServerConfig()
        
        assert config.server_host == "0.0.0.0"
        assert config.server_port == 8080
        assert config.min_participants == 2
        assert config.aggregation_strategy == "weighted_averaging"
        assert config.enable_encryption is True
        assert config.local_epochs == 5
    
    def test_federated_client_config_defaults(self):
        """Test client configuration defaults"""
        config = FederatedClientConfig()
        
        assert config.data_directory == "data/local"
        assert config.enable_local_validation is True
        assert config.validation_split == 0.2
        assert config.model_cache_size == 5
        assert config.enable_differential_privacy is False
    
    def test_healthcare_institutions_config(self):
        """Test predefined healthcare institutions"""
        config = FederatedServerConfig()
        
        institutions = config.healthcare_institutions
        
        assert "HOSP_A" in institutions
        assert "HOSP_B" in institutions
        assert "CLINIC_C" in institutions
        
        assert institutions["HOSP_A"]["name"] == "Hospital A"
        assert institutions["HOSP_B"]["name"] == "Hospital B"
        assert institutions["CLINIC_C"]["name"] == "Clinic C"
        
        # Check endpoints
        assert "hosp-a.neurodx.local" in institutions["HOSP_A"]["endpoint"]
        assert "hosp-b.neurodx.local" in institutions["HOSP_B"]["endpoint"]
        assert "clinic-c.neurodx.local" in institutions["CLINIC_C"]["endpoint"]
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid aggregation strategy
        with pytest.raises(ValueError, match="Aggregation strategy must be one of"):
            FederatedServerConfig(aggregation_strategy="invalid_strategy")
        
        # Test invalid node ID
        with pytest.raises(ValueError, match="Node ID must start with"):
            FederatedClientConfig(node_id="INVALID_ID")
        
        # Test invalid minimum participants
        with pytest.raises(ValueError, match="Minimum participants must be at least 2"):
            FederatedServerConfig(min_participants=1)
        
        # Test invalid validation split
        with pytest.raises(ValueError, match="Validation split must be between"):
            FederatedClientConfig(validation_split=1.5)


class TestFederatedLearningIntegration:
    """Integration tests for federated learning components"""
    
    @pytest.mark.asyncio
    async def test_aggregation_workflow(self):
        """Test complete aggregation workflow"""
        engine = ModelAggregationEngine()
        
        # Create mock model parameters for 3 institutions
        base_params = {
            'conv.weight': torch.randn(16, 3, 3, 3),
            'conv.bias': torch.randn(16),
            'fc.weight': torch.randn(10, 16),
            'fc.bias': torch.randn(10)
        }
        
        institutions = [
            ("HOSP_A", 150, {'loss': 0.4, 'dice': 0.85}),
            ("HOSP_B", 200, {'loss': 0.3, 'dice': 0.88}),
            ("CLINIC_C", 100, {'loss': 0.5, 'dice': 0.82})
        ]
        
        # Add model updates from all institutions
        for node_id, samples, metrics in institutions:
            # Create slightly different parameters for each institution
            params = {k: v.clone() + torch.randn_like(v) * 0.1 
                     for k, v in base_params.items()}
            
            success = await engine.add_model_update(node_id, params, metrics, samples)
            assert success is True
        
        # Verify all updates are pending
        assert len(engine.pending_updates) == 3
        
        # Perform weighted aggregation
        aggregated_params, aggregated_metrics = await engine.aggregate_models()
        
        # Verify aggregation results
        assert len(aggregated_params) == len(base_params)
        for param_name in base_params:
            assert param_name in aggregated_params
            assert aggregated_params[param_name].shape == base_params[param_name].shape
        
        # Verify aggregated metrics
        assert aggregated_metrics['num_clients'] == 3
        assert aggregated_metrics['total_samples'] == 450
        assert 'loss' in aggregated_metrics
        assert 'dice' in aggregated_metrics
        
        # Verify pending updates are cleared
        assert len(engine.pending_updates) == 0
        
        # Verify aggregation history
        history = engine.get_aggregation_history()
        assert len(history) == 1
        assert history[0]['strategy'] == 'weighted_averaging'
        assert history[0]['num_clients'] == 3
    
    def test_config_integration(self):
        """Test configuration integration across components"""
        server_config = FederatedServerConfig(
            min_participants=3,
            aggregation_strategy="median_aggregation",
            local_epochs=10
        )
        
        client_config = FederatedClientConfig(
            node_id="HOSP_A",
            institution_name="Hospital A Test",
            validation_split=0.15
        )
        
        # Verify server config
        assert server_config.min_participants == 3
        assert server_config.aggregation_strategy == "median_aggregation"
        assert server_config.local_epochs == 10
        
        # Verify client config
        assert client_config.node_id == "HOSP_A"
        assert client_config.institution_name == "Hospital A Test"
        assert client_config.validation_split == 0.15
        
        # Verify inheritance from base config
        assert client_config.aggregation_strategy == "weighted_averaging"  # default
        assert client_config.local_epochs == 5  # default