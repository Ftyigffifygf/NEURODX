"""
Unit tests for MONAI Federated Learning Infrastructure

Tests federated learning server, client, aggregation engine, and secure communication.
"""

import pytest
import asyncio
import torch
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any

from src.services.federated_learning.federated_server import (
    FederatedLearningServer, FederatedNode, FederatedRound
)
from src.services.federated_learning.federated_client import (
    FederatedLearningClient, ClientConfig
)
from src.services.federated_learning.aggregation_engine import (
    ModelAggregationEngine, ModelUpdate
)
from src.services.federated_learning.secure_communication import (
    SecureCommunicationManager
)
from src.config.federated_learning_config import (
    FederatedServerConfig, FederatedClientConfig
)


class TestFederatedNode:
    """Test FederatedNode data class"""
    
    def test_valid_node_creation(self):
        """Test creating a valid federated node"""
        node = FederatedNode(
            node_id="HOSP_A",
            institution_name="Hospital A",
            endpoint="https://hosp-a.example.com",
            public_key="test-public-key",
            last_seen=datetime.now()
        )
        
        assert node.node_id == "HOSP_A"
        assert node.institution_name == "Hospital A"
        assert node.is_active is True
        assert node.total_samples == 0
    
    def test_invalid_node_id(self):
        """Test node creation with invalid node ID"""
        with pytest.raises(ValueError, match="Node ID must start with"):
            FederatedNode(
                node_id="INVALID_ID",
                institution_name="Test Hospital",
                endpoint="https://test.example.com",
                public_key="test-key",
                last_seen=datetime.now()
            )
    
    def test_missing_institution_name(self):
        """Test node creation with missing institution name"""
        with pytest.raises(ValueError, match="Institution name is required"):
            FederatedNode(
                node_id="HOSP_A",
                institution_name="",
                endpoint="https://test.example.com",
                public_key="test-key",
                last_seen=datetime.now()
            )
    
    def test_invalid_endpoint(self):
        """Test node creation with invalid endpoint"""
        with pytest.raises(ValueError, match="Valid endpoint URL is required"):
            FederatedNode(
                node_id="HOSP_A",
                institution_name="Hospital A",
                endpoint="invalid-url",
                public_key="test-key",
                last_seen=datetime.now()
            )


class TestModelAggregationEngine:
    """Test ModelAggregationEngine functionality"""
    
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
        assert update.weight == 100.0  # Weight equals num_samples
    
    @pytest.mark.asyncio
    async def test_federated_averaging(self, aggregation_engine, mock_model_params):
        """Test federated averaging aggregation"""
        # Add multiple model updates
        for i, node_id in enumerate(["HOSP_A", "HOSP_B", "CLINIC_C"]):
            # Create slightly different parameters for each node
            params = {k: v + i * 0.1 for k, v in mock_model_params.items()}
            metrics = {'loss': 0.5 + i * 0.1, 'dice_score': 0.8 - i * 0.05}
            
            await aggregation_engine.add_model_update(
                node_id, params, metrics, 100 + i * 10
            )
        
        # Perform aggregation
        aggregated_params, aggregated_metrics = await aggregation_engine.aggregate_models(
            strategy='federated_averaging'
        )
        
        # Check that parameters are averaged
        assert 'layer1.weight' in aggregated_params
        assert aggregated_params['layer1.weight'].shape == mock_model_params['layer1.weight'].shape
        
        # Check aggregated metrics
        assert 'loss' in aggregated_metrics
        assert 'dice_score' in aggregated_metrics
        assert 'num_clients' in aggregated_metrics
        assert aggregated_metrics['num_clients'] == 3
        
        # Check that pending updates are cleared
        assert len(aggregation_engine.pending_updates) == 0
    
    @pytest.mark.asyncio
    async def test_weighted_averaging(self, aggregation_engine, mock_model_params):
        """Test weighted averaging aggregation"""
        # Add model updates with different sample counts
        sample_counts = [50, 100, 200]
        for i, (node_id, samples) in enumerate(zip(["HOSP_A", "HOSP_B", "CLINIC_C"], sample_counts)):
            params = {k: v + i * 0.1 for k, v in mock_model_params.items()}
            metrics = {'loss': 0.5, 'dice_score': 0.8}
            
            await aggregation_engine.add_model_update(
                node_id, params, metrics, samples
            )
        
        # Perform weighted aggregation
        aggregated_params, aggregated_metrics = await aggregation_engine.aggregate_models(
            strategy='weighted_averaging'
        )
        
        # Check results
        assert 'layer1.weight' in aggregated_params
        assert 'total_samples' in aggregated_metrics
        assert aggregated_metrics['total_samples'] == sum(sample_counts)
    
    @pytest.mark.asyncio
    async def test_median_aggregation(self, aggregation_engine, mock_model_params):
        """Test median aggregation"""
        # Add model updates
        for i, node_id in enumerate(["HOSP_A", "HOSP_B", "CLINIC_C"]):
            params = {k: v + i * 0.1 for k, v in mock_model_params.items()}
            metrics = {'loss': 0.5, 'dice_score': 0.8}
            
            await aggregation_engine.add_model_update(
                node_id, params, metrics, 100
            )
        
        # Perform median aggregation
        aggregated_params, aggregated_metrics = await aggregation_engine.aggregate_models(
            strategy='median_aggregation'
        )
        
        # Check results
        assert 'layer1.weight' in aggregated_params
        assert aggregated_params['layer1.weight'].shape == mock_model_params['layer1.weight'].shape
    
    @pytest.mark.asyncio
    async def test_empty_updates_error(self, aggregation_engine):
        """Test error when no updates to aggregate"""
        with pytest.raises(ValueError, match="No pending model updates"):
            await aggregation_engine.aggregate_models()
    
    def test_aggregation_stats(self, aggregation_engine):
        """Test aggregation statistics"""
        stats = aggregation_engine.get_aggregation_stats()
        
        assert 'pending_updates' in stats
        assert 'total_aggregations' in stats
        assert 'supported_strategies' in stats
        assert 'default_strategy' in stats
        assert stats['pending_updates'] == 0
        assert stats['total_aggregations'] == 0


class TestSecureCommunicationManager:
    """Test SecureCommunicationManager functionality"""
    
    @pytest.fixture
    def comm_manager(self):
        """Create communication manager for testing"""
        with patch('src.services.federated_learning.secure_communication.Path.mkdir'):
            return SecureCommunicationManager()
    
    def test_initialization(self, comm_manager):
        """Test communication manager initialization"""
        assert comm_manager.private_key is not None
        assert comm_manager.public_key is not None
        assert comm_manager.ssl_context is not None
    
    @pytest.mark.asyncio
    async def test_encrypt_decrypt_message(self, comm_manager):
        """Test message encryption and decryption"""
        message = {
            'type': 'test_message',
            'data': {'key': 'value', 'number': 42}
        }
        
        # Encrypt message
        encrypted_payload = await comm_manager.encrypt_message(
            message, comm_manager.public_key
        )
        
        assert isinstance(encrypted_payload, bytes)
        
        # Decrypt message
        decrypted_message = await comm_manager.decrypt_message(encrypted_payload)
        
        assert decrypted_message == message
    
    def test_message_padding(self, comm_manager):
        """Test message padding and unpadding"""
        message = b"test message"
        
        # Pad message
        padded = comm_manager._pad_message(message)
        assert len(padded) % 16 == 0
        assert len(padded) > len(message)
        
        # Unpad message
        unpadded = comm_manager._unpad_message(padded)
        assert unpadded == message
    
    @pytest.mark.asyncio
    async def test_load_public_key_from_string(self, comm_manager):
        """Test loading public key from string"""
        # Get public key as string
        public_key_pem = comm_manager.public_key.public_bytes(
            encoding=comm_manager.public_key.__class__.__module__.split('.')[0] == 'cryptography' and 
            getattr(__import__('cryptography.hazmat.primitives.serialization'), 'Encoding').PEM or None,
            format=getattr(__import__('cryptography.hazmat.primitives.serialization'), 'PublicFormat').SubjectPublicKeyInfo
        ).decode('utf-8')
        
        # Load key from string
        loaded_key = await comm_manager.load_public_key_from_string(public_key_pem)
        
        assert loaded_key is not None


class TestFederatedLearningServer:
    """Test FederatedLearningServer functionality"""
    
    @pytest.fixture
    def mock_server(self):
        """Create mock federated learning server"""
        with patch('src.services.federated_learning.federated_server.get_settings'):
            with patch('monai.fl.client.ClientAlgo'):
                return FederatedLearningServer()
    
    @pytest.mark.asyncio
    async def test_register_node(self, mock_server):
        """Test node registration"""
        node_config = {
            'node_id': 'HOSP_A',
            'institution_name': 'Hospital A',
            'endpoint': 'https://hosp-a.example.com',
            'public_key': 'test-public-key'
        }
        
        # Mock credential validation
        mock_server.comm_manager.validate_node_credentials = AsyncMock(return_value=True)
        
        success = await mock_server.register_node(node_config)
        
        assert success is True
        assert 'HOSP_A' in mock_server.nodes
        assert mock_server.nodes['HOSP_A'].institution_name == 'Hospital A'
    
    @pytest.mark.asyncio
    async def test_start_federated_round(self, mock_server):
        """Test starting federated learning round"""
        # Register nodes first
        for node_id in ['HOSP_A', 'HOSP_B']:
            node_config = {
                'node_id': node_id,
                'institution_name': f'Hospital {node_id[-1]}',
                'endpoint': f'https://{node_id.lower()}.example.com',
                'public_key': 'test-key'
            }
            mock_server.comm_manager.validate_node_credentials = AsyncMock(return_value=True)
            await mock_server.register_node(node_config)
        
        # Mock broadcast method
        mock_server._broadcast_round_start = AsyncMock()
        
        success = await mock_server.start_federated_round(min_participants=2)
        
        assert success is True
        assert mock_server.current_round is not None
        assert mock_server.current_round.status == "active"
        assert len(mock_server.current_round.participating_nodes) == 2
    
    @pytest.mark.asyncio
    async def test_insufficient_participants(self, mock_server):
        """Test starting round with insufficient participants"""
        success = await mock_server.start_federated_round(min_participants=2)
        
        assert success is False
        assert mock_server.current_round is None
    
    def test_server_status(self, mock_server):
        """Test getting server status"""
        status = mock_server.get_server_status()
        
        assert 'total_nodes' in status
        assert 'active_nodes' in status
        assert 'current_round' in status
        assert 'completed_rounds' in status
        assert 'node_list' in status
        
        assert status['total_nodes'] == 0
        assert status['active_nodes'] == 0


class TestFederatedLearningClient:
    """Test FederatedLearningClient functionality"""
    
    @pytest.fixture
    def client_config(self):
        """Create client configuration for testing"""
        return ClientConfig(
            node_id="HOSP_A",
            institution_name="Hospital A",
            server_endpoint="https://fl-server.example.com",
            private_key_path="keys/private_key.pem",
            public_key_path="keys/public_key.pem"
        )
    
    @pytest.fixture
    def mock_client(self, client_config):
        """Create mock federated learning client"""
        with patch('src.services.federated_learning.federated_client.get_settings'):
            with patch('monai.fl.client.ClientAlgo'):
                with patch('src.services.ml_inference.swin_unetr_model.SwinUNETRModel'):
                    return FederatedLearningClient(client_config)
    
    def test_client_initialization(self, mock_client, client_config):
        """Test client initialization"""
        assert mock_client.config.node_id == client_config.node_id
        assert mock_client.config.institution_name == client_config.institution_name
        assert mock_client.is_registered is False
        assert mock_client.current_round_id is None
    
    @pytest.mark.asyncio
    async def test_register_with_server(self, mock_client):
        """Test client registration with server"""
        # Mock communication manager methods
        mock_client.comm_manager.load_public_key = AsyncMock(return_value="test-public-key")
        mock_client.comm_manager.send_registration_request = AsyncMock(
            return_value={'status': 'success'}
        )
        
        success = await mock_client.register_with_server()
        
        assert success is True
        assert mock_client.is_registered is True
    
    def test_client_status(self, mock_client):
        """Test getting client status"""
        status = mock_client.get_client_status()
        
        assert 'node_id' in status
        assert 'institution_name' in status
        assert 'is_registered' in status
        assert 'current_round' in status
        assert 'model_initialized' in status
        assert 'server_endpoint' in status
        
        assert status['node_id'] == "HOSP_A"
        assert status['is_registered'] is False


class TestClientConfig:
    """Test ClientConfig validation"""
    
    def test_valid_config(self):
        """Test creating valid client configuration"""
        config = ClientConfig(
            node_id="HOSP_A",
            institution_name="Hospital A",
            server_endpoint="https://server.example.com",
            private_key_path="keys/private.pem",
            public_key_path="keys/public.pem"
        )
        
        assert config.node_id == "HOSP_A"
        assert config.local_epochs == 5  # default value
        assert config.batch_size == 4  # default value
    
    def test_invalid_node_id(self):
        """Test invalid node ID validation"""
        with pytest.raises(ValueError, match="Node ID must start with"):
            ClientConfig(
                node_id="INVALID",
                institution_name="Hospital A",
                server_endpoint="https://server.example.com",
                private_key_path="keys/private.pem",
                public_key_path="keys/public.pem"
            )
    
    def test_missing_institution_name(self):
        """Test missing institution name validation"""
        with pytest.raises(ValueError, match="Institution name is required"):
            ClientConfig(
                node_id="HOSP_A",
                institution_name="",
                server_endpoint="https://server.example.com",
                private_key_path="keys/private.pem",
                public_key_path="keys/public.pem"
            )
    
    def test_missing_server_endpoint(self):
        """Test missing server endpoint validation"""
        with pytest.raises(ValueError, match="Server endpoint is required"):
            ClientConfig(
                node_id="HOSP_A",
                institution_name="Hospital A",
                server_endpoint="",
                private_key_path="keys/private.pem",
                public_key_path="keys/public.pem"
            )


@pytest.mark.asyncio
async def test_federated_learning_integration():
    """Integration test for federated learning workflow"""
    # This test would simulate a complete federated learning round
    # with server and multiple clients
    
    # Mock components for integration test
    with patch('src.services.federated_learning.federated_server.get_settings'):
        with patch('monai.fl.client.ClientAlgo'):
            server = FederatedLearningServer()
    
    # Register multiple nodes
    nodes = ['HOSP_A', 'HOSP_B', 'CLINIC_C']
    for node_id in nodes:
        node_config = {
            'node_id': node_id,
            'institution_name': f'Institution {node_id}',
            'endpoint': f'https://{node_id.lower()}.example.com',
            'public_key': 'test-public-key'
        }
        server.comm_manager.validate_node_credentials = AsyncMock(return_value=True)
        await server.register_node(node_config)
    
    # Start federated round
    server._broadcast_round_start = AsyncMock()
    success = await server.start_federated_round()
    
    assert success is True
    assert server.current_round is not None
    assert len(server.current_round.participating_nodes) == 3
    
    # Simulate model updates from clients
    mock_params = {
        'layer.weight': torch.randn(5, 3),
        'layer.bias': torch.randn(5)
    }
    
    server._broadcast_global_model = AsyncMock()
    server._all_updates_received = AsyncMock(return_value=True)
    
    for node_id in nodes:
        metrics = {'loss': 0.5, 'dice_score': 0.8}
        await server.receive_model_update(node_id, mock_params, metrics, 100)
    
    # Check that round was completed
    assert len(server.round_history) > 0
    assert server.current_round is None  # Round should be completed