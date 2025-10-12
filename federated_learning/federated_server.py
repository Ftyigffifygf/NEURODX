"""
MONAI Federated Learning Server

Implements the central server for coordinating federated learning across multiple healthcare institutions.
Uses MONAI's federated learning capabilities for secure model parameter aggregation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import torch
import numpy as np
from monai.fl.client import ClientAlgo
from typing import Union

# Define FLModelParamType for compatibility
FLModelParamType = Union[Dict[str, torch.Tensor], torch.nn.Module]

from src.config.settings import get_settings
from .aggregation_engine import ModelAggregationEngine
from .secure_communication import SecureCommunicationManager
from .fault_tolerance import FaultToleranceManager, NodeStatus

logger = logging.getLogger(__name__)


@dataclass
class FederatedNode:
    """Represents a federated learning client node (hospital/clinic)"""
    node_id: str
    institution_name: str
    endpoint: str
    public_key: str
    last_seen: datetime
    is_active: bool = True
    total_samples: int = 0
    model_version: int = 0
    
    def __post_init__(self):
        """Validate federated node configuration"""
        if not self.node_id or not self.node_id.startswith(('HOSP_', 'CLINIC_')):
            raise ValueError("Node ID must start with 'HOSP_' or 'CLINIC_'")
        if not self.institution_name:
            raise ValueError("Institution name is required")
        if not self.endpoint or not self.endpoint.startswith(('http://', 'https://')):
            raise ValueError("Valid endpoint URL is required")


@dataclass
class FederatedRound:
    """Represents a federated learning training round"""
    round_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    participating_nodes: List[str] = field(default_factory=list)
    aggregated_metrics: Dict[str, float] = field(default_factory=dict)
    global_model_version: int = 0
    status: str = "pending"  # pending, active, completed, failed


class FederatedLearningServer:
    """
    MONAI Federated Learning Server for coordinating multi-institutional training
    
    Manages federated learning rounds, node registration, and secure model aggregation
    across Hospital A, Hospital B, and Clinic C nodes as specified in requirements.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.nodes: Dict[str, FederatedNode] = {}
        self.current_round: Optional[FederatedRound] = None
        self.round_history: List[FederatedRound] = []
        self.global_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.aggregation_engine = ModelAggregationEngine()
        self.comm_manager = SecureCommunicationManager()
        self.fault_tolerance = FaultToleranceManager()
        self.server_algorithm: Optional[ClientAlgo] = None
        self._initialize_server()
    
    def _initialize_server(self):
        """Initialize MONAI federated learning server"""
        try:
            # Initialize MONAI FL server algorithm
            # Note: Using ClientAlgo as base for server coordination
            self.server_algorithm = ClientAlgo()
            logger.info("MONAI federated learning server initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize federated learning server: {e}")
            raise
    
    async def register_node(self, node_config: Dict[str, Any]) -> bool:
        """
        Register a new federated learning client node
        
        Args:
            node_config: Node configuration including ID, institution, endpoint, public key
            
        Returns:
            bool: True if registration successful
        """
        try:
            node = FederatedNode(
                node_id=node_config['node_id'],
                institution_name=node_config['institution_name'],
                endpoint=node_config['endpoint'],
                public_key=node_config['public_key'],
                last_seen=datetime.now()
            )
            
            # Validate node credentials and establish secure connection
            if not await self.comm_manager.validate_node_credentials(node):
                logger.error(f"Failed to validate credentials for node {node.node_id}")
                return False
            
            self.nodes[node.node_id] = node
            
            # Register with fault tolerance system
            await self.fault_tolerance.register_node(node.node_id)
            
            logger.info(f"Successfully registered federated node: {node.node_id} ({node.institution_name})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register node: {e}")
            return False
    
    async def start_federated_round(self, min_participants: int = 2) -> bool:
        """
        Start a new federated learning round
        
        Args:
            min_participants: Minimum number of active nodes required
            
        Returns:
            bool: True if round started successfully
        """
        try:
            # Check if we have enough active nodes using fault tolerance system
            active_node_ids = self.fault_tolerance.get_active_nodes()
            active_nodes = [self.nodes[node_id] for node_id in active_node_ids if node_id in self.nodes]
            
            if len(active_nodes) < min_participants:
                logger.warning(f"Insufficient active nodes ({len(active_nodes)}) for federated round")
                return False
            
            # Create new federated round
            round_id = len(self.round_history) + 1
            self.current_round = FederatedRound(
                round_id=round_id,
                start_time=datetime.now(),
                participating_nodes=[node.node_id for node in active_nodes],
                status="active"
            )
            
            # Broadcast round start to all participating nodes
            await self._broadcast_round_start()
            
            logger.info(f"Started federated learning round {round_id} with {len(active_nodes)} participants")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start federated round: {e}")
            return False
    
    async def _broadcast_round_start(self):
        """Broadcast round start message to all participating nodes"""
        if not self.current_round:
            return
        
        message = {
            'type': 'round_start',
            'round_id': self.current_round.round_id,
            'global_model_version': self.current_round.global_model_version,
            'participants': self.current_round.participating_nodes
        }
        
        # Send to all participating nodes
        for node_id in self.current_round.participating_nodes:
            node = self.nodes[node_id]
            try:
                await self.comm_manager.send_secure_message(node, message)
                logger.debug(f"Sent round start message to {node_id}")
            except Exception as e:
                logger.error(f"Failed to send round start to {node_id}: {e}")
    
    async def receive_model_update(self, node_id: str, model_params: FLModelParamType, 
                                 metrics: Dict[str, float], num_samples: int) -> bool:
        """
        Receive model parameters from a federated client
        
        Args:
            node_id: ID of the sending node
            model_params: MONAI model parameters from client training
            metrics: Training metrics from client
            num_samples: Number of training samples used
            
        Returns:
            bool: True if update received successfully
        """
        try:
            if not self.current_round or node_id not in self.current_round.participating_nodes:
                logger.warning(f"Received update from non-participating node: {node_id}")
                return False
            
            # Store model update for aggregation
            await self.aggregation_engine.add_model_update(
                node_id, model_params, metrics, num_samples
            )
            
            # Update node information
            if node_id in self.nodes:
                self.nodes[node_id].last_seen = datetime.now()
                self.nodes[node_id].total_samples = num_samples
            
            logger.info(f"Received model update from {node_id} with {num_samples} samples")
            
            # Check if we have all updates for this round
            if await self._all_updates_received():
                await self._complete_federated_round()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to receive model update from {node_id}: {e}")
            return False
    
    async def _all_updates_received(self) -> bool:
        """Check if all participating nodes have sent their updates"""
        if not self.current_round:
            return False
        
        received_updates = await self.aggregation_engine.get_received_updates()
        expected_nodes = set(self.current_round.participating_nodes)
        received_nodes = set(received_updates.keys())
        
        return expected_nodes.issubset(received_nodes)
    
    async def _complete_federated_round(self):
        """Complete the current federated round by aggregating model parameters"""
        try:
            if not self.current_round:
                return
            
            # Aggregate model parameters using MONAI federated learning
            aggregated_params, aggregated_metrics = await self.aggregation_engine.aggregate_models()
            
            # Update global model state
            self.global_model_state = aggregated_params
            
            # Complete the round
            self.current_round.end_time = datetime.now()
            self.current_round.aggregated_metrics = aggregated_metrics
            self.current_round.status = "completed"
            self.current_round.global_model_version += 1
            
            # Create model checkpoint for fault tolerance
            participating_nodes = set(self.current_round.participating_nodes)
            await self.fault_tolerance.create_model_checkpoint(
                self.current_round.round_id, aggregated_params, participating_nodes
            )
            
            # Broadcast updated global model to all nodes
            await self._broadcast_global_model()
            
            # Archive completed round
            self.round_history.append(self.current_round)
            self.current_round = None
            
            logger.info(f"Completed federated round with aggregated metrics: {aggregated_metrics}")
            
        except Exception as e:
            logger.error(f"Failed to complete federated round: {e}")
            if self.current_round:
                self.current_round.status = "failed"
    
    async def _broadcast_global_model(self):
        """Broadcast updated global model to all participating nodes"""
        if not self.current_round or not self.global_model_state:
            return
        
        message = {
            'type': 'global_model_update',
            'round_id': self.current_round.round_id,
            'model_version': self.current_round.global_model_version,
            'model_params': self.global_model_state,
            'aggregated_metrics': self.current_round.aggregated_metrics
        }
        
        # Send to all participating nodes
        for node_id in self.current_round.participating_nodes:
            node = self.nodes[node_id]
            try:
                await self.comm_manager.send_secure_message(node, message)
                logger.debug(f"Sent global model update to {node_id}")
            except Exception as e:
                logger.error(f"Failed to send global model to {node_id}: {e}")
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get current server status and statistics"""
        active_nodes = [node for node in self.nodes.values() if node.is_active]
        
        return {
            'total_nodes': len(self.nodes),
            'active_nodes': len(active_nodes),
            'current_round': self.current_round.round_id if self.current_round else None,
            'completed_rounds': len(self.round_history),
            'global_model_version': self.current_round.global_model_version if self.current_round else 0,
            'node_list': [
                {
                    'node_id': node.node_id,
                    'institution': node.institution_name,
                    'active': node.is_active,
                    'last_seen': node.last_seen.isoformat(),
                    'total_samples': node.total_samples
                }
                for node in self.nodes.values()
            ]
        }
    
    async def handle_node_failure(self, node_id: str):
        """Handle node failure during federated learning"""
        try:
            if node_id in self.nodes:
                self.nodes[node_id].is_active = False
                logger.warning(f"Marked node {node_id} as inactive due to failure")
                
                # Update fault tolerance system
                await self.fault_tolerance.update_node_health(
                    node_id, response_time_ms=float('inf'), success=False, 
                    error_message="Node failure detected"
                )
                
                # If node was participating in current round, handle gracefully
                if (self.current_round and 
                    node_id in self.current_round.participating_nodes):
                    
                    # Remove from current round participants
                    self.current_round.participating_nodes.remove(node_id)
                    
                    # Check if we still have minimum participants
                    active_participants = [
                        pid for pid in self.current_round.participating_nodes
                        if pid in self.fault_tolerance.get_active_nodes()
                    ]
                    
                    if len(active_participants) < 2:
                        logger.error("Insufficient participants after node failure, attempting recovery")
                        
                        # Try to recover from checkpoint
                        recovered_model = await self.fault_tolerance.recover_from_checkpoint()
                        if recovered_model:
                            logger.info("Recovered global model from checkpoint")
                            self.global_model_state = recovered_model
                        
                        # Abort current round
                        self.current_round.status = "failed"
                        self.current_round = None
                    else:
                        logger.info(f"Continuing round with {len(active_participants)} participants")
                        self.current_round.participating_nodes = active_participants
                        
        except Exception as e:
            logger.error(f"Failed to handle node failure for {node_id}: {e}")
    
    async def handle_node_recovery(self, node_id: str):
        """Handle node recovery after failure"""
        try:
            if node_id in self.nodes:
                self.nodes[node_id].is_active = True
                logger.info(f"Node {node_id} has recovered and is now active")
                
                # Update fault tolerance system
                await self.fault_tolerance.update_node_health(
                    node_id, response_time_ms=100.0, success=True
                )
                
                # If there's an ongoing round and we need more participants, add the recovered node
                if (self.current_round and 
                    len(self.current_round.participating_nodes) < len(self.nodes) and
                    node_id not in self.current_round.participating_nodes):
                    
                    # Send current round information to recovered node
                    await self._send_round_sync_to_node(node_id)
                    
        except Exception as e:
            logger.error(f"Failed to handle node recovery for {node_id}: {e}")
    
    async def _send_round_sync_to_node(self, node_id: str):
        """Send current round synchronization to a recovered node"""
        try:
            if not self.current_round:
                return
            
            node = self.nodes[node_id]
            sync_message = {
                'type': 'round_sync',
                'round_id': self.current_round.round_id,
                'global_model_version': self.current_round.global_model_version,
                'global_model_state': self.global_model_state,
                'round_status': self.current_round.status
            }
            
            await self.comm_manager.send_secure_message(node, sync_message)
            logger.info(f"Sent round synchronization to recovered node {node_id}")
            
        except Exception as e:
            logger.error(f"Failed to send round sync to node {node_id}: {e}")