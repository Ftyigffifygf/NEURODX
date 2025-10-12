"""
Model Aggregation Engine for MONAI Federated Learning

Implements secure model parameter aggregation using MONAI's federated learning capabilities.
Supports weighted averaging and advanced aggregation strategies for medical AI models.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import torch
import numpy as np
from typing import Union, Dict
from collections import defaultdict

# Define FLModelParamType for compatibility
FLModelParamType = Union[Dict[str, torch.Tensor], torch.nn.Module]

logger = logging.getLogger(__name__)


@dataclass
class ModelUpdate:
    """Represents a model update from a federated client"""
    node_id: str
    model_params: FLModelParamType
    metrics: Dict[str, float]
    num_samples: int
    timestamp: str
    weight: float = 1.0
    
    def __post_init__(self):
        """Calculate aggregation weight based on number of samples"""
        # Weight by number of samples (more samples = higher weight)
        self.weight = max(1.0, float(self.num_samples))


class ModelAggregationEngine:
    """
    MONAI Model Parameter Aggregation Engine
    
    Implements federated averaging and advanced aggregation strategies
    for combining model parameters from multiple healthcare institutions.
    """
    
    def __init__(self):
        self.pending_updates: Dict[str, ModelUpdate] = {}
        self.aggregation_history: List[Dict[str, Any]] = []
        self.supported_strategies = ['federated_averaging', 'weighted_averaging', 'median_aggregation']
        self.default_strategy = 'weighted_averaging'
    
    async def add_model_update(self, node_id: str, model_params: FLModelParamType,
                             metrics: Dict[str, float], num_samples: int) -> bool:
        """
        Add a model update from a federated client
        
        Args:
            node_id: ID of the client node
            model_params: MONAI model parameters
            metrics: Training metrics from client
            num_samples: Number of training samples used
            
        Returns:
            bool: True if update added successfully
        """
        try:
            from datetime import datetime
            
            update = ModelUpdate(
                node_id=node_id,
                model_params=model_params,
                metrics=metrics,
                num_samples=num_samples,
                timestamp=datetime.now().isoformat()
            )
            
            self.pending_updates[node_id] = update
            logger.info(f"Added model update from {node_id} with {num_samples} samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add model update from {node_id}: {e}")
            return False
    
    async def aggregate_models(self, strategy: str = None) -> Tuple[FLModelParamType, Dict[str, float]]:
        """
        Aggregate model parameters from all pending updates
        
        Args:
            strategy: Aggregation strategy to use
            
        Returns:
            Tuple of (aggregated_parameters, aggregated_metrics)
        """
        try:
            if not self.pending_updates:
                raise ValueError("No pending model updates to aggregate")
            
            strategy = strategy or self.default_strategy
            
            if strategy == 'federated_averaging':
                return await self._federated_averaging()
            elif strategy == 'weighted_averaging':
                return await self._weighted_averaging()
            elif strategy == 'median_aggregation':
                return await self._median_aggregation()
            else:
                raise ValueError(f"Unsupported aggregation strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Failed to aggregate models: {e}")
            raise
    
    async def _federated_averaging(self) -> Tuple[FLModelParamType, Dict[str, float]]:
        """
        Standard federated averaging (FedAvg) algorithm
        
        Simple average of all model parameters without weighting.
        """
        try:
            updates = list(self.pending_updates.values())
            num_clients = len(updates)
            
            # Initialize aggregated parameters with first client's parameters
            aggregated_params = {}
            first_params = updates[0].model_params
            
            for param_name, param_tensor in first_params.items():
                aggregated_params[param_name] = torch.zeros_like(param_tensor)
            
            # Sum all parameters
            for update in updates:
                for param_name, param_tensor in update.model_params.items():
                    aggregated_params[param_name] += param_tensor
            
            # Average the parameters
            for param_name in aggregated_params:
                aggregated_params[param_name] /= num_clients
            
            # Aggregate metrics
            aggregated_metrics = await self._aggregate_metrics(updates)
            
            # Clear pending updates
            self._record_aggregation('federated_averaging', updates)
            self.pending_updates.clear()
            
            logger.info(f"Completed federated averaging with {num_clients} clients")
            return aggregated_params, aggregated_metrics
            
        except Exception as e:
            logger.error(f"Failed federated averaging: {e}")
            raise
    
    async def _weighted_averaging(self) -> Tuple[FLModelParamType, Dict[str, float]]:
        """
        Weighted federated averaging based on number of training samples
        
        Clients with more training data have higher influence on the global model.
        """
        try:
            updates = list(self.pending_updates.values())
            
            # Calculate total weight
            total_weight = sum(update.weight for update in updates)
            
            # Initialize aggregated parameters
            aggregated_params = {}
            first_params = updates[0].model_params
            
            for param_name, param_tensor in first_params.items():
                aggregated_params[param_name] = torch.zeros_like(param_tensor)
            
            # Weighted sum of parameters
            for update in updates:
                weight_ratio = update.weight / total_weight
                for param_name, param_tensor in update.model_params.items():
                    aggregated_params[param_name] += weight_ratio * param_tensor
            
            # Aggregate metrics with weighting
            aggregated_metrics = await self._aggregate_metrics(updates, weighted=True)
            
            # Clear pending updates
            self._record_aggregation('weighted_averaging', updates)
            self.pending_updates.clear()
            
            logger.info(f"Completed weighted averaging with total weight {total_weight}")
            return aggregated_params, aggregated_metrics
            
        except Exception as e:
            logger.error(f"Failed weighted averaging: {e}")
            raise
    
    async def _median_aggregation(self) -> Tuple[FLModelParamType, Dict[str, float]]:
        """
        Median-based aggregation for robustness against outliers
        
        Uses element-wise median of model parameters for Byzantine fault tolerance.
        """
        try:
            updates = list(self.pending_updates.values())
            
            # Initialize aggregated parameters
            aggregated_params = {}
            first_params = updates[0].model_params
            
            for param_name, param_tensor in first_params.items():
                # Stack all parameter tensors for this layer
                param_stack = torch.stack([
                    update.model_params[param_name] for update in updates
                ])
                
                # Calculate element-wise median
                aggregated_params[param_name] = torch.median(param_stack, dim=0)[0]
            
            # Aggregate metrics
            aggregated_metrics = await self._aggregate_metrics(updates)
            
            # Clear pending updates
            self._record_aggregation('median_aggregation', updates)
            self.pending_updates.clear()
            
            logger.info(f"Completed median aggregation with {len(updates)} clients")
            return aggregated_params, aggregated_metrics
            
        except Exception as e:
            logger.error(f"Failed median aggregation: {e}")
            raise
    
    async def _aggregate_metrics(self, updates: List[ModelUpdate], 
                               weighted: bool = False) -> Dict[str, float]:
        """
        Aggregate training metrics from all clients
        
        Args:
            updates: List of model updates
            weighted: Whether to use weighted averaging
            
        Returns:
            Dict of aggregated metrics
        """
        try:
            if not updates:
                return {}
            
            # Collect all metric names
            all_metrics = set()
            for update in updates:
                all_metrics.update(update.metrics.keys())
            
            aggregated_metrics = {}
            
            for metric_name in all_metrics:
                metric_values = []
                weights = []
                
                for update in updates:
                    if metric_name in update.metrics:
                        metric_values.append(update.metrics[metric_name])
                        weights.append(update.weight if weighted else 1.0)
                
                if metric_values:
                    if weighted:
                        # Weighted average
                        total_weight = sum(weights)
                        weighted_sum = sum(v * w for v, w in zip(metric_values, weights))
                        aggregated_metrics[metric_name] = weighted_sum / total_weight
                    else:
                        # Simple average
                        aggregated_metrics[metric_name] = np.mean(metric_values)
            
            # Add aggregation statistics
            aggregated_metrics['num_clients'] = len(updates)
            aggregated_metrics['total_samples'] = sum(update.num_samples for update in updates)
            
            return aggregated_metrics
            
        except Exception as e:
            logger.error(f"Failed to aggregate metrics: {e}")
            return {}
    
    def _record_aggregation(self, strategy: str, updates: List[ModelUpdate]):
        """Record aggregation details for audit and analysis"""
        try:
            from datetime import datetime
            
            record = {
                'timestamp': datetime.now().isoformat(),
                'strategy': strategy,
                'num_clients': len(updates),
                'total_samples': sum(update.num_samples for update in updates),
                'client_nodes': [update.node_id for update in updates],
                'client_metrics': {
                    update.node_id: update.metrics for update in updates
                }
            }
            
            self.aggregation_history.append(record)
            logger.debug(f"Recorded aggregation: {strategy} with {len(updates)} clients")
            
        except Exception as e:
            logger.error(f"Failed to record aggregation: {e}")
    
    async def get_received_updates(self) -> Dict[str, ModelUpdate]:
        """Get currently pending model updates"""
        return self.pending_updates.copy()
    
    def get_aggregation_history(self) -> List[Dict[str, Any]]:
        """Get history of model aggregations"""
        return self.aggregation_history.copy()
    
    def clear_pending_updates(self):
        """Clear all pending model updates"""
        self.pending_updates.clear()
        logger.info("Cleared all pending model updates")
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get statistics about aggregation engine"""
        return {
            'pending_updates': len(self.pending_updates),
            'total_aggregations': len(self.aggregation_history),
            'supported_strategies': self.supported_strategies,
            'default_strategy': self.default_strategy,
            'pending_nodes': list(self.pending_updates.keys())
        }