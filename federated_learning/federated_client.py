"""
MONAI Federated Learning Client

Implements the client-side federated learning functionality for healthcare institutions.
Handles local model training and secure communication with the federated server.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.fl.client import ClientAlgo
from typing import Union

# Define FLModelParamType for compatibility
FLModelParamType = Union[Dict[str, torch.Tensor], torch.nn.Module]
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

from src.config.settings import get_settings
from src.services.ml_inference.swin_unetr_model import SwinUNETRModel
from .secure_communication import SecureCommunicationManager

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for federated learning client"""
    node_id: str
    institution_name: str
    server_endpoint: str
    private_key_path: str
    public_key_path: str
    local_epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 1e-4
    
    def __post_init__(self):
        """Validate client configuration"""
        if not self.node_id or not self.node_id.startswith(('HOSP_', 'CLINIC_')):
            raise ValueError("Node ID must start with 'HOSP_' or 'CLINIC_'")
        if not self.institution_name:
            raise ValueError("Institution name is required")
        if not self.server_endpoint:
            raise ValueError("Server endpoint is required")


class FederatedLearningClient:
    """
    MONAI Federated Learning Client for healthcare institutions
    
    Handles local model training, secure communication with server,
    and participation in federated learning rounds.
    """
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.settings = get_settings()
        self.model: Optional[SwinUNETRModel] = None
        self.local_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.comm_manager = SecureCommunicationManager()
        self.client_algorithm: Optional[ClientAlgo] = None
        self.is_registered = False
        self.current_round_id: Optional[int] = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize MONAI federated learning client"""
        try:
            # Initialize MONAI FL client algorithm
            self.client_algorithm = ClientAlgo()
            
            # Initialize local model
            self.model = SwinUNETRModel()
            
            logger.info(f"MONAI federated learning client initialized for {self.config.node_id}")
        except Exception as e:
            logger.error(f"Failed to initialize federated learning client: {e}")
            raise
    
    async def register_with_server(self) -> bool:
        """
        Register this client with the federated learning server
        
        Returns:
            bool: True if registration successful
        """
        try:
            # Load public key for secure communication
            public_key = await self.comm_manager.load_public_key(self.config.public_key_path)
            
            registration_data = {
                'node_id': self.config.node_id,
                'institution_name': self.config.institution_name,
                'endpoint': f"https://{self.config.node_id.lower()}.healthcare.local",
                'public_key': public_key
            }
            
            # Send registration request to server
            response = await self.comm_manager.send_registration_request(
                self.config.server_endpoint, registration_data
            )
            
            if response.get('status') == 'success':
                self.is_registered = True
                logger.info(f"Successfully registered with federated learning server")
                return True
            else:
                logger.error(f"Registration failed: {response.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to register with server: {e}")
            return False
    
    async def participate_in_round(self, round_info: Dict[str, Any]) -> bool:
        """
        Participate in a federated learning round
        
        Args:
            round_info: Information about the federated learning round
            
        Returns:
            bool: True if participation successful
        """
        try:
            self.current_round_id = round_info['round_id']
            logger.info(f"Participating in federated learning round {self.current_round_id}")
            
            # Get global model parameters if available
            if 'global_model_params' in round_info:
                await self._update_local_model(round_info['global_model_params'])
            
            # Perform local training
            local_params, metrics, num_samples = await self._train_local_model()
            
            # Send model update to server
            success = await self._send_model_update(local_params, metrics, num_samples)
            
            if success:
                logger.info(f"Successfully completed participation in round {self.current_round_id}")
            else:
                logger.error(f"Failed to send model update for round {self.current_round_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to participate in federated round: {e}")
            return False
    
    async def _update_local_model(self, global_params: FLModelParamType):
        """Update local model with global parameters from server"""
        try:
            if self.model and self.model.model:
                # Load global parameters into local model
                self.model.model.load_state_dict(global_params)
                logger.debug("Updated local model with global parameters")
        except Exception as e:
            logger.error(f"Failed to update local model: {e}")
    
    async def _train_local_model(self) -> Tuple[FLModelParamType, Dict[str, float], int]:
        """
        Perform local model training on institution's data
        
        Returns:
            Tuple of (model_parameters, training_metrics, num_samples)
        """
        try:
            if not self.model:
                raise ValueError("Model not initialized")
            
            # Get local training data (this would be institution-specific)
            train_loader = await self._get_local_training_data()
            
            # Set up training components
            optimizer = torch.optim.Adam(self.model.model.parameters(), lr=self.config.learning_rate)
            loss_function = DiceLoss(to_onehot_y=True, softmax=True)
            dice_metric = DiceMetric(include_background=False, reduction="mean")
            
            # Training loop
            self.model.model.train()
            total_loss = 0.0
            num_batches = 0
            num_samples = 0
            
            for epoch in range(self.config.local_epochs):
                epoch_loss = 0.0
                
                for batch_data in train_loader:
                    inputs = batch_data["image"]
                    labels = batch_data["label"]
                    num_samples += inputs.shape[0]
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model.model(inputs)
                    loss = loss_function(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                total_loss += epoch_loss
                logger.debug(f"Epoch {epoch + 1}/{self.config.local_epochs}, Loss: {epoch_loss:.4f}")
            
            # Calculate metrics
            avg_loss = total_loss / (self.config.local_epochs * len(train_loader))
            
            # Evaluate model
            dice_score = await self._evaluate_local_model(train_loader, dice_metric)
            
            metrics = {
                'loss': avg_loss,
                'dice_score': dice_score,
                'epochs': self.config.local_epochs
            }
            
            # Get model parameters
            model_params = self.model.model.state_dict()
            
            logger.info(f"Local training completed: Loss={avg_loss:.4f}, Dice={dice_score:.4f}, Samples={num_samples}")
            
            return model_params, metrics, num_samples
            
        except Exception as e:
            logger.error(f"Failed to train local model: {e}")
            raise
    
    async def _evaluate_local_model(self, data_loader: DataLoader, metric: DiceMetric) -> float:
        """Evaluate local model performance"""
        try:
            self.model.model.eval()
            metric.reset()
            
            with torch.no_grad():
                for batch_data in data_loader:
                    inputs = batch_data["image"]
                    labels = batch_data["label"]
                    
                    outputs = self.model.model(inputs)
                    metric(y_pred=outputs, y=labels)
            
            # Calculate mean Dice score
            dice_score = metric.aggregate().item()
            return dice_score
            
        except Exception as e:
            logger.error(f"Failed to evaluate local model: {e}")
            return 0.0
    
    async def _get_local_training_data(self) -> DataLoader:
        """
        Get local training data for this institution
        
        This is a placeholder - in practice, each institution would have
        their own data loading logic based on their local data storage.
        """
        # This would be implemented based on institution's data infrastructure
        # For now, return a mock data loader
        from torch.utils.data import TensorDataset
        
        # Mock data - in practice this would load real medical imaging data
        batch_size = self.config.batch_size
        num_samples = 100  # Mock number of samples
        
        # Create mock tensors (4 channels for multi-modal input, 96x96x96 volume)
        mock_images = torch.randn(num_samples, 4, 96, 96, 96)
        mock_labels = torch.randint(0, 3, (num_samples, 1, 96, 96, 96))
        
        dataset = TensorDataset(mock_images, mock_labels)
        
        # Convert to MONAI format
        class MockDataset:
            def __init__(self, images, labels):
                self.images = images
                self.labels = labels
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                return {
                    "image": self.images[idx],
                    "label": self.labels[idx]
                }
        
        mock_dataset = MockDataset(mock_images, mock_labels)
        return DataLoader(mock_dataset, batch_size=batch_size, shuffle=True)
    
    async def _send_model_update(self, model_params: FLModelParamType, 
                               metrics: Dict[str, float], num_samples: int) -> bool:
        """Send model update to federated learning server"""
        try:
            update_data = {
                'type': 'model_update',
                'node_id': self.config.node_id,
                'round_id': self.current_round_id,
                'model_params': model_params,
                'metrics': metrics,
                'num_samples': num_samples,
                'timestamp': datetime.now().isoformat()
            }
            
            response = await self.comm_manager.send_secure_message_to_server(
                self.config.server_endpoint, update_data
            )
            
            return response.get('status') == 'success'
            
        except Exception as e:
            logger.error(f"Failed to send model update: {e}")
            return False
    
    async def handle_global_model_update(self, update_info: Dict[str, Any]):
        """Handle global model update from server"""
        try:
            round_id = update_info['round_id']
            model_version = update_info['model_version']
            global_params = update_info['model_params']
            aggregated_metrics = update_info['aggregated_metrics']
            
            # Update local model with new global parameters
            await self._update_local_model(global_params)
            
            logger.info(f"Updated local model with global parameters from round {round_id}")
            logger.info(f"Global model version: {model_version}")
            logger.info(f"Aggregated metrics: {aggregated_metrics}")
            
        except Exception as e:
            logger.error(f"Failed to handle global model update: {e}")
    
    def get_client_status(self) -> Dict[str, Any]:
        """Get current client status"""
        return {
            'node_id': self.config.node_id,
            'institution_name': self.config.institution_name,
            'is_registered': self.is_registered,
            'current_round': self.current_round_id,
            'model_initialized': self.model is not None,
            'server_endpoint': self.config.server_endpoint
        }