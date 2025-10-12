#!/usr/bin/env python3
"""
Comprehensive MONAI Data Integration for NeuroDx-MultiModal System

This module integrates all MONAI data handling capabilities including:
- Advanced data loading and caching
- Multi-modal dataset management
- Distributed data loading
- Smart caching and prefetching
- Medical image format support
- Cross-validation and data splitting
- Performance optimization
"""

import logging
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist

# MONAI Core Data Components
from monai.data import (
    # Core datasets
    Dataset as MONAIDataset,
    CacheDataset,
    PersistentDataset,
    SmartCacheDataset,
    LMDBDataset,
    ZipDataset,
    ArrayDataset,
    ImageDataset,
    
    # Data loaders
    DataLoader as MONAIDataLoader,
    ThreadDataLoader,
    
    # Utilities
    DatasetSummary,
    partition_dataset,
    partition_dataset_classes,
    select_cross_validation_folds,
    create_cross_validation_datalist,
    
    # Samplers
    DistributedSampler as MONAIDistributedSampler,
    DistributedWeightedRandomSampler,
    
    # Decathlon utilities
    DecathlonDataset,
    load_decathlon_datalist,
    load_decathlon_properties,
    
    # Image readers
    ITKReader,
    NibabelReader,
    NumpyReader,
    PILReader,
    PydicomReader,
    
    # Meta tensor utilities
    MetaTensor,
    convert_to_tensor,
    convert_data_type,
    
    # Folder layout utilities
    FolderLayout,
    FolderLayoutDataset,
    
    # CSV utilities
    CSVDataset,
    CSVSaver,
    
    # Video utilities
    VideoDataset,
    
    # Patch-based utilities
    PatchDataset,
    GridPatchDataset,
    
    # Synthetic data
    SyntheticDataset,
    
    # Utilities
    list_data_collate,
    pad_list_data_collate,
    no_collation,
    set_track_meta,
    is_track_meta,
)

# MONAI Transforms
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ToTensord,
    Spacingd, Orientationd, ScaleIntensityRanged,
    RandFlipd, RandRotated, RandZoomd,
    CacheTransform, apply_transform
)

# MONAI Utils
from monai.utils import (
    ensure_tuple, ensure_tuple_rep, ensure_tuple_size,
    convert_to_numpy, convert_to_tensor,
    GridSampleMode, GridSamplePadMode,
    InterpolateMode, NumpyPadMode,
    BlendMode, PytorchPadMode,
    set_determinism, get_seed
)

from src.utils.logging_config import get_logger
from src.models.patient import PatientRecord, ImagingStudy, WearableSession

logger = get_logger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset creation and management."""
    name: str
    data_root: Path
    cache_dir: Optional[Path] = None
    cache_rate: float = 1.0
    num_workers: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 2
    batch_size: int = 1
    shuffle: bool = True
    drop_last: bool = False
    distributed: bool = False
    seed: Optional[int] = None
    
    # Smart caching parameters
    cache_num: int = 100
    replace_rate: float = 0.1
    
    # Cross-validation parameters
    num_folds: int = 5
    fold_index: int = 0
    
    # Data splitting ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class MultiModalDataItem:
    """Data item for multi-modal medical data."""
    patient_id: str
    study_id: str
    modalities: Dict[str, str]  # modality_name -> file_path
    labels: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    wearable_data: Optional[Dict[str, Any]] = None
    genomic_data: Optional[Dict[str, Any]] = None


class NeuroDxDataset(MONAIDataset):
    """
    Custom MONAI dataset for NeuroDx multi-modal data.
    Extends MONAI Dataset with specialized handling for neurodegenerative disease data.
    """
    
    def __init__(self,
                 data_list: List[MultiModalDataItem],
                 transform: Optional[Callable] = None,
                 cache_transforms: bool = True):
        """
        Initialize NeuroDx dataset.
        
        Args:
            data_list: List of multi-modal data items
            transform: Transform pipeline to apply
            cache_transforms: Whether to cache transform results
        """
        # Convert to MONAI-compatible format
        monai_data_list = self._convert_to_monai_format(data_list)
        
        super().__init__(data=monai_data_list, transform=transform)
        
        self.original_data_list = data_list
        self.cache_transforms = cache_transforms
        self._transform_cache = {} if cache_transforms else None
        
        logger.info(f"Created NeuroDx dataset with {len(data_list)} items")
    
    def _convert_to_monai_format(self, data_list: List[MultiModalDataItem]) -> List[Dict[str, Any]]:
        """Convert NeuroDx data items to MONAI-compatible format."""
        monai_data = []
        
        for item in data_list:
            monai_item = {
                'patient_id': item.patient_id,
                'study_id': item.study_id,
                **item.modalities,  # Add modality file paths
                **item.labels,      # Add labels
                'metadata': item.metadata
            }
            
            # Add wearable and genomic data if available
            if item.wearable_data:
                monai_item['wearable'] = item.wearable_data
            if item.genomic_data:
                monai_item['genomic'] = item.genomic_data
            
            monai_data.append(monai_item)
        
        return monai_data
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get item with optional transform caching."""
        if self.cache_transforms and index in self._transform_cache:
            return self._transform_cache[index]
        
        item = super().__getitem__(index)
        
        if self.cache_transforms:
            self._transform_cache[index] = item
        
        return item


class MONAIDataManager:
    """
    Comprehensive data manager integrating all MONAI data capabilities.
    Handles multi-modal medical data with advanced caching, loading, and optimization.
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize MONAI data manager.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.datasets = {}
        self.data_loaders = {}
        self.readers = self._initialize_readers()
        
        # Set determinism if seed provided
        if config.seed is not None:
            set_determinism(seed=config.seed)
        
        # Initialize distributed training if needed
        if config.distributed:
            self._initialize_distributed()
        
        logger.info(f"Initialized MONAI data manager with config: {config.name}")
    
    def _initialize_readers(self) -> Dict[str, Any]:
        """Initialize specialized image readers for different formats."""
        return {
            'nifti': NibabelReader(),
            'dicom': PydicomReader(),
            'numpy': NumpyReader(),
            'itk': ITKReader(),
            'pil': PILReader()
        }
    
    def _initialize_distributed(self):
        """Initialize distributed training setup."""
        if not dist.is_initialized():
            logger.warning("Distributed training requested but not initialized")
            return
        
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        logger.info(f"Initialized distributed training: rank {self.rank}/{self.world_size}")
    
    def create_dataset_from_patient_records(self,
                                          patient_records: List[PatientRecord],
                                          dataset_type: str = "cache") -> MONAIDataset:
        """
        Create MONAI dataset from patient records.
        
        Args:
            patient_records: List of patient records
            dataset_type: Type of dataset to create
            
        Returns:
            MONAI dataset instance
        """
        # Convert patient records to data items
        data_items = self._convert_patient_records(patient_records)
        
        # Create transforms
        transforms = self._create_default_transforms()
        
        # Create dataset based on type
        if dataset_type == "cache":
            dataset = CacheDataset(
                data=data_items,
                transform=transforms,
                cache_rate=self.config.cache_rate,
                num_workers=self.config.num_workers
            )
        elif dataset_type == "persistent":
            cache_dir = self.config.cache_dir or Path("./cache")
            dataset = PersistentDataset(
                data=data_items,
                transform=transforms,
                cache_dir=cache_dir
            )
        elif dataset_type == "smart_cache":
            dataset = SmartCacheDataset(
                data=data_items,
                transform=transforms,
                cache_num=self.config.cache_num,
                replace_rate=self.config.replace_rate,
                num_init_workers=self.config.num_workers,
                num_replace_workers=self.config.num_workers // 2
            )
        elif dataset_type == "lmdb":
            cache_dir = self.config.cache_dir or Path("./lmdb_cache")
            dataset = LMDBDataset(
                data=data_items,
                transform=transforms,
                cache_dir=cache_dir
            )
        else:
            dataset = NeuroDxDataset(
                data_list=data_items,
                transform=transforms
            )
        
        self.datasets[dataset_type] = dataset
        logger.info(f"Created {dataset_type} dataset with {len(data_items)} items")
        
        return dataset
    
    def _convert_patient_records(self, patient_records: List[PatientRecord]) -> List[Dict[str, Any]]:
        """Convert patient records to MONAI-compatible data format."""
        data_items = []
        
        for patient in patient_records:
            for study in patient.imaging_studies:
                # Create base data item
                data_item = {
                    'patient_id': patient.patient_id,
                    'study_id': study.study_id,
                    'image': study.file_path,
                    'modality': study.modality,
                    'demographics': {
                        'age': patient.demographics.age,
                        'gender': patient.demographics.gender,
                        'weight_kg': patient.demographics.weight_kg,
                        'height_cm': patient.demographics.height_cm
                    },
                    'metadata': {
                        'acquisition_date': study.acquisition_date.isoformat(),
                        'scanner_manufacturer': study.scanner_manufacturer,
                        'scanner_model': study.scanner_model,
                        'slice_thickness': study.slice_thickness,
                        'pixel_spacing': study.pixel_spacing
                    }
                }
                
                # Add wearable data if available
                wearable_data = {}
                for session in patient.wearable_data:
                    if session.device_type not in wearable_data:
                        wearable_data[session.device_type] = []
                    
                    wearable_data[session.device_type].append({
                        'session_id': session.session_id,
                        'start_time': session.start_time.isoformat(),
                        'end_time': session.end_time.isoformat(),
                        'sampling_rate': session.sampling_rate,
                        'processed_features': session.processed_features
                    })
                
                if wearable_data:
                    data_item['wearable'] = wearable_data
                
                # Add longitudinal data if available
                if patient.longitudinal_tracking:
                    data_item['longitudinal'] = {
                        'baseline_date': patient.longitudinal_tracking.baseline_date.isoformat(),
                        'follow_up_dates': [d.isoformat() for d in patient.longitudinal_tracking.follow_up_dates],
                        'progression_metrics': patient.longitudinal_tracking.progression_metrics
                    }
                
                data_items.append(data_item)
        
        return data_items
    
    def _create_default_transforms(self) -> Compose:
        """Create default transform pipeline."""
        transforms = [
            LoadImaged(keys=['image'], reader=self.readers['nifti']),
            EnsureChannelFirstd(keys=['image']),
            Orientationd(keys=['image'], axcodes="RAS"),
            Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            ScaleIntensityRanged(keys=['image'], a_min=-1000, a_max=4000, b_min=0.0, b_max=1.0, clip=True),
            ToTensord(keys=['image'])
        ]
        
        return Compose(transforms)
    
    def create_data_loader(self,
                          dataset: MONAIDataset,
                          loader_type: str = "standard",
                          **kwargs) -> DataLoader:
        """
        Create optimized data loader.
        
        Args:
            dataset: MONAI dataset
            loader_type: Type of data loader
            **kwargs: Additional loader arguments
            
        Returns:
            Configured data loader
        """
        # Default loader configuration
        loader_config = {
            'batch_size': self.config.batch_size,
            'shuffle': self.config.shuffle,
            'num_workers': self.config.num_workers,
            'pin_memory': self.config.pin_memory,
            'persistent_workers': self.config.persistent_workers,
            'prefetch_factor': self.config.prefetch_factor,
            'drop_last': self.config.drop_last,
            'collate_fn': list_data_collate
        }
        
        # Update with provided kwargs
        loader_config.update(kwargs)
        
        # Add distributed sampler if needed
        if self.config.distributed:
            sampler = MONAIDistributedSampler(dataset, shuffle=loader_config['shuffle'])
            loader_config['sampler'] = sampler
            loader_config['shuffle'] = False  # Disable shuffle when using sampler
        
        # Create loader based on type
        if loader_type == "thread":
            data_loader = ThreadDataLoader(dataset, **loader_config)
        else:
            data_loader = MONAIDataLoader(dataset, **loader_config)
        
        self.data_loaders[loader_type] = data_loader
        logger.info(f"Created {loader_type} data loader with batch size {loader_config['batch_size']}")
        
        return data_loader
    
    def create_cross_validation_splits(self,
                                     data_list: List[Dict[str, Any]],
                                     stratify_key: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create cross-validation data splits.
        
        Args:
            data_list: List of data items
            stratify_key: Key to use for stratification
            
        Returns:
            Dictionary with train/val splits for each fold
        """
        # Create cross-validation datalist
        cv_datalist = create_cross_validation_datalist(
            data_list=data_list,
            nfolds=self.config.num_folds,
            train_folds=list(range(self.config.num_folds)),
            val_folds=[self.config.fold_index]
        )
        
        # Select specific fold
        train_data, val_data = select_cross_validation_folds(
            partitions=cv_datalist,
            folds=self.config.fold_index
        )
        
        splits = {
            'train': train_data,
            'validation': val_data
        }
        
        logger.info(f"Created CV splits: train={len(train_data)}, val={len(val_data)}")
        return splits
    
    def create_train_val_test_splits(self,
                                   data_list: List[Dict[str, Any]],
                                   stratify_key: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create train/validation/test splits.
        
        Args:
            data_list: List of data items
            stratify_key: Key to use for stratification
            
        Returns:
            Dictionary with train/val/test splits
        """
        # Calculate split sizes
        total_size = len(data_list)
        train_size = int(total_size * self.config.train_ratio)
        val_size = int(total_size * self.config.val_ratio)
        test_size = total_size - train_size - val_size
        
        # Partition dataset
        if stratify_key:
            # Stratified split based on key
            partitions = partition_dataset_classes(
                data=data_list,
                classes=[item[stratify_key] for item in data_list],
                ratios=[self.config.train_ratio, self.config.val_ratio, self.config.test_ratio],
                shuffle=True,
                seed=self.config.seed
            )
            train_data, val_data, test_data = partitions
        else:
            # Random split
            partitions = partition_dataset(
                data=data_list,
                ratios=[self.config.train_ratio, self.config.val_ratio, self.config.test_ratio],
                shuffle=True,
                seed=self.config.seed
            )
            train_data, val_data, test_data = partitions
        
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        logger.info(f"Created splits: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        return splits
    
    def create_decathlon_dataset(self,
                               data_root: Path,
                               task_name: str,
                               section: str = "training") -> DecathlonDataset:
        """
        Create Medical Segmentation Decathlon dataset.
        
        Args:
            data_root: Root directory of decathlon data
            task_name: Name of the decathlon task
            section: Dataset section (training/validation)
            
        Returns:
            Decathlon dataset
        """
        dataset = DecathlonDataset(
            root_dir=data_root,
            task=task_name,
            section=section,
            transform=self._create_default_transforms(),
            download=False,  # Assume data is already downloaded
            cache_rate=self.config.cache_rate,
            num_workers=self.config.num_workers
        )
        
        logger.info(f"Created Decathlon dataset: {task_name} ({section})")
        return dataset
    
    def create_folder_layout_dataset(self,
                                   root_dir: Path,
                                   folder_layout: FolderLayout) -> FolderLayoutDataset:
        """
        Create dataset from folder layout.
        
        Args:
            root_dir: Root directory
            folder_layout: Folder layout specification
            
        Returns:
            Folder layout dataset
        """
        dataset = FolderLayoutDataset(
            root_dir=root_dir,
            folder_layout=folder_layout,
            transform=self._create_default_transforms()
        )
        
        logger.info(f"Created folder layout dataset from {root_dir}")
        return dataset
    
    def create_csv_dataset(self,
                          csv_file: Path,
                          transform: Optional[Callable] = None) -> CSVDataset:
        """
        Create dataset from CSV file.
        
        Args:
            csv_file: Path to CSV file
            transform: Transform pipeline
            
        Returns:
            CSV dataset
        """
        dataset = CSVDataset(
            src=csv_file,
            transform=transform or self._create_default_transforms()
        )
        
        logger.info(f"Created CSV dataset from {csv_file}")
        return dataset
    
    def create_synthetic_dataset(self,
                               num_samples: int,
                               image_shape: Tuple[int, ...] = (96, 96, 96),
                               num_classes: int = 4) -> SyntheticDataset:
        """
        Create synthetic dataset for testing.
        
        Args:
            num_samples: Number of synthetic samples
            image_shape: Shape of synthetic images
            num_classes: Number of classes for labels
            
        Returns:
            Synthetic dataset
        """
        dataset = SyntheticDataset(
            size=num_samples,
            image_shape=image_shape,
            num_classes=num_classes,
            transform=self._create_default_transforms()
        )
        
        logger.info(f"Created synthetic dataset with {num_samples} samples")
        return dataset
    
    def get_dataset_summary(self, dataset: MONAIDataset) -> Dict[str, Any]:
        """
        Get comprehensive dataset summary.
        
        Args:
            dataset: MONAI dataset
            
        Returns:
            Dataset summary information
        """
        summary = DatasetSummary(dataset)
        
        summary_info = {
            'num_samples': len(dataset),
            'data_shape': summary.data_shape,
            'data_type': summary.data_type,
            'data_range': summary.data_range,
            'spacing': getattr(summary, 'spacing', None),
            'orientation': getattr(summary, 'orientation', None)
        }
        
        logger.info(f"Generated dataset summary: {summary_info}")
        return summary_info
    
    def optimize_data_loading(self,
                            dataset: MONAIDataset,
                            profile_iterations: int = 100) -> Dict[str, Any]:
        """
        Profile and optimize data loading performance.
        
        Args:
            dataset: Dataset to optimize
            profile_iterations: Number of iterations for profiling
            
        Returns:
            Optimization recommendations
        """
        logger.info("Profiling data loading performance...")
        
        # Test different configurations
        configs_to_test = [
            {'num_workers': 0, 'pin_memory': False},
            {'num_workers': 2, 'pin_memory': True},
            {'num_workers': 4, 'pin_memory': True},
            {'num_workers': 8, 'pin_memory': True},
        ]
        
        results = {}
        
        for config in configs_to_test:
            loader = self.create_data_loader(dataset, **config)
            
            # Time data loading
            start_time = time.time()
            for i, batch in enumerate(loader):
                if i >= profile_iterations:
                    break
            end_time = time.time()
            
            avg_time = (end_time - start_time) / min(profile_iterations, len(loader))
            results[str(config)] = avg_time
        
        # Find optimal configuration
        optimal_config = min(results.items(), key=lambda x: x[1])
        
        optimization_report = {
            'tested_configs': results,
            'optimal_config': optimal_config[0],
            'optimal_time': optimal_config[1],
            'recommendations': self._generate_optimization_recommendations(results)
        }
        
        logger.info(f"Data loading optimization complete: {optimization_report}")
        return optimization_report
    
    def _generate_optimization_recommendations(self, results: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations based on profiling results."""
        recommendations = []
        
        # Analyze results
        times = list(results.values())
        min_time = min(times)
        max_time = max(times)
        
        if max_time / min_time > 2.0:
            recommendations.append("Significant performance variation detected - consider optimal configuration")
        
        # Check if multiprocessing helps
        single_worker_time = None
        multi_worker_times = []
        
        for config_str, time_val in results.items():
            config = eval(config_str)
            if config['num_workers'] == 0:
                single_worker_time = time_val
            else:
                multi_worker_times.append(time_val)
        
        if single_worker_time and multi_worker_times:
            avg_multi_time = sum(multi_worker_times) / len(multi_worker_times)
            if single_worker_time < avg_multi_time:
                recommendations.append("Single-threaded loading may be optimal for this dataset")
            else:
                recommendations.append("Multi-threaded loading provides performance benefits")
        
        return recommendations
    
    def save_dataset_cache(self, dataset: MONAIDataset, cache_path: Path):
        """Save dataset cache to disk."""
        if hasattr(dataset, '_cache'):
            with open(cache_path, 'wb') as f:
                pickle.dump(dataset._cache, f)
            logger.info(f"Saved dataset cache to {cache_path}")
    
    def load_dataset_cache(self, dataset: MONAIDataset, cache_path: Path):
        """Load dataset cache from disk."""
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            if hasattr(dataset, '_cache'):
                dataset._cache = cache
            logger.info(f"Loaded dataset cache from {cache_path}")
    
    def cleanup_resources(self):
        """Clean up data manager resources."""
        # Clear dataset caches
        for dataset in self.datasets.values():
            if hasattr(dataset, '_cache'):
                dataset._cache.clear()
        
        # Close data loaders
        for loader in self.data_loaders.values():
            if hasattr(loader, '_shutdown_workers'):
                loader._shutdown_workers()
        
        logger.info("Cleaned up data manager resources")


def create_optimized_data_pipeline(config: DatasetConfig,
                                 patient_records: List[PatientRecord]) -> Tuple[MONAIDataManager, Dict[str, DataLoader]]:
    """
    Create optimized data pipeline for NeuroDx system.
    
    Args:
        config: Dataset configuration
        patient_records: List of patient records
        
    Returns:
        Tuple of data manager and data loaders
    """
    # Initialize data manager
    data_manager = MONAIDataManager(config)
    
    # Create dataset
    dataset = data_manager.create_dataset_from_patient_records(
        patient_records=patient_records,
        dataset_type="smart_cache"  # Use smart caching for optimal performance
    )
    
    # Create data splits
    data_list = data_manager._convert_patient_records(patient_records)
    splits = data_manager.create_train_val_test_splits(data_list)
    
    # Create datasets for each split
    train_dataset = data_manager.create_dataset_from_patient_records(
        [pr for pr in patient_records if any(study.study_id in [item['study_id'] for item in splits['train']] 
                                           for study in pr.imaging_studies)],
        dataset_type="smart_cache"
    )
    
    val_dataset = data_manager.create_dataset_from_patient_records(
        [pr for pr in patient_records if any(study.study_id in [item['study_id'] for item in splits['validation']] 
                                           for study in pr.imaging_studies)],
        dataset_type="cache"
    )
    
    test_dataset = data_manager.create_dataset_from_patient_records(
        [pr for pr in patient_records if any(study.study_id in [item['study_id'] for item in splits['test']] 
                                           for study in pr.imaging_studies)],
        dataset_type="cache"
    )
    
    # Create optimized data loaders
    data_loaders = {
        'train': data_manager.create_data_loader(
            train_dataset,
            shuffle=True,
            batch_size=config.batch_size
        ),
        'validation': data_manager.create_data_loader(
            val_dataset,
            shuffle=False,
            batch_size=config.batch_size
        ),
        'test': data_manager.create_data_loader(
            test_dataset,
            shuffle=False,
            batch_size=1  # Single sample for detailed analysis
        )
    }
    
    logger.info("Created optimized data pipeline with smart caching and performance optimization")
    
    return data_manager, data_loaders


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = DatasetConfig(
        name="neurodx_multimodal",
        data_root=Path("./data"),
        cache_dir=Path("./cache"),
        cache_rate=1.0,
        num_workers=4,
        batch_size=2,
        num_folds=5,
        seed=42
    )
    
    # Create synthetic patient records for testing
    from datetime import datetime
    from src.models.patient import PatientRecord, Demographics, ImagingStudy
    
    # This would normally come from your database
    patient_records = []
    
    # Create data manager and pipeline
    data_manager = MONAIDataManager(config)
    
    # Create synthetic dataset for testing
    synthetic_dataset = data_manager.create_synthetic_dataset(
        num_samples=100,
        image_shape=(96, 96, 96),
        num_classes=4
    )
    
    # Create data loader
    data_loader = data_manager.create_data_loader(synthetic_dataset)
    
    # Test data loading
    for batch in data_loader:
        print(f"Batch shape: {batch['image'].shape}")
        break
    
    print("MONAI data integration test completed successfully!")