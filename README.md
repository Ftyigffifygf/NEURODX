# NeuroDx-MultiModal

A comprehensive diagnostic assistant system built on NVIDIA's MONAI framework for detecting and monitoring neurodegenerative diseases through multi-modal medical imaging and wearable sensor data fusion.

## Overview

NeuroDx-MultiModal leverages NVIDIA's ecosystem including:
- **MONAI Core**: Medical imaging deep learning with SwinUNETR architecture
- **MONAI Label**: Active learning annotation workflows
- **MONAI Deploy**: Federated learning and production deployment
- **NVIDIA Palmyra-Med-70B**: Medical text analysis and report generation
- **NVIDIA Genomics**: Genetic variant analysis for neurodegenerative markers

## Features

- **Multi-Modal Data Fusion**: Combines MRI, CT, Ultrasound imaging with EEG, heart rate, sleep, and gait sensor data
- **AI-Powered Diagnostics**: SwinUNETR-based segmentation and classification with explainability features
- **Active Learning**: MONAI Label integration for efficient annotation workflows
- **Federated Learning**: Privacy-preserving multi-institutional model training
- **Healthcare Integration**: FHIR/HL7 compatibility for seamless clinical workflows
- **HIPAA Compliance**: End-to-end encryption and comprehensive audit logging

## Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/neurodx/neurodx-multimodal.git
cd neurodx-multimodal
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your NVIDIA API keys and configuration
```

4. Initialize the system:
```bash
python main.py
```

### Configuration

Copy `.env.example` to `.env` and configure the following:

#### NVIDIA API Configuration
```bash
NVIDIA_PALMYRA_API_KEY=your_nvidia_palmyra_api_key_here
NVIDIA_GENOMICS_WORKFLOW_PATH=./genomics-analysis-blueprint
```

#### MONAI Configuration
```bash
MONAI_DATA_DIRECTORY=./data
MONAI_MODEL_CACHE=./models
```

#### Database Configuration
```bash
DATABASE_URL=postgresql://user:password@localhost:5432/neurodx
REDIS_URL=redis://localhost:6379/0
```

## Architecture

### Directory Structure

```
neurodx-multimodal/
├── src/
│   ├── services/
│   │   ├── image_processing/     # MONAI image processing
│   │   ├── wearable_sensor/      # Sensor data processing
│   │   ├── data_fusion/          # Multi-modal fusion
│   │   ├── ml_inference/         # MONAI ML service
│   │   └── nvidia_integration/   # NVIDIA API clients
│   ├── models/                   # Data models and schemas
│   ├── api/                      # Flask REST API
│   ├── config/                   # Configuration management
│   └── utils/                    # Utility functions
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment configuration template
└── main.py                       # Application entry point
```

### Core Components

1. **Image Processing Service**: MONAI-based medical image preprocessing and validation
2. **Wearable Sensor Service**: Multi-source sensor data collection and synchronization
3. **Data Fusion Service**: Spatial-temporal alignment of multi-modal data
4. **ML Inference Service**: SwinUNETR model for segmentation and classification
5. **NVIDIA Integration**: Unified interface for Palmyra-Med-70B and genomics analysis

## Usage

### Basic Diagnostic Workflow

```python
from src.services.nvidia_integration.nvidia_service import get_nvidia_service

# Initialize NVIDIA service
nvidia_service = get_nvidia_service()

# Generate diagnostic report
report = await nvidia_service.generate_diagnostic_report_with_failover(
    imaging_findings={"segmentation": {...}, "classification": {...}},
    wearable_data={"eeg": {...}, "heart_rate": {...}},
    patient_context={"age": 65, "symptoms": [...]}
)

# Analyze genomic variants
genomics_results = nvidia_service.analyze_genomic_variants_with_monitoring(
    fastq_files=["patient_R1.fastq", "patient_R2.fastq"],
    patient_id="patient_001"
)

# Integrate multi-modal analysis
integrated_results = nvidia_service.integrate_multimodal_analysis(
    imaging_results=imaging_findings,
    wearable_data=wearable_data,
    genomics_results=genomics_results
)
```

### MONAI Label Integration

The system includes MONAI Label for active learning annotation workflows:

```python
# Active learning sample selection
samples = active_learning_engine.select_informative_samples(
    uncertainty_threshold=0.7,
    diversity_metric="entropy"
)

# Annotation interface integration
annotations = monai_label_client.get_annotations(
    task_type="segmentation",
    sample_ids=samples
)
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Deployment

### Docker Deployment

```bash
# Build container
docker build -t neurodx-multimodal .

# Run with docker-compose
docker-compose up -d
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=neurodx-multimodal
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA MONAI team for the medical imaging framework
- NVIDIA AI team for Palmyra-Med-70B and genomics analysis capabilities
- Healthcare partners for clinical validation and feedback

## Support

For support and questions:
- Documentation: [https://neurodx-multimodal.readthedocs.io](https://neurodx-multimodal.readthedocs.io)
- Issues: [GitHub Issues](https://github.com/neurodx/neurodx-multimodal/issues)
- Email: support@neurodx.ai