-- Initialize NeuroDx-MultiModal database schema

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS neurodx;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set search path
SET search_path TO neurodx, public;

-- Patient records table
CREATE TABLE IF NOT EXISTS patients (
    patient_id VARCHAR(50) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    demographics JSONB,
    medical_history JSONB,
    consent_status BOOLEAN DEFAULT FALSE,
    privacy_settings JSONB
);

-- Imaging studies table
CREATE TABLE IF NOT EXISTS imaging_studies (
    study_id VARCHAR(50) PRIMARY KEY,
    patient_id VARCHAR(50) REFERENCES patients(patient_id),
    modality VARCHAR(20) NOT NULL,
    acquisition_date TIMESTAMP WITH TIME ZONE,
    file_path TEXT,
    file_size BIGINT,
    preprocessing_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Wearable sessions table
CREATE TABLE IF NOT EXISTS wearable_sessions (
    session_id VARCHAR(50) PRIMARY KEY,
    patient_id VARCHAR(50) REFERENCES patients(patient_id),
    device_type VARCHAR(20) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    sampling_rate FLOAT,
    data_quality_score FLOAT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Diagnostic results table
CREATE TABLE IF NOT EXISTS diagnostic_results (
    result_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id VARCHAR(50) REFERENCES patients(patient_id),
    study_id VARCHAR(50) REFERENCES imaging_studies(study_id),
    model_version VARCHAR(50),
    segmentation_path TEXT,
    classification_results JSONB,
    confidence_scores JSONB,
    explainability_data JSONB,
    metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Annotations table for MONAI Label
CREATE TABLE IF NOT EXISTS annotations (
    annotation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    study_id VARCHAR(50) REFERENCES imaging_studies(study_id),
    annotator_id VARCHAR(50),
    annotation_type VARCHAR(20),
    annotation_data JSONB,
    quality_score FLOAT,
    review_status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Federated learning nodes table
CREATE TABLE IF NOT EXISTS federated_nodes (
    node_id VARCHAR(50) PRIMARY KEY,
    node_name VARCHAR(100),
    institution VARCHAR(100),
    endpoint_url TEXT,
    status VARCHAR(20) DEFAULT 'inactive',
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    capabilities JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model training sessions table
CREATE TABLE IF NOT EXISTS training_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_type VARCHAR(50),
    training_config JSONB,
    federated_nodes TEXT[],
    status VARCHAR(20) DEFAULT 'pending',
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    metrics JSONB,
    model_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit log table for HIPAA compliance
CREATE TABLE IF NOT EXISTS audit_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(50),
    action VARCHAR(50),
    resource_type VARCHAR(50),
    resource_id VARCHAR(50),
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    details JSONB
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_patients_created_at ON patients(created_at);
CREATE INDEX IF NOT EXISTS idx_imaging_studies_patient_id ON imaging_studies(patient_id);
CREATE INDEX IF NOT EXISTS idx_imaging_studies_modality ON imaging_studies(modality);
CREATE INDEX IF NOT EXISTS idx_wearable_sessions_patient_id ON wearable_sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_wearable_sessions_device_type ON wearable_sessions(device_type);
CREATE INDEX IF NOT EXISTS idx_diagnostic_results_patient_id ON diagnostic_results(patient_id);
CREATE INDEX IF NOT EXISTS idx_annotations_study_id ON annotations(study_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);

-- Create monitoring schema tables
SET search_path TO monitoring, public;

-- System health metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(50),
    metric_name VARCHAR(50),
    metric_value FLOAT,
    unit VARCHAR(20),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tags JSONB
);

-- API performance metrics
CREATE TABLE IF NOT EXISTS api_metrics (
    request_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(100),
    method VARCHAR(10),
    status_code INTEGER,
    response_time_ms FLOAT,
    request_size BIGINT,
    response_size BIGINT,
    user_id VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model performance metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(50),
    model_version VARCHAR(50),
    metric_type VARCHAR(50),
    metric_value FLOAT,
    dataset_size INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Create indexes for monitoring tables
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_metrics_service ON system_metrics(service_name);
CREATE INDEX IF NOT EXISTS idx_api_metrics_timestamp ON api_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_metrics_endpoint ON api_metrics(endpoint);
CREATE INDEX IF NOT EXISTS idx_model_metrics_timestamp ON model_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_model_metrics_model ON model_metrics(model_name);

-- Reset search path
SET search_path TO public;