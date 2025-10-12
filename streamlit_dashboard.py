#!/usr/bin/env python3
"""
NeuroDx-MultiModal Streamlit Dashboard
Interactive visualization for neurodegenerative disease diagnostics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import json
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.models.patient import PatientRecord, ImagingStudy, WearableSession
    from src.models.diagnostics import DiagnosticResult, ModelMetrics
    from src.services.nvidia_integration.nvidia_enhanced_service import NVIDIAEnhancedService
    from src.config.settings import Settings
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="NeuroDx-MultiModal Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high { border-left-color: #d62728; }
    .risk-moderate { border-left-color: #ff7f0e; }
    .risk-low { border-left-color: #2ca02c; }
</style>
""", unsafe_allow_html=True)

def create_sample_patient_data():
    """Create sample patient data for demonstration"""
    return {
        'patient_id': 'PAT_20241012_00001',
        'age': 68,
        'gender': 'F',
        'diagnosis': 'Mild Cognitive Impairment',
        'risk_score': 72.0,
        'confidence': 84.0,
        'imaging_studies': ['MRI', 'CT'],
        'wearable_sessions': ['EEG', 'HeartRate', 'Sleep', 'Gait'],
        'last_visit': '2024-10-12'
    }

def create_diagnostic_metrics():
    """Create sample diagnostic metrics"""
    return {
        'dice_score': 0.890,
        'auc_score': 0.920,
        'sensitivity': 0.870,
        'specificity': 0.940,
        'accuracy': 0.905
    }

def create_risk_assessment():
    """Create risk assessment data"""
    return {
        'Alzheimer\'s Disease': 68.0,
        'Mild Cognitive Impairment': 82.0,
        'Vascular Dementia': 23.0,
        'Frontotemporal Dementia': 15.0,
        'Parkinson\'s Disease': 12.0
    }

def create_brain_visualization():
    """Create 3D brain visualization"""
    # Generate sample brain data
    theta = np.linspace(0, 2*np.pi, 50)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    
    # Brain-like shape
    r = 1 + 0.3 * np.sin(3*theta) * np.sin(2*phi)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    fig = go.Figure(data=[go.Surface(
        x=x, y=y, z=z,
        colorscale='Viridis',
        opacity=0.8,
        showscale=False
    )])
    
    fig.update_layout(
        title="3D Brain Model with Segmentation",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=400
    )
    
    return fig

def create_eeg_waveform():
    """Create EEG waveform visualization"""
    time = np.linspace(0, 10, 1000)
    
    # Simulate different EEG frequency bands
    alpha = np.sin(2 * np.pi * 10 * time) * np.exp(-time/5)
    beta = 0.5 * np.sin(2 * np.pi * 20 * time) * np.random.normal(1, 0.1, len(time))
    theta = 0.8 * np.sin(2 * np.pi * 6 * time)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=alpha, name='Alpha (8-12 Hz)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=time, y=beta + 2, name='Beta (13-30 Hz)', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=time, y=theta + 4, name='Theta (4-7 Hz)', line=dict(color='green')))
    
    fig.update_layout(
        title="EEG Waveform Analysis",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (μV)",
        height=300
    )
    
    return fig

def create_risk_gauge(risk_score, title):
    """Create risk assessment gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 NeuroDx-MultiModal Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", [
        "Overview",
        "Patient Analysis", 
        "System Monitoring",
        "Configuration"
    ])
    
    if page == "Overview":
        show_overview()
    elif page == "Patient Analysis":
        show_patient_analysis()
    elif page == "System Monitoring":
        show_system_monitoring()
    elif page == "Configuration":
        show_configuration()

def show_overview():
    """Show system overview page"""
    st.header("System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Patients", "1,247", "↑ 23")
    with col2:
        st.metric("Processed Studies", "3,891", "↑ 156")
    with col3:
        st.metric("System Uptime", "99.8%", "↑ 0.1%")
    with col4:
        st.metric("AI Accuracy", "92.0%", "↑ 1.2%")
    
    # Recent activity
    st.subheader("Recent Diagnostic Activity")
    
    # Sample data
    recent_data = pd.DataFrame({
        'Patient ID': ['PAT_20241012_00001', 'PAT_20241012_00002', 'PAT_20241012_00003'],
        'Study Type': ['MRI + EEG', 'CT + Gait', 'MRI + Sleep'],
        'Risk Score': [72.0, 45.0, 89.0],
        'Status': ['Completed', 'Processing', 'Completed'],
        'Timestamp': ['2024-10-12 14:30', '2024-10-12 14:15', '2024-10-12 14:00']
    })
    
    st.dataframe(recent_data, use_container_width=True)
    
    # System health
    st.subheader("System Health")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Service status
        services = {
            'NVIDIA Palmyra': 'Healthy',
            'MONAI Services': 'Healthy', 
            'Database': 'Healthy',
            'API Gateway': 'Degraded'
        }
        
        for service, status in services.items():
            color = "🟢" if status == "Healthy" else "🟡" if status == "Degraded" else "🔴"
            st.write(f"{color} {service}: {status}")
    
    with col2:
        # Performance metrics
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(24)),
            y=np.random.normal(95, 5, 24),
            mode='lines+markers',
            name='System Performance (%)',
            line=dict(color='blue')
        ))
        fig.update_layout(
            title="24-Hour Performance",
            xaxis_title="Hour",
            yaxis_title="Performance (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

def show_patient_analysis():
    """Show detailed patient analysis page"""
    st.header("Patient Analysis")
    
    # Patient selector
    patient_data = create_sample_patient_data()
    st.subheader(f"Patient: {patient_data['patient_id']}")
    
    # Patient info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Age", f"{patient_data['age']} years")
    with col2:
        st.metric("Gender", patient_data['gender'])
    with col3:
        st.metric("Last Visit", patient_data['last_visit'])
    
    # Risk assessment
    st.subheader("Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall risk gauge
        risk_fig = create_risk_gauge(patient_data['risk_score'], "Overall Risk Score")
        st.plotly_chart(risk_fig, use_container_width=True)
    
    with col2:
        # Confidence gauge
        conf_fig = create_risk_gauge(patient_data['confidence'], "AI Confidence")
        st.plotly_chart(conf_fig, use_container_width=True)
    
    # Detailed risk breakdown
    st.subheader("Condition-Specific Risk Analysis")
    
    risk_data = create_risk_assessment()
    risk_df = pd.DataFrame(list(risk_data.items()), columns=['Condition', 'Risk %'])
    
    fig = px.bar(risk_df, x='Condition', y='Risk %', 
                 color='Risk %', color_continuous_scale='RdYlGn_r',
                 title="Risk Assessment by Condition")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Multi-modal analysis
    st.subheader("Multi-Modal Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 3D Brain visualization
        brain_fig = create_brain_visualization()
        st.plotly_chart(brain_fig, use_container_width=True)
    
    with col2:
        # EEG waveform
        eeg_fig = create_eeg_waveform()
        st.plotly_chart(eeg_fig, use_container_width=True)
    
    # Model performance
    st.subheader("Model Performance Metrics")
    
    metrics = create_diagnostic_metrics()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Dice Score", f"{metrics['dice_score']:.3f}")
    with col2:
        st.metric("AUC Score", f"{metrics['auc_score']:.3f}")
    with col3:
        st.metric("Sensitivity", f"{metrics['sensitivity']:.3f}")
    with col4:
        st.metric("Specificity", f"{metrics['specificity']:.3f}")
    with col5:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")

def show_system_monitoring():
    """Show system monitoring page"""
    st.header("System Monitoring")
    
    # Real-time metrics
    st.subheader("Real-Time System Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CPU Usage", "67%", "↑ 5%")
    with col2:
        st.metric("Memory Usage", "4.2 GB", "↑ 0.3 GB")
    with col3:
        st.metric("GPU Usage", "89%", "↑ 12%")
    with col4:
        st.metric("API Calls/min", "234", "↓ 12")
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU/Memory usage over time
        time_data = pd.date_range(start='2024-10-12 00:00', periods=24, freq='H')
        cpu_data = np.random.normal(65, 10, 24)
        memory_data = np.random.normal(4.0, 0.5, 24)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=time_data, y=cpu_data, name="CPU %"), secondary_y=False)
        fig.add_trace(go.Scatter(x=time_data, y=memory_data, name="Memory GB"), secondary_y=True)
        
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="CPU Usage (%)", secondary_y=False)
        fig.update_yaxes(title_text="Memory Usage (GB)", secondary_y=True)
        fig.update_layout(title="System Resource Usage", height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # API response times
        response_times = np.random.lognormal(mean=2, sigma=0.5, size=100)
        
        fig = go.Figure(data=[go.Histogram(x=response_times, nbinsx=20)])
        fig.update_layout(
            title="API Response Time Distribution",
            xaxis_title="Response Time (ms)",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Service health
    st.subheader("Service Health Status")
    
    services_health = {
        'NVIDIA Palmyra API': {'status': 'Healthy', 'response_time': '145ms', 'uptime': '99.9%'},
        'MONAI Label Server': {'status': 'Healthy', 'response_time': '89ms', 'uptime': '99.8%'},
        'PostgreSQL Database': {'status': 'Healthy', 'response_time': '12ms', 'uptime': '100%'},
        'Redis Cache': {'status': 'Healthy', 'response_time': '3ms', 'uptime': '99.9%'},
        'MinIO Storage': {'status': 'Degraded', 'response_time': '234ms', 'uptime': '98.5%'}
    }
    
    for service, health in services_health.items():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            status_color = "🟢" if health['status'] == 'Healthy' else "🟡"
            st.write(f"{status_color} **{service}**")
        with col2:
            st.write(f"Status: {health['status']}")
        with col3:
            st.write(f"Response: {health['response_time']}")
        with col4:
            st.write(f"Uptime: {health['uptime']}")

def show_configuration():
    """Show configuration page"""
    st.header("System Configuration")
    
    # NVIDIA API Configuration
    st.subheader("NVIDIA API Configuration")
    
    with st.expander("API Settings"):
        api_endpoint = st.text_input("API Endpoint", "https://integrate.api.nvidia.com/v1")
        api_keys = st.number_input("Number of API Keys", min_value=1, max_value=10, value=3)
        temperature = st.slider("Model Temperature", 0.0, 1.0, 0.7)
        max_tokens = st.number_input("Max Tokens", min_value=100, max_value=4000, value=1000)
    
    # Model Configuration
    st.subheader("Model Configuration")
    
    with st.expander("MONAI Model Settings"):
        model_type = st.selectbox("Model Type", ["SwinUNETR", "UNETR", "SegResNet"])
        input_size = st.selectbox("Input Size", ["96x96x96", "128x128x128", "256x256x256"])
        batch_size = st.number_input("Batch Size", min_value=1, max_value=16, value=4)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
    
    # Security Configuration
    st.subheader("Security Configuration")
    
    with st.expander("Security Settings"):
        encryption_enabled = st.checkbox("Enable Encryption", value=True)
        audit_logging = st.checkbox("Enable Audit Logging", value=True)
        mfa_required = st.checkbox("Require Multi-Factor Authentication", value=True)
        session_timeout = st.number_input("Session Timeout (minutes)", min_value=5, max_value=480, value=60)
    
    # Save configuration
    if st.button("Save Configuration"):
        st.success("Configuration saved successfully!")
        
        # Show current configuration
        config = {
            "nvidia_api": {
                "endpoint": api_endpoint,
                "api_keys": api_keys,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            "model": {
                "type": model_type,
                "input_size": input_size,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            },
            "security": {
                "encryption_enabled": encryption_enabled,
                "audit_logging": audit_logging,
                "mfa_required": mfa_required,
                "session_timeout": session_timeout
            }
        }
        
        st.json(config)

if __name__ == "__main__":
    main()