import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Alert, Button, Spinner, Badge } from 'react-bootstrap';
import SegmentationViewer from '../components/DiagnosticResults/SegmentationViewer';
import ClassificationResults from '../components/DiagnosticResults/ClassificationResults';
import ExplainabilityVisualization from '../components/DiagnosticResults/ExplainabilityVisualization';
import axios from 'axios';

const DiagnosticResultsPage = () => {
    const [diagnosticResults, setDiagnosticResults] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [selectedStudy, setSelectedStudy] = useState(null);
    const [availableStudies, setAvailableStudies] = useState([]);

    useEffect(() => {
        loadAvailableStudies();
    }, []);

    useEffect(() => {
        if (selectedStudy) {
            loadDiagnosticResults(selectedStudy);
        }
    }, [selectedStudy]);

    const loadAvailableStudies = async () => {
        try {
            const response = await axios.get('/api/diagnostics/studies');
            if (response.data.success) {
                const studies = response.data.studies;
                setAvailableStudies(studies);

                // Auto-select the most recent study
                if (studies.length > 0) {
                    setSelectedStudy(studies[0].study_id);
                }
            }
        } catch (error) {
            console.error('Error loading studies:', error);
            setError('Failed to load available studies');
        } finally {
            setLoading(false);
        }
    };

    const loadDiagnosticResults = async (studyId) => {
        setLoading(true);
        setError(null);

        try {
            const response = await axios.get(`/api/diagnostics/results/${studyId}`);
            if (response.data.success) {
                setDiagnosticResults(response.data.results);
            } else {
                setError(response.data.error || 'Failed to load diagnostic results');
            }
        } catch (error) {
            console.error('Error loading diagnostic results:', error);
            setError(error.response?.data?.error || 'Failed to load diagnostic results');
        } finally {
            setLoading(false);
        }
    };

    const runDiagnosticAnalysis = async () => {
        if (!selectedStudy) return;

        setLoading(true);
        setError(null);

        try {
            const response = await axios.post(`/api/diagnostics/analyze/${selectedStudy}`);
            if (response.data.success) {
                // Reload results after analysis
                await loadDiagnosticResults(selectedStudy);
            } else {
                setError(response.data.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Error running analysis:', error);
            setError(error.response?.data?.error || 'Failed to run diagnostic analysis');
        } finally {
            setLoading(false);
        }
    };

    const getStudyStatusBadge = (status) => {
        const variants = {
            'processed': 'success',
            'analyzed': 'primary',
            'pending': 'warning',
            'error': 'danger'
        };
        return <Badge bg={variants[status] || 'secondary'}>{status}</Badge>;
    };

    if (loading && !diagnosticResults) {
        return (
            <div className="text-center py-5">
                <Spinner animation="border" role="status">
                    <span className="visually-hidden">Loading...</span>
                </Spinner>
                <p className="mt-2">Loading diagnostic results...</p>
            </div>
        );
    }

    return (
        <div>
            {/* Study Selection */}
            <Row className="mb-4">
                <Col md={12}>
                    <Card>
                        <Card.Header>
                            <Row className="align-items-center">
                                <Col>
                                    <h5 className="mb-0">Diagnostic Results Dashboard</h5>
                                </Col>
                                <Col xs="auto">
                                    {selectedStudy && (
                                        <Button
                                            variant="primary"
                                            onClick={runDiagnosticAnalysis}
                                            disabled={loading}
                                        >
                                            {loading ? 'Analyzing...' : 'Run Analysis'}
                                        </Button>
                                    )}
                                </Col>
                            </Row>
                        </Card.Header>
                        <Card.Body>
                            {availableStudies.length === 0 ? (
                                <Alert variant="info">
                                    No studies available. Please upload medical images first.
                                </Alert>
                            ) : (
                                <div>
                                    <h6>Available Studies:</h6>
                                    <div className="d-flex flex-wrap gap-2">
                                        {availableStudies.map(study => (
                                            <Button
                                                key={study.study_id}
                                                variant={selectedStudy === study.study_id ? 'primary' : 'outline-primary'}
                                                size="sm"
                                                onClick={() => setSelectedStudy(study.study_id)}
                                                className="d-flex align-items-center gap-2"
                                            >
                                                <span>{study.study_id}</span>
                                                {getStudyStatusBadge(study.status)}
                                            </Button>
                                        ))}
                                    </div>

                                    {selectedStudy && (
                                        <div className="mt-3">
                                            <small className="text-muted">
                                                Selected Study: <strong>{selectedStudy}</strong>
                                            </small>
                                        </div>
                                    )}
                                </div>
                            )}
                        </Card.Body>
                    </Card>
                </Col>
            </Row>

            {error && (
                <Row className="mb-4">
                    <Col md={12}>
                        <Alert variant="danger">
                            {error}
                        </Alert>
                    </Col>
                </Row>
            )}

            {diagnosticResults ? (
                <div>
                    {/* Classification Results */}
                    <Row className="mb-4">
                        <Col md={12}>
                            <ClassificationResults
                                classificationData={diagnosticResults.classification}
                                confidenceScores={diagnosticResults.confidence_scores}
                            />
                        </Col>
                    </Row>

                    {/* Segmentation Results */}
                    {diagnosticResults.segmentation && (
                        <Row className="mb-4">
                            <Col md={12}>
                                <SegmentationViewer
                                    segmentationData={diagnosticResults.segmentation}
                                    originalImage={diagnosticResults.original_image}
                                    onSegmentationChange={(slice, label) => {
                                        console.log(`Segmentation view changed: slice ${slice}, label ${label}`);
                                    }}
                                />
                            </Col>
                        </Row>
                    )}

                    {/* Explainability Visualization */}
                    {diagnosticResults.explainability && (
                        <Row className="mb-4">
                            <Col md={12}>
                                <ExplainabilityVisualization
                                    explainabilityData={diagnosticResults.explainability}
                                    originalImage={diagnosticResults.original_image}
                                />
                            </Col>
                        </Row>
                    )}

                    {/* Model Metrics */}
                    {diagnosticResults.metrics && (
                        <Row className="mb-4">
                            <Col md={12}>
                                <Card>
                                    <Card.Header>
                                        <h6 className="mb-0">Model Performance Metrics</h6>
                                    </Card.Header>
                                    <Card.Body>
                                        <Row>
                                            {Object.entries(diagnosticResults.metrics).map(([metric, value]) => (
                                                <Col md={3} key={metric} className="text-center mb-3">
                                                    <div className="border rounded p-3">
                                                        <h5 className="mb-1">
                                                            {typeof value === 'number' ? value.toFixed(3) : value}
                                                        </h5>
                                                        <small className="text-muted text-uppercase">
                                                            {metric.replace('_', ' ')}
                                                        </small>
                                                    </div>
                                                </Col>
                                            ))}
                                        </Row>
                                    </Card.Body>
                                </Card>
                            </Col>
                        </Row>
                    )}

                    {/* Analysis Summary */}
                    <Row>
                        <Col md={12}>
                            <Card>
                                <Card.Header>
                                    <h6 className="mb-0">Analysis Summary</h6>
                                </Card.Header>
                                <Card.Body>
                                    <div className="mb-3">
                                        <strong>Study ID:</strong> {diagnosticResults.study_id}
                                    </div>
                                    <div className="mb-3">
                                        <strong>Analysis Date:</strong> {new Date(diagnosticResults.timestamp).toLocaleString()}
                                    </div>
                                    <div className="mb-3">
                                        <strong>Processing Time:</strong> {diagnosticResults.processing_time}s
                                    </div>
                                    {diagnosticResults.model_version && (
                                        <div className="mb-3">
                                            <strong>Model Version:</strong> {diagnosticResults.model_version}
                                        </div>
                                    )}

                                    <Alert variant="info" className="mt-3">
                                        <small>
                                            <strong>Disclaimer:</strong> These results are generated by an AI model and should be
                                            reviewed by qualified medical professionals. This system is intended for research
                                            and educational purposes and should not be used as the sole basis for clinical decisions.
                                        </small>
                                    </Alert>
                                </Card.Body>
                            </Card>
                        </Col>
                    </Row>
                </div>
            ) : (
                !loading && selectedStudy && (
                    <Row>
                        <Col md={12}>
                            <Alert variant="info" className="text-center">
                                <h6>No diagnostic results available for this study</h6>
                                <p>Click "Run Analysis" to generate diagnostic results for the selected study.</p>
                            </Alert>
                        </Col>
                    </Row>
                )
            )}
        </div>
    );
};

export default DiagnosticResultsPage;