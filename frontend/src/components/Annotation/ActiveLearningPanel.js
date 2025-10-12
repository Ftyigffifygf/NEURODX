import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Button, Alert, Badge, ListGroup, ProgressBar } from 'react-bootstrap';
import axios from 'axios';

const ActiveLearningPanel = ({ onSampleSelected, onAnnotationComplete }) => {
    const [suggestedSamples, setSuggestedSamples] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [selectedSample, setSelectedSample] = useState(null);
    const [annotationProgress, setAnnotationProgress] = useState({});

    useEffect(() => {
        loadSuggestedSamples();
    }, []);

    const loadSuggestedSamples = async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await axios.get('/api/monai-label/active-learning/suggestions');
            if (response.data.success) {
                setSuggestedSamples(response.data.samples);
            } else {
                setError(response.data.error || 'Failed to load suggested samples');
            }
        } catch (error) {
            console.error('Error loading suggested samples:', error);
            setError(error.response?.data?.error || 'Failed to load suggested samples');
        } finally {
            setLoading(false);
        }
    };

    const handleSampleSelection = async (sample) => {
        setSelectedSample(sample);

        try {
            // Load the sample data for annotation
            const response = await axios.get(`/api/monai-label/samples/${sample.sample_id}`);
            if (response.data.success) {
                if (onSampleSelected) {
                    onSampleSelected(response.data.sample_data);
                }
            }
        } catch (error) {
            console.error('Error loading sample data:', error);
            setError('Failed to load sample data');
        }
    };

    const submitAnnotation = async (annotationData) => {
        if (!selectedSample) return;

        setLoading(true);

        try {
            const response = await axios.post(`/api/monai-label/annotations/${selectedSample.sample_id}`, {
                annotation_data: annotationData,
                annotation_type: 'segmentation',
                quality_score: calculateAnnotationQuality(annotationData)
            });

            if (response.data.success) {
                // Update annotation progress
                setAnnotationProgress(prev => ({
                    ...prev,
                    [selectedSample.sample_id]: 'completed'
                }));

                // Remove from suggested samples
                setSuggestedSamples(prev =>
                    prev.filter(s => s.sample_id !== selectedSample.sample_id)
                );

                if (onAnnotationComplete) {
                    onAnnotationComplete(selectedSample.sample_id, annotationData);
                }

                // Load new suggestions
                await loadSuggestedSamples();
            } else {
                setError(response.data.error || 'Failed to submit annotation');
            }
        } catch (error) {
            console.error('Error submitting annotation:', error)
            setError(error.response?.data?.error || 'Failed to submit annotation');
        } finally {
            setLoading(false);
        }
    };

    const calculateAnnotationQuality = (annotationData) => {
        // Simple quality metric based on annotation coverage and consistency
        if (!annotationData || !annotationData.slices) return 0;

        let totalVoxels = 0;
        let annotatedVoxels = 0;

        annotationData.slices.forEach(slice => {
            if (slice && slice.data) {
                totalVoxels += slice.data.length;
                annotatedVoxels += slice.data.filter(voxel => voxel > 0).length;
            }
        });

        const coverage = totalVoxels > 0 ? annotatedVoxels / totalVoxels : 0;
        return Math.min(1.0, coverage * 2); // Scale to 0-1 range
    };

    const skipSample = async (sample) => {
        try {
            await axios.post(`/api/monai-label/active-learning/skip/${sample.sample_id}`);
            setSuggestedSamples(prev =>
                prev.filter(s => s.sample_id !== sample.sample_id)
            );
        } catch (error) {
            console.error('Error skipping sample:', error);
        }
    };

    const getUncertaintyColor = (uncertainty) => {
        if (uncertainty >= 0.8) return 'danger';
        if (uncertainty >= 0.6) return 'warning';
        return 'success';
    };

    const getPriorityBadge = (priority) => {
        const variants = {
            'high': 'danger',
            'medium': 'warning',
            'low': 'success'
        };
        return <Badge bg={variants[priority] || 'secondary'}>{priority}</Badge>;
    };

    return (
        <Card>
            <Card.Header>
                <Row className="align-items-center">
                    <Col>
                        <h6 className="mb-0">Active Learning Suggestions</h6>
                    </Col>
                    <Col xs="auto">
                        <Button
                            size="sm"
                            variant="outline-primary"
                            onClick={loadSuggestedSamples}
                            disabled={loading}
                        >
                            Refresh
                        </Button>
                    </Col>
                </Row>
            </Card.Header>

            <Card.Body>
                {error && (
                    <Alert variant="danger" className="mb-3">
                        {error}
                    </Alert>
                )}

                {loading && suggestedSamples.length === 0 ? (
                    <div className="text-center py-3">
                        <div className="spinner-border spinner-border-sm me-2" role="status"></div>
                        Loading suggestions...
                    </div>
                ) : suggestedSamples.length === 0 ? (
                    <Alert variant="info">
                        No samples suggested for annotation at this time.
                        The active learning engine will suggest new samples as the model improves.
                    </Alert>
                ) : (
                    <div>
                        <div className="mb-3">
                            <small className="text-muted">
                                {suggestedSamples.length} samples suggested for annotation based on model uncertainty
                            </small>
                        </div>

                        <ListGroup>
                            {suggestedSamples.map((sample, index) => (
                                <ListGroup.Item
                                    key={sample.sample_id}
                                    className={`d-flex justify-content-between align-items-center ${selectedSample?.sample_id === sample.sample_id ? 'active' : ''
                                        }`}
                                >
                                    <div className="flex-grow-1">
                                        <div className="d-flex align-items-center mb-2">
                                            <strong className="me-2">Sample {index + 1}</strong>
                                            {getPriorityBadge(sample.priority)}
                                        </div>

                                        <div className="mb-2">
                                            <small className="text-muted">
                                                <strong>Study:</strong> {sample.study_id}<br />
                                                <strong>Modality:</strong> {sample.modality}<br />
                                                <strong>Reason:</strong> {sample.suggestion_reason}
                                            </small>
                                        </div>

                                        <div className="mb-2">
                                            <div className="d-flex justify-content-between align-items-center mb-1">
                                                <small>Model Uncertainty</small>
                                                <small>{(sample.uncertainty_score * 100).toFixed(1)}%</small>
                                            </div>
                                            <ProgressBar
                                                now={sample.uncertainty_score * 100}
                                                variant={getUncertaintyColor(sample.uncertainty_score)}
                                                size="sm"
                                            />
                                        </div>

                                        {sample.predicted_labels && (
                                            <div className="mb-2">
                                                <small className="text-muted">
                                                    <strong>Predicted:</strong> {sample.predicted_labels.join(', ')}
                                                </small>
                                            </div>
                                        )}
                                    </div>

                                    <div className="ms-3">
                                        <div className="d-grid gap-1">
                                            <Button
                                                size="sm"
                                                variant={selectedSample?.sample_id === sample.sample_id ? 'success' : 'primary'}
                                                onClick={() => handleSampleSelection(sample)}
                                                disabled={loading}
                                            >
                                                {selectedSample?.sample_id === sample.sample_id ? 'Selected' : 'Annotate'}
                                            </Button>

                                            <Button
                                                size="sm"
                                                variant="outline-secondary"
                                                onClick={() => skipSample(sample)}
                                                disabled={loading}
                                            >
                                                Skip
                                            </Button>
                                        </div>
                                    </div>
                                </ListGroup.Item>
                            ))}
                        </ListGroup>
                    </div>
                )}

                {selectedSample && (
                    <Alert variant="success" className="mt-3">
                        <small>
                            <strong>Selected Sample:</strong> {selectedSample.sample_id}
                            <br />
                            Use the annotation canvas to create segmentation labels.
                            Click "Submit Annotation" when complete.
                        </small>
                    </Alert>
                )}

                <div className="mt-3">
                    <Card className="bg-light">
                        <Card.Body className="py-2">
                            <small className="text-muted">
                                <strong>Active Learning Tips:</strong>
                                <ul className="mb-0 mt-1">
                                    <li>High uncertainty samples help improve model performance most</li>
                                    <li>Focus on clear, accurate annotations for suggested regions</li>
                                    <li>Skip samples that are too difficult or ambiguous to annotate</li>
                                    <li>Quality annotations are more valuable than quantity</li>
                                </ul>
                            </small>
                        </Card.Body>
                    </Card>
                </div>
            </Card.Body>
        </Card>
    );
};

export default ActiveLearningPanel;