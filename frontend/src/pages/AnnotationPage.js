import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Button, ButtonGroup, Alert, Modal } from 'react-bootstrap';
import AnnotationCanvas from '../components/Annotation/AnnotationCanvas';
import ActiveLearningPanel from '../components/Annotation/ActiveLearningPanel';
import AnnotationQualityFeedback from '../components/Annotation/AnnotationQualityFeedback';
import axios from 'axios';

const AnnotationPage = () => {
    const [imageData, setImageData] = useState(null);
    const [annotations, setAnnotations] = useState(null);
    const [annotationMode, setAnnotationMode] = useState('draw');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [showQualityModal, setShowQualityModal] = useState(false);
    const [currentSample, setCurrentSample] = useState(null);
    const [annotationHistory, setAnnotationHistory] = useState([]);
    const [canUndo, setCanUndo] = useState(false);

    const handleSampleSelected = (sampleData) => {
        setImageData(sampleData.image_data);
        setCurrentSample(sampleData);

        // Initialize empty annotations if none exist
        if (sampleData.existing_annotations) {
            setAnnotations(sampleData.existing_annotations);
        } else {
            initializeEmptyAnnotations(sampleData.image_data);
        }

        setError(null);
    };

    const initializeEmptyAnnotations = (imageData) => {
        if (!imageData || !imageData.slices) return;

        const emptyAnnotations = {
            slices: imageData.slices.map(slice => ({
                data: new Array(slice.width * slice.height).fill(0),
                width: slice.width,
                height: slice.height
            }))
        };

        setAnnotations(emptyAnnotations);
    };

    const handleAnnotationChange = (newAnnotations) => {
        // Save current state to history for undo functionality
        if (annotations) {
            setAnnotationHistory(prev => [...prev.slice(-9), annotations]); // Keep last 10 states
            setCanUndo(true);
        }

        setAnnotations(newAnnotations);
    };

    const handleUndo = () => {
        if (annotationHistory.length > 0) {
            const previousState = annotationHistory[annotationHistory.length - 1];
            setAnnotations(previousState);
            setAnnotationHistory(prev => prev.slice(0, -1));
            setCanUndo(annotationHistory.length > 1);
        }
    };

    const clearAnnotations = () => {
        if (imageData) {
            initializeEmptyAnnotations(imageData);
            setAnnotationHistory([]);
            setCanUndo(false);
        }
    };

    const submitAnnotation = async () => {
        if (!currentSample || !annotations) {
            setError('No annotation data to submit');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await axios.post(`/api/monai-label/annotations/${currentSample.sample_id}`, {
                annotation_data: annotations,
                annotation_type: 'segmentation',
                quality_metrics: calculateQualityMetrics(annotations)
            });

            if (response.data.success) {
                setShowQualityModal(true);

                // Reset for next annotation
                setCurrentSample(null);
                setImageData(null);
                setAnnotations(null);
                setAnnotationHistory([]);
                setCanUndo(false);
            } else {
                setError(response.data.error || 'Failed to submit annotation');
            }
        } catch (error) {
            console.error('Error submitting annotation:', error);
            setError(error.response?.data?.error || 'Failed to submit annotation');
        } finally {
            setLoading(false);
        }
    };

    const calculateQualityMetrics = (annotationData) => {
        if (!annotationData || !annotationData.slices) return {};

        let totalVoxels = 0;
        let annotatedVoxels = 0;

        annotationData.slices.forEach(slice => {
            if (slice && slice.data) {
                totalVoxels += slice.data.length;
                annotatedVoxels += slice.data.filter(voxel => voxel > 0).length;
            }
        });

        return {
            coverage: totalVoxels > 0 ? annotatedVoxels / totalVoxels : 0,
            annotated_voxels: annotatedVoxels,
            total_voxels: totalVoxels
        };
    };

    const handleQualityImprovement = (action) => {
        switch (action) {
            case 'increase_coverage':
                setAnnotationMode('draw');
                setError(null);
                break;
            case 'improve_boundaries':
                // Could suggest smaller brush size or different tools
                setError('Try using a smaller brush size for more precise boundaries');
                break;
            case 'improve_consistency':
                setError('Use consistent labels for similar structures across all slices');
                break;
            default:
                break;
        }
    };

    const exportAnnotations = async () => {
        if (!annotations || !currentSample) return;

        try {
            const response = await axios.post('/api/monai-label/export-annotations', {
                sample_id: currentSample.sample_id,
                annotation_data: annotations,
                format: 'nifti'
            });

            if (response.data.success) {
                // Trigger download
                const link = document.createElement('a');
                link.href = response.data.download_url;
                link.download = `annotation_${currentSample.sample_id}.nii.gz`;
                link.click();
            }
        } catch (error) {
            console.error('Error exporting annotations:', error);
            setError('Failed to export annotations');
        }
    };

    return (
        <div>
            {/* Annotation Tools Header */}
            <Row className="mb-4">
                <Col md={12}>
                    <Card>
                        <Card.Header>
                            <Row className="align-items-center">
                                <Col>
                                    <h5 className="mb-0">MONAI Label Annotation Interface</h5>
                                </Col>
                                <Col xs="auto">
                                    <ButtonGroup>
                                        <Button
                                            variant={annotationMode === 'draw' ? 'primary' : 'outline-primary'}
                                            onClick={() => setAnnotationMode('draw')}
                                        >
                                            Draw
                                        </Button>
                                        <Button
                                            variant={annotationMode === 'erase' ? 'primary' : 'outline-primary'}
                                            onClick={() => setAnnotationMode('erase')}
                                        >
                                            Erase
                                        </Button>
                                        <Button
                                            variant={annotationMode === 'pan' ? 'primary' : 'outline-primary'}
                                            onClick={() => setAnnotationMode('pan')}
                                        >
                                            Pan
                                        </Button>
                                    </ButtonGroup>
                                </Col>
                            </Row>
                        </Card.Header>

                        <Card.Body>
                            <Row className="align-items-center">
                                <Col>
                                    {currentSample ? (
                                        <small className="text-muted">
                                            Current Sample: <strong>{currentSample.sample_id}</strong>
                                            {currentSample.modality && ` (${currentSample.modality})`}
                                        </small>
                                    ) : (
                                        <small className="text-muted">
                                            Select a sample from the Active Learning panel to begin annotation
                                        </small>
                                    )}
                                </Col>

                                <Col xs="auto">
                                    <ButtonGroup size="sm">
                                        <Button
                                            variant="outline-secondary"
                                            onClick={handleUndo}
                                            disabled={!canUndo}
                                        >
                                            Undo
                                        </Button>
                                        <Button
                                            variant="outline-warning"
                                            onClick={clearAnnotations}
                                            disabled={!annotations}
                                        >
                                            Clear
                                        </Button>
                                        <Button
                                            variant="outline-info"
                                            onClick={exportAnnotations}
                                            disabled={!annotations || !currentSample}
                                        >
                                            Export
                                        </Button>
                                        <Button
                                            variant="success"
                                            onClick={submitAnnotation}
                                            disabled={!annotations || !currentSample || loading}
                                        >
                                            {loading ? 'Submitting...' : 'Submit Annotation'}
                                        </Button>
                                    </ButtonGroup>
                                </Col>
                            </Row>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>

            {error && (
                <Row className="mb-4">
                    <Col md={12}>
                        <Alert variant="danger" dismissible onClose={() => setError(null)}>
                            {error}
                        </Alert>
                    </Col>
                </Row>
            )}

            <Row>
                {/* Active Learning Panel */}
                <Col md={4}>
                    <ActiveLearningPanel
                        onSampleSelected={handleSampleSelected}
                        onAnnotationComplete={(sampleId, annotationData) => {
                            console.log(`Annotation completed for sample ${sampleId}`);
                        }}
                    />
                </Col>

                {/* Annotation Canvas */}
                <Col md={8}>
                    {imageData ? (
                        <AnnotationCanvas
                            imageData={imageData}
                            annotations={annotations}
                            onAnnotationChange={handleAnnotationChange}
                            annotationMode={annotationMode}
                        />
                    ) : (
                        <Card>
                            <Card.Body className="text-center py-5">
                                <h6 className="text-muted">No Image Selected</h6>
                                <p className="text-muted">
                                    Select a sample from the Active Learning panel to begin annotation
                                </p>
                            </Card.Body>
                        </Card>
                    )}
                </Col>
            </Row>

            {/* Quality Feedback */}
            {annotations && (
                <Row className="mt-4">
                    <Col md={12}>
                        <Card>
                            <Card.Header>
                                <h6 className="mb-0">Annotation Quality Feedback</h6>
                            </Card.Header>
                            <Card.Body>
                                <AnnotationQualityFeedback
                                    annotationData={annotations}
                                    onQualityImprovement={handleQualityImprovement}
                                />
                            </Card.Body>
                        </Card>
                    </Col>
                </Row>
            )}

            {/* Success Modal */}
            <Modal show={showQualityModal} onHide={() => setShowQualityModal(false)}>
                <Modal.Header closeButton>
                    <Modal.Title>Annotation Submitted Successfully</Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    <Alert variant="success">
                        Your annotation has been submitted and will be used to improve the model.
                        Thank you for contributing to the active learning process!
                    </Alert>

                    <div className="text-center">
                        <p>The model will be retrained with your annotation and new suggestions will be generated.</p>
                    </div>
                </Modal.Body>
                <Modal.Footer>
                    <Button variant="primary" onClick={() => setShowQualityModal(false)}>
                        Continue Annotating
                    </Button>
                </Modal.Footer>
            </Modal>
        </div>
    );
};

export default AnnotationPage;