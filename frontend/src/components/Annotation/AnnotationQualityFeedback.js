import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Alert, Badge, ProgressBar, Button, Form } from 'react-bootstrap';
import Plot from 'react-plotly.js';

const AnnotationQualityFeedback = ({ annotationData, onQualityImprovement }) => {
    const [qualityMetrics, setQualityMetrics] = useState(null);
    const [suggestions, setSuggestions] = useState([]);
    const [showDetailedFeedback, setShowDetailedFeedback] = useState(false);

    useEffect(() => {
        if (annotationData) {
            calculateQualityMetrics();
            generateSuggestions();
        }
    }, [annotationData]);

    const calculateQualityMetrics = () => {
        if (!annotationData || !annotationData.slices) return;

        let totalVoxels = 0;
        let annotatedVoxels = 0;
        let labelConsistency = 0;
        let boundaryQuality = 0;
        let slicesWithAnnotations = 0;

        annotationData.slices.forEach((slice, sliceIndex) => {
            if (!slice || !slice.data) return;

            const sliceVoxels = slice.data.length;
            const sliceAnnotated = slice.data.filter(voxel => voxel > 0).length;

            totalVoxels += sliceVoxels;
            annotatedVoxels += sliceAnnotated;

            if (sliceAnnotated > 0) {
                slicesWithAnnotations++;

                // Calculate label consistency (fewer different labels = more consistent)
                const uniqueLabels = new Set(slice.data.filter(v => v > 0));
                labelConsistency += 1 / Math.max(1, uniqueLabels.size);

                // Calculate boundary quality (simplified edge detection)
                boundaryQuality += calculateBoundaryQuality(slice.data, slice.width, slice.height);
            }
        });

        const coverage = totalVoxels > 0 ? annotatedVoxels / totalVoxels : 0;
        const consistency = slicesWithAnnotations > 0 ? labelConsistency / slicesWithAnnotations : 0;
        const avgBoundaryQuality = slicesWithAnnotations > 0 ? boundaryQuality / slicesWithAnnotations : 0;
        const completeness = slicesWithAnnotations / annotationData.slices.length;

        const overallScore = (coverage * 0.3 + consistency * 0.2 + avgBoundaryQuality * 0.3 + completeness * 0.2);

        setQualityMetrics({
            overall_score: overallScore,
            coverage: coverage,
            consistency: consistency,
            boundary_quality: avgBoundaryQuality,
            completeness: completeness,
            annotated_slices: slicesWithAnnotations,
            total_slices: annotationData.slices.length,
            annotated_voxels: annotatedVoxels,
            total_voxels: totalVoxels
        });
    };

    const calculateBoundaryQuality = (data, width, height) => {
        let boundaryPixels = 0;
        let smoothBoundaryPixels = 0;

        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = y * width + x;
                const current = data[idx];

                if (current > 0) {
                    // Check if this is a boundary pixel
                    const neighbors = [
                        data[(y - 1) * width + x],     // top
                        data[(y + 1) * width + x],     // bottom
                        data[y * width + (x - 1)],     // left
                        data[y * width + (x + 1)]      // right
                    ];

                    const isBoundary = neighbors.some(n => n !== current);

                    if (isBoundary) {
                        boundaryPixels++;

                        // Check if boundary is smooth (similar to neighbors)
                        const similarNeighbors = neighbors.filter(n => n === current || n === 0).length;
                        if (similarNeighbors >= 3) {
                            smoothBoundaryPixels++;
                        }
                    }
                }
            }
        }

        return boundaryPixels > 0 ? smoothBoundaryPixels / boundaryPixels : 1;
    };

    const generateSuggestions = () => {
        if (!qualityMetrics) return;

        const newSuggestions = [];

        if (qualityMetrics.coverage < 0.1) {
            newSuggestions.push({
                type: 'warning',
                title: 'Low Annotation Coverage',
                message: 'Consider annotating more regions to improve model training.',
                action: 'increase_coverage'
            });
        }

        if (qualityMetrics.consistency < 0.7) {
            newSuggestions.push({
                type: 'info',
                title: 'Label Consistency',
                message: 'Try to use consistent labels for similar structures across slices.',
                action: 'improve_consistency'
            });
        }

        if (qualityMetrics.boundary_quality < 0.6) {
            newSuggestions.push({
                type: 'warning',
                title: 'Boundary Quality',
                message: 'Annotation boundaries could be smoother. Use smaller brush sizes for detailed areas.',
                action: 'improve_boundaries'
            });
        }

        if (qualityMetrics.completeness < 0.5) {
            newSuggestions.push({
                type: 'info',
                title: 'Annotation Completeness',
                message: 'Consider annotating more slices for better 3D consistency.',
                action: 'increase_completeness'
            });
        }

        if (qualityMetrics.overall_score >= 0.8) {
            newSuggestions.push({
                type: 'success',
                title: 'Excellent Annotation Quality',
                message: 'Your annotation meets high quality standards!',
                action: 'maintain_quality'
            });
        }

        setSuggestions(newSuggestions);
    };

    const getScoreColor = (score) => {
        if (score >= 0.8) return 'success';
        if (score >= 0.6) return 'warning';
        return 'danger';
    };

    const getScoreLabel = (score) => {
        if (score >= 0.8) return 'Excellent';
        if (score >= 0.6) return 'Good';
        if (score >= 0.4) return 'Fair';
        return 'Needs Improvement';
    };

    const generateQualityChart = () => {
        if (!qualityMetrics) return null;

        const metrics = [
            { name: 'Coverage', value: qualityMetrics.coverage },
            { name: 'Consistency', value: qualityMetrics.consistency },
            { name: 'Boundary Quality', value: qualityMetrics.boundary_quality },
            { name: 'Completeness', value: qualityMetrics.completeness }
        ];

        const data = [{
            type: 'scatterpolar',
            r: metrics.map(m => m.value),
            theta: metrics.map(m => m.name),
            fill: 'toself',
            name: 'Quality Metrics'
        }];

        const layout = {
            polar: {
                radialaxis: {
                    visible: true,
                    range: [0, 1]
                }
            },
            showlegend: false,
            height: 300,
            margin: { t: 50, b: 50, l: 50, r: 50 }
        };

        return (
            <Plot
                data={data}
                layout={layout}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
            />
        );
    };

    if (!qualityMetrics) {
        return (
            <Alert variant="info">
                Create annotations to see quality feedback and suggestions.
            </Alert>
        );
    }

    return (
        <div>
            <Row>
                <Col md={6}>
                    <Card className="mb-3">
                        <Card.Header>
                            <h6 className="mb-0">Overall Quality Score</h6>
                        </Card.Header>
                        <Card.Body className="text-center">
                            <div className="mb-3">
                                <h2 className={`text-${getScoreColor(qualityMetrics.overall_score)}`}>
                                    {(qualityMetrics.overall_score * 100).toFixed(0)}%
                                </h2>
                                <Badge bg={getScoreColor(qualityMetrics.overall_score)} className="fs-6">
                                    {getScoreLabel(qualityMetrics.overall_score)}
                                </Badge>
                            </div>

                            <div className="text-start">
                                <small className="text-muted">
                                    <strong>Annotated Voxels:</strong> {qualityMetrics.annotated_voxels.toLocaleString()}<br />
                                    <strong>Annotated Slices:</strong> {qualityMetrics.annotated_slices} / {qualityMetrics.total_slices}
                                </small>
                            </div>
                        </Card.Body>
                    </Card>
                </Col>

                <Col md={6}>
                    <Card className="mb-3">
                        <Card.Header>
                            <h6 className="mb-0">Quality Breakdown</h6>
                        </Card.Header>
                        <Card.Body>
                            {[
                                { name: 'Coverage', value: qualityMetrics.coverage, description: 'Portion of image annotated' },
                                { name: 'Consistency', value: qualityMetrics.consistency, description: 'Label usage consistency' },
                                { name: 'Boundary Quality', value: qualityMetrics.boundary_quality, description: 'Smoothness of boundaries' },
                                { name: 'Completeness', value: qualityMetrics.completeness, description: 'Slices with annotations' }
                            ].map(metric => (
                                <div key={metric.name} className="mb-3">
                                    <div className="d-flex justify-content-between mb-1">
                                        <small>
                                            <strong>{metric.name}</strong>
                                            <br />
                                            <span className="text-muted">{metric.description}</span>
                                        </small>
                                        <small>{(metric.value * 100).toFixed(0)}%</small>
                                    </div>
                                    <ProgressBar
                                        now={metric.value * 100}
                                        variant={getScoreColor(metric.value)}
                                    />
                                </div>
                            ))}
                        </Card.Body>
                    </Card>
                </Col>
            </Row>

            <Row>
                <Col md={6}>
                    <Card className="mb-3">
                        <Card.Header>
                            <h6 className="mb-0">Quality Radar Chart</h6>
                        </Card.Header>
                        <Card.Body>
                            {generateQualityChart()}
                        </Card.Body>
                    </Card>
                </Col>

                <Col md={6}>
                    <Card className="mb-3">
                        <Card.Header>
                            <Row className="align-items-center">
                                <Col>
                                    <h6 className="mb-0">Improvement Suggestions</h6>
                                </Col>
                                <Col xs="auto">
                                    <Form.Check
                                        type="switch"
                                        label="Detailed"
                                        checked={showDetailedFeedback}
                                        onChange={(e) => setShowDetailedFeedback(e.target.checked)}
                                    />
                                </Col>
                            </Row>
                        </Card.Header>
                        <Card.Body>
                            {suggestions.length === 0 ? (
                                <Alert variant="success">
                                    No specific suggestions - your annotation quality looks good!
                                </Alert>
                            ) : (
                                suggestions.map((suggestion, index) => (
                                    <Alert key={index} variant={suggestion.type} className="mb-2">
                                        <strong>{suggestion.title}</strong>
                                        <br />
                                        <small>{suggestion.message}</small>

                                        {showDetailedFeedback && suggestion.action && (
                                            <div className="mt-2">
                                                <Button
                                                    size="sm"
                                                    variant={`outline-${suggestion.type}`}
                                                    onClick={() => onQualityImprovement && onQualityImprovement(suggestion.action)}
                                                >
                                                    Apply Suggestion
                                                </Button>
                                            </div>
                                        )}
                                    </Alert>
                                ))
                            )}
                        </Card.Body>
                    </Card>
                </Col>
            </Row>
        </div>
    );
};

export default AnnotationQualityFeedback;