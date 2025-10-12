import React from 'react';
import { Card, Row, Col, ProgressBar, Badge, Alert } from 'react-bootstrap';
import Plot from 'react-plotly.js';

const ClassificationResults = ({ classificationData, confidenceScores }) => {
    if (!classificationData) {
        return (
            <Alert variant="info">
                No classification results available
            </Alert>
        );
    }

    const { probabilities, predicted_class, risk_assessment } = classificationData;

    const getConfidenceColor = (confidence) => {
        if (confidence >= 0.8) return 'success';
        if (confidence >= 0.6) return 'warning';
        return 'danger';
    };

    const getRiskColor = (risk) => {
        switch (risk?.toLowerCase()) {
            case 'low': return 'success';
            case 'moderate': return 'warning';
            case 'high': return 'danger';
            default: return 'secondary';
        }
    };

    // Prepare data for probability chart
    const chartData = [{
        x: Object.keys(probabilities),
        y: Object.values(probabilities),
        type: 'bar',
        marker: {
            color: Object.values(probabilities).map(prob =>
                prob >= 0.5 ? '#28a745' : '#6c757d'
            )
        },
        text: Object.values(probabilities).map(prob => `${(prob * 100).toFixed(1)}%`),
        textposition: 'auto'
    }];

    const chartLayout = {
        title: 'Classification Probabilities',
        xaxis: { title: 'Disease Classes' },
        yaxis: { title: 'Probability', range: [0, 1] },
        margin: { t: 50, b: 100, l: 50, r: 50 },
        height: 300
    };

    return (
        <div>
            <Row>
                <Col md={6}>
                    <Card className="mb-3">
                        <Card.Header>
                            <h6 className="mb-0">Primary Classification</h6>
                        </Card.Header>
                        <Card.Body>
                            <div className="text-center mb-3">
                                <h4 className="mb-2">{predicted_class}</h4>
                                <Badge
                                    bg={getConfidenceColor(probabilities[predicted_class])}
                                    className="fs-6"
                                >
                                    {(probabilities[predicted_class] * 100).toFixed(1)}% Confidence
                                </Badge>
                            </div>

                            {risk_assessment && (
                                <div className="text-center">
                                    <small className="text-muted">Risk Assessment:</small>
                                    <br />
                                    <Badge bg={getRiskColor(risk_assessment.level)} className="mt-1">
                                        {risk_assessment.level} Risk
                                    </Badge>
                                    {risk_assessment.score && (
                                        <div className="mt-2">
                                            <small>Risk Score: {risk_assessment.score.toFixed(3)}</small>
                                        </div>
                                    )}
                                </div>
                            )}
                        </Card.Body>
                    </Card>
                </Col>

                <Col md={6}>
                    <Card className="mb-3">
                        <Card.Header>
                            <h6 className="mb-0">All Class Probabilities</h6>
                        </Card.Header>
                        <Card.Body>
                            {Object.entries(probabilities).map(([className, probability]) => (
                                <div key={className} className="mb-3">
                                    <div className="d-flex justify-content-between mb-1">
                                        <small>{className}</small>
                                        <small>{(probability * 100).toFixed(1)}%</small>
                                    </div>
                                    <ProgressBar
                                        now={probability * 100}
                                        variant={probability >= 0.5 ? 'success' : 'secondary'}
                                    />
                                </div>
                            ))}
                        </Card.Body>
                    </Card>
                </Col>
            </Row>

            <Row>
                <Col md={12}>
                    <Card className="mb-3">
                        <Card.Header>
                            <h6 className="mb-0">Probability Distribution</h6>
                        </Card.Header>
                        <Card.Body>
                            <Plot
                                data={chartData}
                                layout={chartLayout}
                                style={{ width: '100%' }}
                                config={{ displayModeBar: false }}
                            />
                        </Card.Body>
                    </Card>
                </Col>
            </Row>

            {confidenceScores && (
                <Row>
                    <Col md={12}>
                        <Card>
                            <Card.Header>
                                <h6 className="mb-0">Model Confidence Metrics</h6>
                            </Card.Header>
                            <Card.Body>
                                <Row>
                                    {Object.entries(confidenceScores).map(([metric, value]) => (
                                        <Col md={3} key={metric} className="text-center mb-3">
                                            <div className="border rounded p-3">
                                                <h5 className="mb-1">{value.toFixed(3)}</h5>
                                                <small className="text-muted text-uppercase">
                                                    {metric.replace('_', ' ')}
                                                </small>
                                            </div>
                                        </Col>
                                    ))}
                                </Row>

                                <Alert variant="info" className="mt-3">
                                    <small>
                                        <strong>Confidence Metrics:</strong>
                                        <ul className="mb-0 mt-2">
                                            <li><strong>Entropy:</strong> Lower values indicate higher confidence</li>
                                            <li><strong>Max Probability:</strong> Higher values indicate stronger predictions</li>
                                            <li><strong>Prediction Margin:</strong> Difference between top two predictions</li>
                                            <li><strong>Temperature Scaling:</strong> Calibrated confidence score</li>
                                        </ul>
                                    </small>
                                </Alert>
                            </Card.Body>
                        </Card>
                    </Col>
                </Row>
            )}
        </div>
    );
};

export default ClassificationResults;