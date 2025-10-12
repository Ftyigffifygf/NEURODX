import React, { useState, useRef, useEffect } from 'react';
import { Card, Row, Col, ButtonGroup, Button, Form, Alert } from 'react-bootstrap';
import Plot from 'react-plotly.js';

const ExplainabilityVisualization = ({ explainabilityData, originalImage }) => {
    const canvasRef = useRef(null);
    const [selectedMethod, setSelectedMethod] = useState('grad_cam');
    const [currentSlice, setCurrentSlice] = useState(0);
    const [overlayOpacity, setOverlayOpacity] = useState(0.6);
    const [colormap, setColormap] = useState('hot');

    useEffect(() => {
        if (explainabilityData && canvasRef.current) {
            renderExplainability();
        }
    }, [explainabilityData, selectedMethod, currentSlice, overlayOpacity, colormap]);

    const renderExplainability = () => {
        const canvas = canvasRef.current;
        if (!canvas || !explainabilityData) return;

        const ctx = canvas.getContext('2d');
        const methodData = explainabilityData[selectedMethod];

        if (!methodData || !methodData.slices) return;

        const slice = methodData.slices[currentSlice];
        if (!slice) return;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw original image as background
        if (originalImage && originalImage.slices[currentSlice]) {
            const originalSlice = originalImage.slices[currentSlice];
            const originalImageData = createGrayscaleImageData(
                originalSlice.data,
                originalSlice.width,
                originalSlice.height
            );

            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = originalSlice.width;
            tempCanvas.height = originalSlice.height;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.putImageData(originalImageData, 0, 0);

            ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
        }

        // Draw explainability heatmap overlay
        const heatmapImageData = createHeatmapImageData(
            slice.data,
            slice.width,
            slice.height,
            colormap
        );

        const heatmapCanvas = document.createElement('canvas');
        heatmapCanvas.width = slice.width;
        heatmapCanvas.height = slice.height;
        const heatmapCtx = heatmapCanvas.getContext('2d');
        heatmapCtx.putImageData(heatmapImageData, 0, 0);

        // Apply opacity and draw overlay
        ctx.globalAlpha = overlayOpacity;
        ctx.drawImage(heatmapCanvas, 0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 1.0;
    };

    const createGrayscaleImageData = (data, width, height) => {
        const imageData = new Uint8ClampedArray(width * height * 4);

        for (let i = 0; i < data.length; i++) {
            const value = Math.min(255, Math.max(0, data[i]));
            const idx = i * 4;
            imageData[idx] = value;     // R
            imageData[idx + 1] = value; // G
            imageData[idx + 2] = value; // B
            imageData[idx + 3] = 255;   // A
        }

        return new ImageData(imageData, width, height);
    };

    const createHeatmapImageData = (data, width, height, colormapType) => {
        const imageData = new Uint8ClampedArray(width * height * 4);

        // Normalize data to 0-1 range
        const minVal = Math.min(...data);
        const maxVal = Math.max(...data);
        const range = maxVal - minVal;

        for (let i = 0; i < data.length; i++) {
            const normalizedValue = range > 0 ? (data[i] - minVal) / range : 0;
            const color = getColormapColor(normalizedValue, colormapType);

            const idx = i * 4;
            imageData[idx] = color[0];     // R
            imageData[idx + 1] = color[1]; // G
            imageData[idx + 2] = color[2]; // B
            imageData[idx + 3] = normalizedValue > 0.1 ? 255 : 0; // A (transparent for low values)
        }

        return new ImageData(imageData, width, height);
    };

    const getColormapColor = (value, colormapType) => {
        // Clamp value to 0-1 range
        value = Math.max(0, Math.min(1, value));

        switch (colormapType) {
            case 'hot':
                if (value < 0.33) {
                    return [Math.floor(value * 3 * 255), 0, 0];
                } else if (value < 0.66) {
                    return [255, Math.floor((value - 0.33) * 3 * 255), 0];
                } else {
                    return [255, 255, Math.floor((value - 0.66) * 3 * 255)];
                }

            case 'jet':
                if (value < 0.25) {
                    return [0, 0, Math.floor(128 + value * 4 * 127)];
                } else if (value < 0.5) {
                    return [0, Math.floor((value - 0.25) * 4 * 255), 255];
                } else if (value < 0.75) {
                    return [Math.floor((value - 0.5) * 4 * 255), 255, Math.floor(255 - (value - 0.5) * 4 * 255)];
                } else {
                    return [255, Math.floor(255 - (value - 0.75) * 4 * 255), 0];
                }

            case 'viridis':
                // Simplified viridis colormap
                const r = Math.floor(255 * (0.267 + 0.004 * value));
                const g = Math.floor(255 * (0.004 + 0.898 * value));
                const b = Math.floor(255 * (0.329 + 0.718 * value));
                return [r, g, b];

            default:
                return [Math.floor(value * 255), 0, 0];
        }
    };

    const generateFeatureImportanceChart = () => {
        if (!explainabilityData.feature_importance) return null;

        const features = Object.keys(explainabilityData.feature_importance);
        const importance = Object.values(explainabilityData.feature_importance);

        const data = [{
            x: importance,
            y: features,
            type: 'bar',
            orientation: 'h',
            marker: {
                color: importance.map(imp => imp > 0 ? '#28a745' : '#dc3545')
            }
        }];

        const layout = {
            title: 'Feature Importance',
            xaxis: { title: 'Importance Score' },
            margin: { l: 150, r: 50, t: 50, b: 50 },
            height: 400
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

    if (!explainabilityData) {
        return (
            <Alert variant="info">
                No explainability data available. Run model inference to generate explanations.
            </Alert>
        );
    }

    const availableMethods = Object.keys(explainabilityData).filter(key =>
        key !== 'feature_importance' && explainabilityData[key].slices
    );
    const maxSlices = explainabilityData[selectedMethod]?.slices?.length || 0;

    return (
        <div>
            <Row>
                <Col md={12}>
                    <Card className="mb-3">
                        <Card.Header>
                            <Row className="align-items-center">
                                <Col>
                                    <h6 className="mb-0">Explainability Visualization</h6>
                                </Col>
                                <Col xs="auto">
                                    <ButtonGroup size="sm">
                                        {availableMethods.map(method => (
                                            <Button
                                                key={method}
                                                variant={selectedMethod === method ? 'primary' : 'outline-primary'}
                                                onClick={() => setSelectedMethod(method)}
                                            >
                                                {method.replace('_', ' ').toUpperCase()}
                                            </Button>
                                        ))}
                                    </ButtonGroup>
                                </Col>
                            </Row>
                        </Card.Header>

                        <Card.Body>
                            <Row>
                                <Col md={8}>
                                    <div className="medical-viewer position-relative">
                                        <canvas
                                            ref={canvasRef}
                                            width={512}
                                            height={512}
                                            style={{ width: '100%', height: 'auto', maxHeight: '400px' }}
                                        />

                                        {/* Slice navigation */}
                                        <div className="position-absolute bottom-0 start-0 end-0 p-2 bg-dark bg-opacity-75">
                                            <Row className="align-items-center text-white">
                                                <Col>
                                                    <Form.Range
                                                        min={0}
                                                        max={maxSlices - 1}
                                                        value={currentSlice}
                                                        onChange={(e) => setCurrentSlice(parseInt(e.target.value))}
                                                    />
                                                </Col>
                                                <Col xs="auto">
                                                    <small>Slice {currentSlice + 1} / {maxSlices}</small>
                                                </Col>
                                            </Row>
                                        </div>
                                    </div>
                                </Col>

                                <Col md={4}>
                                    <Card>
                                        <Card.Header>
                                            <small>Visualization Controls</small>
                                        </Card.Header>
                                        <Card.Body>
                                            <div className="mb-3">
                                                <Form.Label>Colormap</Form.Label>
                                                <Form.Select
                                                    value={colormap}
                                                    onChange={(e) => setColormap(e.target.value)}
                                                >
                                                    <option value="hot">Hot</option>
                                                    <option value="jet">Jet</option>
                                                    <option value="viridis">Viridis</option>
                                                </Form.Select>
                                            </div>

                                            <div className="mb-3">
                                                <Form.Label>Overlay Opacity</Form.Label>
                                                <Form.Range
                                                    min={0}
                                                    max={1}
                                                    step={0.1}
                                                    value={overlayOpacity}
                                                    onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                                                />
                                            </div>

                                            <Alert variant="info" className="mt-3">
                                                <small>
                                                    <strong>{selectedMethod.replace('_', ' ').toUpperCase()}:</strong>
                                                    <br />
                                                    {selectedMethod === 'grad_cam' &&
                                                        'Shows regions that most influenced the model\'s decision. Warmer colors indicate higher importance.'
                                                    }
                                                    {selectedMethod === 'integrated_gradients' &&
                                                        'Shows pixel-level attribution scores. Positive values (warm colors) support the prediction, negative values (cool colors) oppose it.'
                                                    }
                                                </small>
                                            </Alert>
                                        </Card.Body>
                                    </Card>
                                </Col>
                            </Row>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>

            {explainabilityData.feature_importance && (
                <Row>
                    <Col md={12}>
                        <Card>
                            <Card.Header>
                                <h6 className="mb-0">Feature Importance Analysis</h6>
                            </Card.Header>
                            <Card.Body>
                                {generateFeatureImportanceChart()}

                                <Alert variant="info" className="mt-3">
                                    <small>
                                        Feature importance scores show which input features (imaging regions, wearable sensor data)
                                        contributed most to the model's prediction. Positive scores support the predicted class,
                                        while negative scores oppose it.
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

export default ExplainabilityVisualization;