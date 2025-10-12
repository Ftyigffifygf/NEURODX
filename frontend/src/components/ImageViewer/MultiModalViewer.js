import React, { useState, useEffect, useRef } from 'react';
import { Card, Row, Col, Button, ButtonGroup, Form, Alert } from 'react-bootstrap';

const MultiModalViewer = ({ imageData, overlayData, onSliceChange }) => {
    const canvasRef = useRef(null);
    const [currentSlice, setCurrentSlice] = useState(0);
    const [selectedModality, setSelectedModality] = useState('MRI');
    const [showOverlay, setShowOverlay] = useState(true);
    const [overlayOpacity, setOverlayOpacity] = useState(0.5);
    const [windowLevel, setWindowLevel] = useState({ center: 128, width: 256 });
    const [zoom, setZoom] = useState(1.0);
    const [pan, setPan] = useState({ x: 0, y: 0 });

    useEffect(() => {
        if (imageData && canvasRef.current) {
            renderImage();
        }
    }, [imageData, currentSlice, selectedModality, showOverlay, overlayOpacity, windowLevel, zoom, pan]);

    const renderImage = () => {
        const canvas = canvasRef.current;
        if (!canvas || !imageData) return;

        const ctx = canvas.getContext('2d');
        const modalityData = imageData[selectedModality];

        if (!modalityData || !modalityData.slices) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            return;
        }

        const slice = modalityData.slices[currentSlice];
        if (!slice) return;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Apply window/level and render base image
        const imageData2D = applyWindowLevel(slice.data, windowLevel);
        const imageDataObj = new ImageData(imageData2D, slice.width, slice.height);

        // Apply zoom and pan transformations
        ctx.save();
        ctx.translate(pan.x, pan.y);
        ctx.scale(zoom, zoom);

        // Draw the medical image
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = slice.width;
        tempCanvas.height = slice.height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.putImageData(imageDataObj, 0, 0);

        ctx.drawImage(tempCanvas, 0, 0);

        // Draw overlay if enabled
        if (showOverlay && overlayData && overlayData[selectedModality]) {
            const overlaySlice = overlayData[selectedModality].slices[currentSlice];
            if (overlaySlice) {
                ctx.globalAlpha = overlayOpacity;
                const overlayImageData = createOverlayImageData(overlaySlice.data, slice.width, slice.height);
                const overlayCanvas = document.createElement('canvas');
                overlayCanvas.width = slice.width;
                overlayCanvas.height = slice.height;
                const overlayCtx = overlayCanvas.getContext('2d');
                overlayCtx.putImageData(overlayImageData, 0, 0);
                ctx.drawImage(overlayCanvas, 0, 0);
                ctx.globalAlpha = 1.0;
            }
        }

        ctx.restore();
    };

    const applyWindowLevel = (data, wl) => {
        const { center, width } = wl;
        const min = center - width / 2;
        const max = center + width / 2;

        const result = new Uint8ClampedArray(data.length * 4);

        for (let i = 0; i < data.length; i++) {
            let value = data[i];
            value = Math.max(min, Math.min(max, value));
            value = ((value - min) / (max - min)) * 255;

            const idx = i * 4;
            result[idx] = value;     // R
            result[idx + 1] = value; // G
            result[idx + 2] = value; // B
            result[idx + 3] = 255;   // A
        }

        return result;
    };

    const createOverlayImageData = (data, width, height) => {
        const result = new Uint8ClampedArray(width * height * 4);

        for (let i = 0; i < data.length; i++) {
            const idx = i * 4;
            if (data[i] > 0) {
                // Color-code different segmentation labels
                const label = data[i];
                switch (label) {
                    case 1: // Red for pathological regions
                        result[idx] = 255;
                        result[idx + 1] = 0;
                        result[idx + 2] = 0;
                        break;
                    case 2: // Green for normal tissue
                        result[idx] = 0;
                        result[idx + 1] = 255;
                        result[idx + 2] = 0;
                        break;
                    case 3: // Blue for other structures
                        result[idx] = 0;
                        result[idx + 1] = 0;
                        result[idx + 2] = 255;
                        break;
                    default:
                        result[idx] = 255;
                        result[idx + 1] = 255;
                        result[idx + 2] = 0;
                }
                result[idx + 3] = 255;
            } else {
                result[idx + 3] = 0; // Transparent
            }
        }

        return new ImageData(result, width, height);
    };

    const handleSliceChange = (newSlice) => {
        setCurrentSlice(newSlice);
        if (onSliceChange) {
            onSliceChange(newSlice, selectedModality);
        }
    };

    const handleZoom = (factor) => {
        setZoom(prev => Math.max(0.1, Math.min(5.0, prev * factor)));
    };

    const handleReset = () => {
        setZoom(1.0);
        setPan({ x: 0, y: 0 });
        setWindowLevel({ center: 128, width: 256 });
    };

    const maxSlices = imageData && imageData[selectedModality]
        ? imageData[selectedModality].slices.length
        : 0;

    if (!imageData) {
        return (
            <Alert variant="info">
                No image data available. Please upload medical images to view them here.
            </Alert>
        );
    }

    return (
        <Card>
            <Card.Header>
                <Row className="align-items-center">
                    <Col md={6}>
                        <h6 className="mb-0">Multi-Modal Image Viewer</h6>
                    </Col>
                    <Col md={6} className="text-end">
                        <ButtonGroup size="sm">
                            {Object.keys(imageData).map(modality => (
                                <Button
                                    key={modality}
                                    variant={selectedModality === modality ? 'primary' : 'outline-primary'}
                                    onClick={() => setSelectedModality(modality)}
                                >
                                    {modality}
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
                                style={{ width: '100%', height: 'auto', maxHeight: '500px' }}
                            />

                            {/* Slice navigation */}
                            <div className="position-absolute bottom-0 start-0 end-0 p-2 bg-dark bg-opacity-75">
                                <Row className="align-items-center text-white">
                                    <Col>
                                        <Form.Range
                                            min={0}
                                            max={maxSlices - 1}
                                            value={currentSlice}
                                            onChange={(e) => handleSliceChange(parseInt(e.target.value))}
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
                        {/* Viewer Controls */}
                        <Card className="mb-3">
                            <Card.Header>
                                <small>Viewer Controls</small>
                            </Card.Header>
                            <Card.Body>
                                <div className="mb-3">
                                    <Form.Check
                                        type="switch"
                                        label="Show Overlay"
                                        checked={showOverlay}
                                        onChange={(e) => setShowOverlay(e.target.checked)}
                                    />
                                </div>

                                {showOverlay && (
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
                                )}

                                <div className="mb-3">
                                    <Form.Label>Window Center</Form.Label>
                                    <Form.Range
                                        min={0}
                                        max={255}
                                        value={windowLevel.center}
                                        onChange={(e) => setWindowLevel(prev => ({ ...prev, center: parseInt(e.target.value) }))}
                                    />
                                </div>

                                <div className="mb-3">
                                    <Form.Label>Window Width</Form.Label>
                                    <Form.Range
                                        min={1}
                                        max={512}
                                        value={windowLevel.width}
                                        onChange={(e) => setWindowLevel(prev => ({ ...prev, width: parseInt(e.target.value) }))}
                                    />
                                </div>

                                <div className="d-grid gap-2">
                                    <ButtonGroup>
                                        <Button size="sm" onClick={() => handleZoom(1.2)}>Zoom In</Button>
                                        <Button size="sm" onClick={() => handleZoom(0.8)}>Zoom Out</Button>
                                    </ButtonGroup>
                                    <Button size="sm" variant="outline-secondary" onClick={handleReset}>
                                        Reset View
                                    </Button>
                                </div>
                            </Card.Body>
                        </Card>

                        {/* Image Information */}
                        <Card>
                            <Card.Header>
                                <small>Image Information</small>
                            </Card.Header>
                            <Card.Body>
                                {imageData[selectedModality] && (
                                    <div>
                                        <small>
                                            <strong>Modality:</strong> {selectedModality}<br />
                                            <strong>Dimensions:</strong> {imageData[selectedModality].dimensions?.join(' × ')}<br />
                                            <strong>Spacing:</strong> {imageData[selectedModality].spacing?.join(' × ')} mm<br />
                                            <strong>Slices:</strong> {maxSlices}<br />
                                            <strong>Data Type:</strong> {imageData[selectedModality].dataType}
                                        </small>
                                    </div>
                                )}
                            </Card.Body>
                        </Card>
                    </Col>
                </Row>
            </Card.Body>
        </Card>
    );
};

export default MultiModalViewer;