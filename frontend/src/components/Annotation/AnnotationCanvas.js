import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Card, Row, Col, ButtonGroup, Button, Form, Alert } from 'react-bootstrap';

const AnnotationCanvas = ({ imageData, annotations, onAnnotationChange, annotationMode }) => {
    const canvasRef = useRef(null);
    const overlayCanvasRef = useRef(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [currentSlice, setCurrentSlice] = useState(0);
    const [selectedLabel, setSelectedLabel] = useState(1);
    const [brushSize, setBrushSize] = useState(5);
    const [zoom, setZoom] = useState(1.0);
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });

    const annotationLabels = {
        0: { name: 'Background', color: [0, 0, 0, 0] },
        1: { name: 'Pathological Region', color: [255, 0, 0, 128] },
        2: { name: 'Normal Tissue', color: [0, 255, 0, 128] },
        3: { name: 'Ventricles', color: [0, 0, 255, 128] },
        4: { name: 'White Matter', color: [255, 255, 0, 128] },
        5: { name: 'Gray Matter', color: [255, 0, 255, 128] }
    };

    useEffect(() => {
        if (imageData && canvasRef.current) {
            renderImage();
        }
    }, [imageData, currentSlice, zoom, pan]);

    useEffect(() => {
        if (annotations && overlayCanvasRef.current) {
            renderAnnotations();
        }
    }, [annotations, currentSlice, selectedLabel]);

    const renderImage = () => {
        const canvas = canvasRef.current;
        if (!canvas || !imageData) return;

        const ctx = canvas.getContext('2d');
        const slice = imageData.slices[currentSlice];

        if (!slice) return;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Apply transformations
        ctx.save();
        ctx.translate(pan.x, pan.y);
        ctx.scale(zoom, zoom);

        // Create grayscale image data
        const imageDataObj = createGrayscaleImageData(slice.data, slice.width, slice.height);

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = slice.width;
        tempCanvas.height = slice.height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.putImageData(imageDataObj, 0, 0);

        ctx.drawImage(tempCanvas, 0, 0);
        ctx.restore();
    };

    const renderAnnotations = () => {
        const canvas = overlayCanvasRef.current;
        if (!canvas || !annotations) return;

        const ctx = canvas.getContext('2d');
        const annotationSlice = annotations.slices[currentSlice];

        if (!annotationSlice) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            return;
        }

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Apply transformations
        ctx.save();
        ctx.translate(pan.x, pan.y);
        ctx.scale(zoom, zoom);

        // Create annotation overlay
        const annotationImageData = createAnnotationImageData(
            annotationSlice.data,
            annotationSlice.width,
            annotationSlice.height
        );

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = annotationSlice.width;
        tempCanvas.height = annotationSlice.height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.putImageData(annotationImageData, 0, 0);

        ctx.drawImage(tempCanvas, 0, 0);
        ctx.restore();
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

    const createAnnotationImageData = (data, width, height) => {
        const imageData = new Uint8ClampedArray(width * height * 4);

        for (let i = 0; i < data.length; i++) {
            const label = data[i];
            const idx = i * 4;

            if (label > 0) {
                const color = annotationLabels[label]?.color || [255, 255, 255, 128];
                imageData[idx] = color[0];     // R
                imageData[idx + 1] = color[1]; // G
                imageData[idx + 2] = color[2]; // B
                imageData[idx + 3] = color[3]; // A
            } else {
                imageData[idx + 3] = 0; // Transparent
            }
        }

        return new ImageData(imageData, width, height);
    };

    const getMousePosition = (e) => {
        const canvas = overlayCanvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    };

    const worldToImageCoords = (worldX, worldY) => {
        return {
            x: (worldX - pan.x) / zoom,
            y: (worldY - pan.y) / zoom
        };
    };

    const handleMouseDown = useCallback((e) => {
        const mousePos = getMousePosition(e);
        setLastMousePos(mousePos);

        if (annotationMode === 'pan') {
            setIsDragging(true);
        } else if (annotationMode === 'draw' || annotationMode === 'erase') {
            setIsDrawing(true);
            drawAnnotation(mousePos.x, mousePos.y);
        }
    }, [annotationMode, pan, zoom, selectedLabel, brushSize]);

    const handleMouseMove = useCallback((e) => {
        const mousePos = getMousePosition(e);

        if (isDragging && annotationMode === 'pan') {
            const deltaX = mousePos.x - lastMousePos.x;
            const deltaY = mousePos.y - lastMousePos.y;
            setPan(prev => ({
                x: prev.x + deltaX,
                y: prev.y + deltaY
            }));
        } else if (isDrawing && (annotationMode === 'draw' || annotationMode === 'erase')) {
            drawAnnotation(mousePos.x, mousePos.y);
        }

        setLastMousePos(mousePos);
    }, [isDragging, isDrawing, annotationMode, lastMousePos, selectedLabel, brushSize]);

    const handleMouseUp = useCallback(() => {
        setIsDrawing(false);
        setIsDragging(false);
    }, []);

    const drawAnnotation = (x, y) => {
        if (!annotations || !imageData) return;

        const imageCoords = worldToImageCoords(x, y);
        const slice = imageData.slices[currentSlice];

        if (!slice) return;

        // Calculate brush area
        const label = annotationMode === 'erase' ? 0 : selectedLabel;
        const updatedAnnotations = { ...annotations };

        if (!updatedAnnotations.slices[currentSlice]) {
            updatedAnnotations.slices[currentSlice] = {
                data: new Array(slice.width * slice.height).fill(0),
                width: slice.width,
                height: slice.height
            };
        }

        const annotationData = [...updatedAnnotations.slices[currentSlice].data];

        // Draw circular brush
        for (let dy = -brushSize; dy <= brushSize; dy++) {
            for (let dx = -brushSize; dx <= brushSize; dx++) {
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance <= brushSize) {
                    const pixelX = Math.floor(imageCoords.x + dx);
                    const pixelY = Math.floor(imageCoords.y + dy);

                    if (pixelX >= 0 && pixelX < slice.width && pixelY >= 0 && pixelY < slice.height) {
                        const index = pixelY * slice.width + pixelX;
                        annotationData[index] = label;
                    }
                }
            }
        }

        updatedAnnotations.slices[currentSlice].data = annotationData;

        if (onAnnotationChange) {
            onAnnotationChange(updatedAnnotations);
        }
    };

    const handleZoom = (factor) => {
        setZoom(prev => Math.max(0.1, Math.min(5.0, prev * factor)));
    };

    const handleReset = () => {
        setZoom(1.0);
        setPan({ x: 0, y: 0 });
    };

    const maxSlices = imageData ? imageData.slices.length : 0;

    if (!imageData) {
        return (
            <Alert variant="info">
                No image data available for annotation. Please load an image first.
            </Alert>
        );
    }

    return (
        <Card>
            <Card.Header>
                <Row className="align-items-center">
                    <Col>
                        <h6 className="mb-0">Annotation Canvas</h6>
                    </Col>
                    <Col xs="auto">
                        <ButtonGroup size="sm">
                            <Button onClick={() => handleZoom(1.2)}>Zoom In</Button>
                            <Button onClick={() => handleZoom(0.8)}>Zoom Out</Button>
                            <Button onClick={handleReset}>Reset</Button>
                        </ButtonGroup>
                    </Col>
                </Row>
            </Card.Header>

            <Card.Body>
                <Row>
                    <Col md={8}>
                        <div className="position-relative" style={{ overflow: 'hidden' }}>
                            {/* Base image canvas */}
                            <canvas
                                ref={canvasRef}
                                width={512}
                                height={512}
                                style={{
                                    position: 'absolute',
                                    width: '100%',
                                    height: 'auto',
                                    maxHeight: '500px',
                                    cursor: annotationMode === 'pan' ? 'grab' : 'crosshair'
                                }}
                            />

                            {/* Annotation overlay canvas */}
                            <canvas
                                ref={overlayCanvasRef}
                                width={512}
                                height={512}
                                style={{
                                    position: 'relative',
                                    width: '100%',
                                    height: 'auto',
                                    maxHeight: '500px',
                                    cursor: annotationMode === 'pan' ? 'grab' : 'crosshair'
                                }}
                                onMouseDown={handleMouseDown}
                                onMouseMove={handleMouseMove}
                                onMouseUp={handleMouseUp}
                                onMouseLeave={handleMouseUp}
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
                        {/* Annotation Controls */}
                        <Card className="mb-3">
                            <Card.Header>
                                <small>Annotation Tools</small>
                            </Card.Header>
                            <Card.Body>
                                <div className="mb-3">
                                    <Form.Label>Annotation Label</Form.Label>
                                    <Form.Select
                                        value={selectedLabel}
                                        onChange={(e) => setSelectedLabel(parseInt(e.target.value))}
                                        disabled={annotationMode === 'erase'}
                                    >
                                        {Object.entries(annotationLabels).map(([value, label]) => (
                                            value !== '0' && (
                                                <option key={value} value={value}>
                                                    {label.name}
                                                </option>
                                            )
                                        ))}
                                    </Form.Select>
                                </div>

                                <div className="mb-3">
                                    <Form.Label>Brush Size: {brushSize}px</Form.Label>
                                    <Form.Range
                                        min={1}
                                        max={20}
                                        value={brushSize}
                                        onChange={(e) => setBrushSize(parseInt(e.target.value))}
                                    />
                                </div>

                                <div className="mb-3">
                                    <small className="text-muted">
                                        Current Mode: <strong>{annotationMode}</strong>
                                    </small>
                                </div>
                            </Card.Body>
                        </Card>

                        {/* Label Legend */}
                        <Card>
                            <Card.Header>
                                <small>Label Legend</small>
                            </Card.Header>
                            <Card.Body>
                                {Object.entries(annotationLabels).map(([value, label]) => {
                                    if (value === '0') return null;

                                    return (
                                        <div key={value} className="d-flex align-items-center mb-2">
                                            <div
                                                className="me-2"
                                                style={{
                                                    width: '16px',
                                                    height: '16px',
                                                    backgroundColor: `rgba(${label.color.slice(0, 3).join(',')}, 0.7)`,
                                                    border: '1px solid #ccc'
                                                }}
                                            ></div>
                                            <small>{label.name}</small>
                                        </div>
                                    );
                                })}
                            </Card.Body>
                        </Card>
                    </Col>
                </Row>
            </Card.Body>
        </Card>
    );
};

export default AnnotationCanvas;