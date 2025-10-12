import React, { useState, useEffect, useRef } from 'react';
import { Card, Row, Col, Form, Badge, ButtonGroup, Button } from 'react-bootstrap';

const SegmentationViewer = ({ segmentationData, originalImage, onSegmentationChange }) => {
    const canvasRef = useRef(null);
    const [currentSlice, setCurrentSlice] = useState(0);
    const [selectedLabel, setSelectedLabel] = useState('all');
    const [overlayOpacity, setOverlayOpacity] = useState(0.7);
    const [showOriginal, setShowOriginal] = useState(true);

    const segmentationLabels = {
        0: { name: 'Background', color: [0, 0, 0, 0] },
        1: { name: 'Pathological Region', color: [255, 0, 0, 255] },
        2: { name: 'Normal Tissue', color: [0, 255, 0, 255] },
        3: { name: 'Ventricles', color: [0, 0, 255, 255] },
        4: { name: 'White Matter', color: [255, 255, 0, 255] },
        5: { name: 'Gray Matter', color: [255, 0, 255, 255] }
    };

    useEffect(() => {
        if (segmentationData && canvasRef.current) {
            renderSegmentation();
        }
    }, [segmentationData, currentSlice, selectedLabel, overlayOpacity, showOriginal]);

    const renderSegmentation = () => {
        const canvas = canvasRef.current;
        if (!canvas || !segmentationData) return;

        const ctx = canvas.getContext('2d');
        const slice = segmentationData.slices[currentSlice];

        if (!slice) return;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw original image if enabled
        if (showOriginal && originalImage && originalImage.slices[currentSlice]) {
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

            // Scale to fit canvas
            ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
        }

        // Draw segmentation overlay
        const segImageData = createSegmentationImageData(
            slice.data,
            slice.width,
            slice.height,
            selectedLabel
        );

        const segCanvas = document.createElement('canvas');
        segCanvas.width = slice.width;
        segCanvas.height = slice.height;
        const segCtx = segCanvas.getContext('2d');
        segCtx.putImageData(segImageData, 0, 0);

        // Apply opacity and draw overlay
        ctx.globalAlpha = overlayOpacity;
        ctx.drawImage(segCanvas, 0, 0, canvas.width, canvas.height);
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

    const createSegmentationImageData = (data, width, height, filterLabel) => {
        const imageData = new Uint8ClampedArray(width * height * 4);

        for (let i = 0; i < data.length; i++) {
            const label = data[i];
            const idx = i * 4;

            if (label > 0 && (filterLabel === 'all' || filterLabel === label.toString())) {
                const color = segmentationLabels[label]?.color || [255, 255, 255, 255];
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

    const calculateSegmentationStats = () => {
        if (!segmentationData) return {};

        const stats = {};
        const totalVoxels = segmentationData.slices.reduce((total, slice) => {
            return total + slice.data.length;
        }, 0);

        // Count voxels per label
        Object.keys(segmentationLabels).forEach(label => {
            stats[label] = 0;
        });

        segmentationData.slices.forEach(slice => {
            slice.data.forEach(voxel => {
                if (stats[voxel] !== undefined) {
                    stats[voxel]++;
                }
            });
        });

        // Calculate percentages
        Object.keys(stats).forEach(label => {
            stats[label] = {
                count: stats[label],
                percentage: ((stats[label] / totalVoxels) * 100).toFixed(2)
            };
        });

        return stats;
    };

    const segmentationStats = calculateSegmentationStats();
    const maxSlices = segmentationData ? segmentationData.slices.length : 0;

    if (!segmentationData) {
        return (
            <Card>
                <Card.Body className="text-center text-muted">
                    No segmentation data available
                </Card.Body>
            </Card>
        );
    }

    return (
        <Card>
            <Card.Header>
                <Row className="align-items-center">
                    <Col>
                        <h6 className="mb-0">Segmentation Results</h6>
                    </Col>
                    <Col xs="auto">
                        <Form.Check
                            type="switch"
                            label="Show Original"
                            checked={showOriginal}
                            onChange={(e) => setShowOriginal(e.target.checked)}
                        />
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
                        {/* Controls */}
                        <Card className="mb-3">
                            <Card.Header>
                                <small>Display Controls</small>
                            </Card.Header>
                            <Card.Body>
                                <div className="mb-3">
                                    <Form.Label>Filter by Label</Form.Label>
                                    <Form.Select
                                        value={selectedLabel}
                                        onChange={(e) => setSelectedLabel(e.target.value)}
                                    >
                                        <option value="all">Show All Labels</option>
                                        {Object.entries(segmentationLabels).map(([value, label]) => (
                                            value !== '0' && (
                                                <option key={value} value={value}>
                                                    {label.name}
                                                </option>
                                            )
                                        ))}
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
                            </Card.Body>
                        </Card>

                        {/* Segmentation Statistics */}
                        <Card>
                            <Card.Header>
                                <small>Segmentation Statistics</small>
                            </Card.Header>
                            <Card.Body>
                                {Object.entries(segmentationLabels).map(([value, label]) => {
                                    const stats = segmentationStats[value];
                                    if (!stats || value === '0' || stats.count === 0) return null;

                                    return (
                                        <div key={value} className="d-flex justify-content-between align-items-center mb-2">
                                            <div className="d-flex align-items-center">
                                                <div
                                                    className="me-2"
                                                    style={{
                                                        width: '12px',
                                                        height: '12px',
                                                        backgroundColor: `rgba(${label.color.slice(0, 3).join(',')}, 1)`,
                                                        border: '1px solid #ccc'
                                                    }}
                                                ></div>
                                                <small>{label.name}</small>
                                            </div>
                                            <Badge bg="secondary">
                                                {stats.percentage}%
                                            </Badge>
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

export default SegmentationViewer;