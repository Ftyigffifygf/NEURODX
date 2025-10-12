import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Button, Alert } from 'react-bootstrap';
import ImageDropzone from '../components/ImageUpload/ImageDropzone';
import ProcessingProgress from '../components/ImageUpload/ProcessingProgress';
import MultiModalViewer from '../components/ImageViewer/MultiModalViewer';
import axios from 'axios';

const ImageUploadPage = () => {
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [processingStatus, setProcessingStatus] = useState({});
    const [processingJobs, setProcessingJobs] = useState([]);
    const [imageData, setImageData] = useState(null);
    const [uploadError, setUploadError] = useState(null);
    const [isUploading, setIsUploading] = useState(false);

    const handleFilesSelected = async (files) => {
        setUploadedFiles(files);
        setUploadError(null);
        setIsUploading(true);

        // Initialize processing status
        const initialStatus = {};
        files.forEach(file => {
            initialStatus[file.name] = 'pending';
        });
        setProcessingStatus(initialStatus);

        try {
            // Upload files to backend
            const formData = new FormData();
            files.forEach(file => {
                formData.append('images', file);
            });

            const response = await axios.post('/api/images/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                onUploadProgress: (progressEvent) => {
                    const percentCompleted = Math.round(
                        (progressEvent.loaded * 100) / progressEvent.total
                    );
                    console.log(`Upload progress: ${percentCompleted}%`);
                },
            });

            if (response.data.success) {
                // Start processing jobs
                const jobs = response.data.processing_jobs || [];
                setProcessingJobs(jobs);

                // Poll for processing status
                jobs.forEach(job => {
                    pollProcessingStatus(job.job_id, job.filename);
                });
            } else {
                setUploadError(response.data.error || 'Upload failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            setUploadError(error.response?.data?.error || 'Failed to upload files');
        } finally {
            setIsUploading(false);
        }
    };

    const pollProcessingStatus = async (jobId, filename) => {
        const maxAttempts = 60; // 5 minutes with 5-second intervals
        let attempts = 0;

        const poll = async () => {
            try {
                const response = await axios.get(`/api/images/processing-status/${jobId}`);
                const job = response.data;

                // Update processing jobs
                setProcessingJobs(prev =>
                    prev.map(j => j.job_id === jobId ? job : j)
                );

                // Update processing status
                setProcessingStatus(prev => ({
                    ...prev,
                    [filename]: job.status
                }));

                if (job.status === 'complete') {
                    // Load processed image data
                    loadImageData(job.result.study_id);
                } else if (job.status === 'error') {
                    console.error(`Processing failed for ${filename}:`, job.error);
                } else if (attempts < maxAttempts) {
                    attempts++;
                    setTimeout(poll, 5000); // Poll every 5 seconds
                }
            } catch (error) {
                console.error('Error polling status:', error);
                if (attempts < maxAttempts) {
                    attempts++;
                    setTimeout(poll, 5000);
                }
            }
        };

        poll();
    };

    const loadImageData = async (studyId) => {
        try {
            const response = await axios.get(`/api/images/study/${studyId}`);
            if (response.data.success) {
                setImageData(response.data.image_data);
            }
        } catch (error) {
            console.error('Error loading image data:', error);
        }
    };

    const handleProcessAll = async () => {
        if (uploadedFiles.length === 0) return;

        try {
            const response = await axios.post('/api/images/process-all', {
                files: uploadedFiles.map(f => f.name)
            });

            if (response.data.success) {
                console.log('Processing started for all files');
            }
        } catch (error) {
            console.error('Error starting processing:', error);
            setUploadError('Failed to start processing');
        }
    };

    const allFilesProcessed = uploadedFiles.length > 0 &&
        uploadedFiles.every(file => processingStatus[file.name] === 'complete');

    return (
        <div>
            <Row>
                <Col md={12}>
                    <Card className="mb-4">
                        <Card.Header>
                            <h5 className="mb-0">Medical Image Upload</h5>
                        </Card.Header>
                        <Card.Body>
                            <ImageDropzone
                                onFilesSelected={handleFilesSelected}
                                acceptedFiles={uploadedFiles}
                                processingStatus={processingStatus}
                            />

                            {uploadError && (
                                <Alert variant="danger" className="mt-3">
                                    {uploadError}
                                </Alert>
                            )}

                            {uploadedFiles.length > 0 && (
                                <div className="mt-3 d-flex gap-2">
                                    <Button
                                        variant="primary"
                                        onClick={handleProcessAll}
                                        disabled={isUploading || allFilesProcessed}
                                    >
                                        {isUploading ? 'Uploading...' : 'Process All Images'}
                                    </Button>

                                    {allFilesProcessed && (
                                        <Button
                                            variant="success"
                                            href="/results"
                                        >
                                            View Results
                                        </Button>
                                    )}
                                </div>
                            )}
                        </Card.Body>
                    </Card>
                </Col>
            </Row>

            {processingJobs.length > 0 && (
                <Row>
                    <Col md={12}>
                        <ProcessingProgress
                            processingJobs={processingJobs}
                            onJobComplete={(jobId) => console.log(`Job ${jobId} completed`)}
                        />
                    </Col>
                </Row>
            )}

            {imageData && (
                <Row className="mt-4">
                    <Col md={12}>
                        <MultiModalViewer
                            imageData={imageData}
                            overlayData={null}
                            onSliceChange={(slice, modality) => {
                                console.log(`Slice changed: ${slice} for ${modality}`);
                            }}
                        />
                    </Col>
                </Row>
            )}
        </div>
    );
};

export default ImageUploadPage;