import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Alert, Badge, ListGroup } from 'react-bootstrap';

const ImageDropzone = ({ onFilesSelected, acceptedFiles, processingStatus }) => {
    const [uploadError, setUploadError] = useState(null);

    const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
        setUploadError(null);

        if (rejectedFiles.length > 0) {
            const errorMessages = rejectedFiles.map(file =>
                `${file.file.name}: ${file.errors.map(e => e.message).join(', ')}`
            );
            setUploadError(`Invalid files: ${errorMessages.join('; ')}`);
            return;
        }

        // Validate medical imaging file formats
        const validFiles = acceptedFiles.filter(file => {
            const extension = file.name.toLowerCase();
            return extension.endsWith('.nii') ||
                extension.endsWith('.nii.gz') ||
                extension.endsWith('.dcm') ||
                extension.endsWith('.dicom');
        });

        if (validFiles.length !== acceptedFiles.length) {
            setUploadError('Only NIfTI (.nii, .nii.gz) and DICOM (.dcm, .dicom) files are supported');
            return;
        }

        onFilesSelected(validFiles);
    }, [onFilesSelected]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'application/octet-stream': ['.nii', '.nii.gz', '.dcm', '.dicom'],
            'application/dicom': ['.dcm', '.dicom']
        },
        multiple: true,
        maxSize: 500 * 1024 * 1024 // 500MB max file size
    });

    const getModalityFromFilename = (filename) => {
        const lower = filename.toLowerCase();
        if (lower.includes('mri') || lower.includes('t1') || lower.includes('t2') || lower.includes('flair')) {
            return 'MRI';
        } else if (lower.includes('ct')) {
            return 'CT';
        } else if (lower.includes('ultrasound') || lower.includes('us')) {
            return 'Ultrasound';
        }
        return 'Unknown';
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'pending': return 'warning';
            case 'processing': return 'primary';
            case 'complete': return 'success';
            case 'error': return 'danger';
            default: return 'secondary';
        }
    };

    return (
        <div>
            <div
                {...getRootProps()}
                className={`upload-zone ${isDragActive ? 'drag-active' : ''}`}
            >
                <input {...getInputProps()} />
                <div>
                    <i className="fas fa-cloud-upload-alt fa-3x mb-3 text-primary"></i>
                    <h5>Drop medical images here or click to browse</h5>
                    <p className="text-muted">
                        Supported formats: NIfTI (.nii, .nii.gz), DICOM (.dcm, .dicom)
                        <br />
                        Maximum file size: 500MB per file
                    </p>
                </div>
            </div>

            {uploadError && (
                <Alert variant="danger" className="mt-3">
                    {uploadError}
                </Alert>
            )}

            {acceptedFiles.length > 0 && (
                <div className="mt-4">
                    <h6>Uploaded Files:</h6>
                    <ListGroup>
                        {acceptedFiles.map((file, index) => {
                            const modality = getModalityFromFilename(file.name);
                            const status = processingStatus[file.name] || 'pending';

                            return (
                                <ListGroup.Item key={index} className="d-flex justify-content-between align-items-center">
                                    <div>
                                        <strong>{file.name}</strong>
                                        <br />
                                        <small className="text-muted">
                                            {(file.size / (1024 * 1024)).toFixed(2)} MB
                                        </small>
                                    </div>
                                    <div className="d-flex align-items-center gap-2">
                                        <Badge bg="info" className="modality-badge">
                                            {modality}
                                        </Badge>
                                        <div className="processing-status">
                                            <div className={`status-indicator status-${status}`}></div>
                                            <Badge bg={getStatusColor(status)}>
                                                {status.charAt(0).toUpperCase() + status.slice(1)}
                                            </Badge>
                                        </div>
                                    </div>
                                </ListGroup.Item>
                            );
                        })}
                    </ListGroup>
                </div>
            )}
        </div>
    );
};

export default ImageDropzone;