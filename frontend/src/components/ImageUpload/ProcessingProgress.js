import React from 'react';
import { Card, ProgressBar, Alert, Badge } from 'react-bootstrap';

const ProcessingProgress = ({ processingJobs, onJobComplete }) => {
    if (!processingJobs || processingJobs.length === 0) {
        return null;
    }

    const getProgressVariant = (status) => {
        switch (status) {
            case 'preprocessing': return 'info';
            case 'processing': return 'primary';
            case 'complete': return 'success';
            case 'error': return 'danger';
            default: return 'secondary';
        }
    };

    const getProgressValue = (job) => {
        if (job.status === 'complete') return 100;
        if (job.status === 'error') return 0;
        return job.progress || 0;
    };

    return (
        <Card className="mt-4">
            <Card.Header>
                <h6 className="mb-0">Processing Status</h6>
            </Card.Header>
            <Card.Body>
                {processingJobs.map((job, index) => (
                    <div key={index} className="mb-3">
                        <div className="d-flex justify-content-between align-items-center mb-2">
                            <div>
                                <strong>{job.filename}</strong>
                                <Badge bg="secondary" className="ms-2">{job.modality}</Badge>
                            </div>
                            <Badge bg={getProgressVariant(job.status)}>
                                {job.status}
                            </Badge>
                        </div>

                        <ProgressBar
                            now={getProgressValue(job)}
                            variant={getProgressVariant(job.status)}
                            label={job.status === 'complete' ? 'Complete' : `${job.progress || 0}%`}
                        />

                        {job.currentStep && (
                            <small className="text-muted d-block mt-1">
                                Current step: {job.currentStep}
                            </small>
                        )}

                        {job.error && (
                            <Alert variant="danger" className="mt-2 mb-0">
                                <small>{job.error}</small>
                            </Alert>
                        )}

                        {job.status === 'complete' && job.processingTime && (
                            <small className="text-success d-block mt-1">
                                Completed in {job.processingTime}s
                            </small>
                        )}
                    </div>
                ))}

                <div className="mt-3">
                    <small className="text-muted">
                        Processing steps: File validation → MONAI preprocessing → Feature extraction → Ready for analysis
                    </small>
                </div>
            </Card.Body>
        </Card>
    );
};

export default ProcessingProgress;