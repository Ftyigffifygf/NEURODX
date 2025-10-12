# NeuroDx-MultiModal Frontend

React-based frontend interface for the NeuroDx-MultiModal diagnostic system.

## Features

### Image Upload and Visualization (Task 10.1)
- **Medical Image Upload**: Drag-and-drop interface supporting NIfTI (.nii, .nii.gz) and DICOM (.dcm, .dicom) formats
- **Multi-Modal Viewer**: Interactive viewer for MRI, CT, and Ultrasound images with:
  - Slice navigation and zoom controls
  - Window/level adjustment
  - Overlay support for segmentation masks
  - Multi-modality switching

### Diagnostic Results Dashboard (Task 10.2)
- **Classification Results**: Display of disease classification probabilities and confidence scores
- **Segmentation Visualization**: Interactive segmentation mask overlay with:
  - Label filtering and opacity controls
  - Statistical analysis of segmented regions
  - Color-coded anatomical structures
- **Explainability Features**: Visualization of AI decision-making through:
  - Grad-CAM attention maps
  - Integrated Gradients attribution
  - Feature importance analysis

### MONAI Label Annotation Interface (Task 10.3)
- **Active Learning Integration**: Smart sample selection based on model uncertainty
- **Interactive Annotation Canvas**: Professional annotation tools with:
  - Drawing and erasing tools
  - Adjustable brush sizes
  - Multi-label support
  - Undo/redo functionality
- **Quality Feedback System**: Real-time annotation quality assessment with:
  - Coverage and consistency metrics
  - Boundary quality analysis
  - Improvement suggestions

## Technology Stack

- **React 18**: Modern React with hooks and functional components
- **React Bootstrap**: Professional UI components
- **Plotly.js**: Interactive charts and visualizations
- **Axios**: HTTP client for API communication
- **Canvas API**: Medical image rendering and annotation

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Environment Configuration**:
   - The frontend is configured to proxy API requests to `http://localhost:5000`
   - Ensure the Flask backend is running on port 5000

3. **Start Development Server**:
   ```bash
   npm start
   ```
   - Opens browser at `http://localhost:3000`
   - Hot reload enabled for development

4. **Build for Production**:
   ```bash
   npm run build
   ```

## Usage Guide

### Image Upload Workflow
1. Navigate to the home page
2. Drag and drop medical images or click to browse
3. Monitor processing status in real-time
4. View processed images in the multi-modal viewer

### Diagnostic Analysis
1. Upload and process medical images
2. Navigate to "Diagnostic Results" page
3. Select a study to analyze
4. Click "Run Analysis" to generate AI predictions
5. Review classification results, segmentation masks, and explainability visualizations

### Annotation Workflow
1. Navigate to "Annotation" page
2. Review active learning suggestions in the left panel
3. Select a high-priority sample for annotation
4. Use annotation tools to create segmentation labels
5. Monitor quality feedback and apply suggestions
6. Submit completed annotations to improve the model

## Component Architecture

```
src/
├── components/
│   ├── ImageUpload/
│   │   ├── ImageDropzone.js          # File upload interface
│   │   └── ProcessingProgress.js     # Processing status tracking
│   ├── ImageViewer/
│   │   └── MultiModalViewer.js       # Medical image viewer
│   ├── DiagnosticResults/
│   │   ├── SegmentationViewer.js     # Segmentation visualization
│   │   ├── ClassificationResults.js  # Classification display
│   │   └── ExplainabilityVisualization.js # AI explainability
│   └── Annotation/
│       ├── AnnotationCanvas.js       # Interactive annotation
│       ├── ActiveLearningPanel.js    # Sample suggestions
│       └── AnnotationQualityFeedback.js # Quality assessment
├── pages/
│   ├── ImageUploadPage.js           # Main upload interface
│   ├── DiagnosticResultsPage.js     # Results dashboard
│   └── AnnotationPage.js            # Annotation interface
└── App.js                           # Main application component
```

## API Integration

The frontend communicates with the Flask backend through these endpoints:

- **Image Processing**: `/api/images/*`
- **Diagnostic Results**: `/api/diagnostics/*`
- **MONAI Label**: `/api/monai-label/*`

## Medical Imaging Features

### Supported Formats
- **NIfTI**: .nii, .nii.gz (preferred for research)
- **DICOM**: .dcm, .dicom (clinical standard)

### Visualization Capabilities
- **Multi-planar reconstruction**: Axial, sagittal, coronal views
- **Window/Level adjustment**: Optimize contrast for different tissues
- **Zoom and pan**: Navigate large high-resolution images
- **Overlay rendering**: Segmentation masks and attention maps

### Annotation Tools
- **Brush-based drawing**: Variable size brushes for precise annotation
- **Multi-label support**: Different colors for anatomical structures
- **Quality metrics**: Real-time feedback on annotation quality
- **Export functionality**: Save annotations in NIfTI format

## Performance Considerations

- **Canvas rendering**: Optimized for large medical images
- **Memory management**: Efficient handling of 3D image data
- **Progressive loading**: Slice-by-slice rendering for responsiveness
- **Caching**: Client-side caching of processed images

## Compliance and Security

- **HIPAA considerations**: No patient data stored in browser localStorage
- **Secure communication**: All API calls use HTTPS in production
- **Data anonymization**: Patient identifiers removed from display
- **Audit logging**: User actions tracked for compliance

## Development Notes

- **Medical imaging standards**: Follows DICOM and NIfTI conventions
- **Accessibility**: Keyboard navigation and screen reader support
- **Cross-browser compatibility**: Tested on Chrome, Firefox, Safari
- **Mobile responsiveness**: Optimized for tablet use in clinical settings

## Testing

Run the test suite:
```bash
npm test -- --run
```

Tests cover:
- Component rendering
- User interactions
- API integration
- Medical image processing
- Annotation functionality