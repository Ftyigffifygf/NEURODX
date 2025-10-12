import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Container, Navbar, Nav } from 'react-bootstrap';
import ImageUploadPage from './pages/ImageUploadPage';
import DiagnosticResultsPage from './pages/DiagnosticResultsPage';
import AnnotationPage from './pages/AnnotationPage';

function App() {
    return (
        <Router>
            <div className="App">
                <Navbar bg="dark" variant="dark" expand="lg">
                    <Container>
                        <Navbar.Brand href="/">NeuroDx-MultiModal</Navbar.Brand>
                        <Navbar.Toggle aria-controls="basic-navbar-nav" />
                        <Navbar.Collapse id="basic-navbar-nav">
                            <Nav className="me-auto">
                                <Nav.Link href="/">Upload Images</Nav.Link>
                                <Nav.Link href="/results">Diagnostic Results</Nav.Link>
                                <Nav.Link href="/annotation">Annotation</Nav.Link>
                            </Nav>
                        </Navbar.Collapse>
                    </Container>
                </Navbar>

                <Container className="mt-4">
                    <Routes>
                        <Route path="/" element={<ImageUploadPage />} />
                        <Route path="/results" element={<DiagnosticResultsPage />} />
                        <Route path="/annotation" element={<AnnotationPage />} />
                    </Routes>
                </Container>
            </div>
        </Router>
    );
}

export default App;