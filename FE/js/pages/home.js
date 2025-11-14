// Home Page Component

class HomePage {
    render() {
        return `
            <div class="min-vh-100">
                <!-- Hero Section -->
                <div class="hero-section">
                    <div class="container py-5">
                        <div class="row align-items-center g-4">
                            <div class="col-lg-6">
                                <h1 class="hero-title mb-4">
                                    In-house AI Object Detection Platform
                                </h1>
                                <p class="lead text-muted mb-4">
                                    Complete lifecycle management for industrial computer vision. From dataset annotation to production deployment, all in one secure platform.
                                </p>
                                <div class="d-flex gap-3">
                                    <a href="#/datasets" class="btn btn-primary btn-lg">
                                        <i class="bi bi-lightning-charge-fill me-2"></i>Get Started
                                    </a>
                                </div>
                            </div>
                            <div class="col-lg-6">
                                <img src="https://images.unsplash.com/photo-1555255707-c07966088b7b?w=800&h=600&fit=crop"
                                     alt="AI Detection Platform"
                                     class="img-fluid rounded-3 shadow-lg hero-image">
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Features Section -->
                <div class="container py-5">
                    <div class="text-center mb-5">
                        <h2 class="display-5 fw-bold mb-3">Complete AI Lifecycle Management</h2>
                        <p class="lead text-muted">
                            Everything you need to build, deploy, and maintain production-ready object detection models
                        </p>
                    </div>

                    <div class="row g-4 mb-5">
                        ${this.renderFeatureCards()}
                    </div>
                </div>

                <!-- Benefits Section -->
                <div class="bg-light py-5">
                    <div class="container">
                        <div class="text-center mb-5">
                            <h2 class="display-5 fw-bold mb-3">Why VisionInApp?</h2>
                            <p class="lead text-muted">
                                Built for enterprise teams who need control, security, and scalability
                            </p>
                        </div>

                        <div class="row g-4">
                            ${this.renderBenefitCards()}
                        </div>
                    </div>
                </div>

                <!-- CTA Section -->
                <div class="container py-5">
                    <div class="card bg-gradient-primary text-white">
                        <div class="card-body p-5 text-center">
                            <h2 class="display-5 fw-bold mb-3">Ready to Transform Your AI Workflow?</h2>
                            <p class="lead mb-4 opacity-75">
                                Start managing your object detection lifecycle with enterprise-grade tools
                            </p>
                            <a href="#/datasets" class="btn btn-light btn-lg">
                                <i class="bi bi-lightning-charge-fill me-2"></i>Start Building
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderFeatureCards() {
        const features = [
            {
                icon: 'database',
                title: 'Dataset Management',
                description: 'Upload, preprocess, and auto-annotate your training data with intelligent automation',
                link: '/datasets'
            },
            {
                icon: 'activity',
                title: 'Model Training',
                description: 'Track real-time metrics and performance graphs during model training',
                link: '/training'
            },
            {
                icon: 'gear',
                title: 'Model Conversion',
                description: 'Export to ONNX, TensorRT, and other formats optimized for your deployment',
                link: '/conversion'
            },
            {
                icon: 'cloud',
                title: 'Flexible Deployment',
                description: 'Deploy to edge devices, cloud platforms, or on-premise infrastructure',
                link: '/deployment'
            },
            {
                icon: 'bar-chart',
                title: 'Insightful Evaluation',
                description: 'Analyze model quality with rich metrics and uncover actionable insights',
                link: '/evaluation'
            },
            {
                icon: 'layers',
                title: 'Seamless Export',
                description: 'Deliver your models and datasets anywhere with streamlined packaging',
                link: '/export'
            }
        ];

        return features.map(feature => `
            <div class="col-md-6 col-lg-4">
                <a href="#${feature.link}" class="text-decoration-none">
                    <div class="card h-100 border-0 shadow-sm hover-card">
                        <div class="card-body">
                            <div class="icon-gradient d-flex align-items-center justify-content-center mb-3">
                                <i class="bi bi-${feature.icon} text-white fs-4"></i>
                            </div>
                            <h5 class="card-title fw-bold">${feature.title}</h5>
                            <p class="card-text text-muted">${feature.description}</p>
                        </div>
                    </div>
                </a>
            </div>
        `).join('');
    }

    renderBenefitCards() {
        const benefits = [
            {
                icon: 'lightning-charge',
                title: 'Automation First',
                description: 'Reduce manual work with intelligent auto-annotation and preprocessing'
            },
            {
                icon: 'shield-check',
                title: 'Enterprise Security',
                description: 'Private deployment ensures your data never leaves your infrastructure'
            },
            {
                icon: 'graph-up-arrow',
                title: 'Continuous Improvement',
                description: 'Automated feedback loops keep models accurate and up-to-date'
            }
        ];

        return benefits.map(benefit => `
            <div class="col-md-4">
                <div class="card h-100 border-0 shadow-sm">
                    <div class="card-body text-center p-4">
                        <div class="rounded-circle bg-gradient-primary d-inline-flex align-items-center justify-content-center mb-3"
                             style="width: 64px; height: 64px;">
                            <i class="bi bi-${benefit.icon} text-white fs-2"></i>
                        </div>
                        <h5 class="fw-bold mb-3">${benefit.title}</h5>
                        <p class="text-muted mb-0">${benefit.description}</p>
                    </div>
                </div>
            </div>
        `).join('');
    }
}
