// Deployment Page Component

class DeploymentPage {
    constructor() {
        this.deployments = [];
    }

    async init() {
        await this.loadDeployments();
    }

    async loadDeployments() {
        try {
            const response = await window.apiService.get('/deployment');
            this.deployments = response;
            this.renderDeploymentsContent();
        } catch (error) {
            console.error('Failed to load deployments:', error);
            this.deployments = [];
            this.renderDeploymentsContent();
        }
    }

    renderDeploymentsContent() {
        const statsContainer = document.getElementById('deployment-stats');
        const listContainer = document.getElementById('deployments-list');

        if (!statsContainer || !listContainer) return;

        // Calculate stats
        const activeCount = this.deployments.filter(d => d.status === 'active').length;
        const totalRequests = this.deployments.reduce((sum, d) => sum + (d.request_count || 0), 0);

        // Render stats
        statsContainer.innerHTML = `
            <div class="col-md-3">
                <div class="card border-0 shadow-sm">
                    <div class="card-body">
                        <h6 class="text-muted mb-3">Active Deployments</h6>
                        <h2 class="fw-bold mb-2">${activeCount}</h2>
                        <p class="${activeCount > 0 ? 'text-success' : 'text-muted'} small mb-0">
                            ${activeCount > 0 ? 'All systems operational' : 'No active deployments'}
                        </p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card border-0 shadow-sm">
                    <div class="card-body">
                        <h6 class="text-muted mb-3">Total Requests</h6>
                        <h2 class="fw-bold mb-2">${totalRequests.toLocaleString()}</h2>
                        <p class="text-muted small mb-0">Total</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card border-0 shadow-sm">
                    <div class="card-body">
                        <h6 class="text-muted mb-3">Avg Response Time</h6>
                        <h2 class="fw-bold mb-2">--</h2>
                        <p class="text-muted small mb-0">Not available</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card border-0 shadow-sm">
                    <div class="card-body">
                        <h6 class="text-muted mb-3">Uptime</h6>
                        <h2 class="fw-bold mb-2">--</h2>
                        <p class="text-muted small mb-0">Not available</p>
                    </div>
                </div>
            </div>
        `;

        // Render deployments list
        if (this.deployments.length === 0) {
            listContainer.innerHTML = `
                <div class="text-center py-5">
                    <i class="bi bi-cloud-upload text-muted" style="font-size: 3rem;"></i>
                    <p class="text-muted mt-3">No deployments yet</p>
                    <p class="text-muted small">Deploy a trained model to start monitoring</p>
                    <button class="btn btn-primary mt-2">Deploy New Model</button>
                </div>
            `;
            return;
        }

        listContainer.innerHTML = this.deployments.map(deployment => `
            <div class="card mb-3">
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start mb-3">
                        <div>
                            <div class="d-flex align-items-center gap-2 mb-1">
                                <h5 class="fw-bold mb-0">${deployment.model_name || 'Unnamed Model'}</h5>
                                <span class="badge badge-${deployment.status === 'active' ? 'success' : 'secondary'}">${deployment.status}</span>
                            </div>
                            <p class="text-muted mb-0 small">${deployment.target_platform || 'Unknown Platform'}</p>
                        </div>
                        <div class="d-flex gap-2">
                            <button class="btn btn-outline-primary btn-sm">Configure</button>
                            <button class="btn btn-outline-secondary btn-sm">Logs</button>
                        </div>
                    </div>

                    <div class="row pt-3 border-top">
                        <div class="col-md-4">
                            <p class="text-muted small mb-1">Endpoint</p>
                            <p class="fw-medium mb-0">${deployment.endpoint_url || 'N/A'}</p>
                        </div>
                        <div class="col-md-4">
                            <p class="text-muted small mb-1">Created At</p>
                            <p class="fw-medium mb-0">${deployment.created_at ? new Date(deployment.created_at).toLocaleDateString() : 'N/A'}</p>
                        </div>
                        <div class="col-md-4">
                            <p class="text-muted small mb-1">Requests</p>
                            <p class="fw-medium mb-0">${deployment.request_count || 0}</p>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    render() {
        const targets = [
            {
                name: 'Edge Devices',
                description: 'Deploy to industrial IoT devices and embedded systems',
                icon: 'cpu',
                devices: ['NVIDIA Jetson', 'Intel NUC', 'Raspberry Pi 4'],
                color: 'primary'
            },
            {
                name: 'Cloud Platform',
                description: 'Scalable cloud deployment for high-volume inference',
                icon: 'cloud',
                devices: ['AWS', 'Google Cloud', 'Azure'],
                color: 'info'
            },
            {
                name: 'On-Premise',
                description: 'Private server deployment for maximum security',
                icon: 'server',
                devices: ['Docker', 'Kubernetes', 'Bare Metal'],
                color: 'success'
            },
            {
                name: 'Mobile',
                description: 'Deploy to iOS and Android applications',
                icon: 'phone',
                devices: ['iOS (CoreML)', 'Android (TFLite)'],
                color: 'warning'
            }
        ];

        return `
            <div class="min-vh-100 bg-light">
                <div class="container py-4">
                    <!-- Page Header -->
                    <div class="mb-4">
                        <h1 class="display-5 fw-bold mb-2">Deployment</h1>
                        <p class="text-muted">Deploy models to edge devices and cloud platforms</p>
                    </div>

                    <!-- Stats Cards -->
                    <div class="row g-4 mb-4" id="deployment-stats">
                        <div class="col-12 text-center py-3">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                        </div>
                    </div>

                    <!-- Active Deployments Card -->
                    <div class="card border-0 shadow-sm mb-4">
                        <div class="card-header bg-white">
                            <div class="row align-items-center">
                                <div class="col">
                                    <h5 class="mb-1 fw-bold">Active Deployments</h5>
                                    <p class="text-muted mb-0 small">Monitor and manage deployed models</p>
                                </div>
                                <div class="col-auto">
                                    <button class="btn btn-primary">Deploy New Model</button>
                                </div>
                            </div>
                        </div>
                        <div class="card-body" id="deployments-list">
                            <div class="text-center py-3">
                                <div class="spinner-border text-primary" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Deployment Targets Card -->
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-white">
                            <h5 class="mb-1 fw-bold">Deployment Targets</h5>
                            <p class="text-muted mb-0 small">Choose where to deploy your models</p>
                        </div>
                        <div class="card-body">
                            <div class="row g-4">
                                ${targets.map(target => `
                                    <div class="col-md-6 col-lg-3">
                                        <div class="card h-100 hover-shadow">
                                            <div class="card-body">
                                                <div class="icon-gradient d-flex align-items-center justify-content-center mb-3">
                                                    <i class="bi bi-${target.icon} text-white fs-3"></i>
                                                </div>
                                                <h6 class="fw-bold mb-2">${target.name}</h6>
                                                <p class="text-muted small mb-3">${target.description}</p>
                                                <div class="d-flex flex-wrap gap-1">
                                                    ${target.devices.map(device => `
                                                        <span class="badge bg-light text-dark border">${device}</span>
                                                    `).join('')}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
}
