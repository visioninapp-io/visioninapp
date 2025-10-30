// Monitoring Page Component

class MonitoringPage {
    constructor() {
        this.monitoringData = [];
    }

    async init() {
        await this.loadMonitoringData();
    }

    async loadMonitoringData() {
        try {
            const response = await window.apiService.get('/monitoring');
            this.monitoringData = response;
            this.renderMonitoringContent();
        } catch (error) {
            console.error('Failed to load monitoring data:', error);
            this.monitoringData = [];
            this.renderMonitoringContent();
        }
    }

    renderMonitoringContent() {
        const statsContainer = document.getElementById('monitoring-stats');
        const alertsContainer = document.getElementById('alerts-list');

        if (!statsContainer || !alertsContainer) return;

        if (this.monitoringData.length === 0) {
            statsContainer.innerHTML = `
                <div class="col-12 text-center py-5">
                    <i class="bi bi-activity text-muted" style="font-size: 3rem;"></i>
                    <p class="text-muted mt-3">No monitoring data available</p>
                    <p class="text-muted small">Deploy a model to start monitoring</p>
                    <a href="#/deployment" class="btn btn-primary mt-2">Go to Deployment</a>
                </div>
            `;
            alertsContainer.innerHTML = '';
            return;
        }

        // Get latest monitoring data
        const latest = this.monitoringData[0];

        // Render stats
        statsContainer.innerHTML = `
            <div class="col-md-3">
                <div class="card border-0 shadow-sm">
                    <div class="card-body">
                        <h6 class="text-muted mb-3">Live Accuracy</h6>
                        <h2 class="fw-bold mb-2">${latest.accuracy ? (latest.accuracy * 100).toFixed(1) : '--'}%</h2>
                        <p class="text-muted small mb-0">Current performance</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card border-0 shadow-sm">
                    <div class="card-body">
                        <h6 class="text-muted mb-3">Avg Latency</h6>
                        <h2 class="fw-bold mb-2">${latest.avg_latency || '--'}ms</h2>
                        <p class="text-success small mb-0">Within SLA</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card border-0 shadow-sm">
                    <div class="card-body">
                        <h6 class="text-muted mb-3">Total Inferences</h6>
                        <h2 class="fw-bold mb-2">${(latest.total_requests || 0).toLocaleString()}</h2>
                        <p class="text-muted small mb-0">Total processed</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card border-0 shadow-sm">
                    <div class="card-body">
                        <h6 class="text-muted mb-3">Alerts</h6>
                        <h2 class="fw-bold mb-2">${latest.alert_count || 0}</h2>
                        <p class="${latest.alert_count > 0 ? 'text-warning' : 'text-muted'} small mb-0">
                            ${latest.alert_count > 0 ? 'Requires attention' : 'No alerts'}
                        </p>
                    </div>
                </div>
            </div>
        `;

        // Render alerts
        if (!latest.alerts || latest.alerts.length === 0) {
            alertsContainer.innerHTML = `
                <div class="text-center py-4">
                    <i class="bi bi-check-circle text-success" style="font-size: 2rem;"></i>
                    <p class="text-muted mt-3">No active alerts</p>
                    <p class="text-muted small">All systems are operating normally</p>
                </div>
            `;
        } else {
            alertsContainer.innerHTML = latest.alerts.map(alert => `
                <div class="card mb-3 border-${alert.severity === 'warning' ? 'warning' : 'info'} border-2">
                    <div class="card-body">
                        <div class="d-flex align-items-start gap-3">
                            <div class="text-${alert.severity === 'warning' ? 'warning' : 'info'} fs-4">
                                <i class="bi bi-${alert.severity === 'warning' ? 'exclamation-triangle' : 'info-circle'}"></i>
                            </div>
                            <div class="flex-grow-1">
                                <div class="d-flex align-items-center gap-2 mb-2">
                                    <h6 class="mb-0 fw-bold">${alert.message}</h6>
                                    <span class="badge bg-${alert.severity === 'warning' ? 'warning' : 'info'}">${alert.severity}</span>
                                </div>
                                <div class="d-flex gap-3 text-muted small">
                                    <span>Model: ${alert.model || 'Unknown'}</span>
                                    <span>${alert.time || 'Recently'}</span>
                                </div>
                            </div>
                            <div class="d-flex gap-2">
                                ${alert.severity === 'warning' ? `
                                    <button class="btn btn-primary btn-sm">Trigger Retraining</button>
                                ` : ''}
                                <button class="btn btn-outline-secondary btn-sm">View Details</button>
                            </div>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        // Initialize charts
        setTimeout(() => this.initCharts(latest), 100);
    }

    render() {
        return `
            <div class="min-vh-100 bg-light">
                <div class="container py-4">
                    <!-- Page Header -->
                    <div class="mb-4">
                        <h1 class="display-5 fw-bold mb-2">Monitoring & Feedback</h1>
                        <p class="text-muted">Track live inference results and model performance</p>
                    </div>

                    <!-- Stats Cards -->
                    <div class="row g-4 mb-4" id="monitoring-stats">
                        <div class="col-12 text-center py-3">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                        </div>
                    </div>

                    <!-- Charts -->
                    <div class="row g-4 mb-4">
                        <div class="col-lg-6">
                            <div class="card border-0 shadow-sm">
                                <div class="card-header bg-white">
                                    <h5 class="mb-1 fw-bold">Inference Volume & Latency</h5>
                                    <p class="text-muted mb-0 small">Real-time request metrics</p>
                                </div>
                                <div class="card-body">
                                    <canvas id="inferenceChart" height="100"></canvas>
                                </div>
                            </div>
                        </div>
                        <div class="col-lg-6">
                            <div class="card border-0 shadow-sm">
                                <div class="card-header bg-white">
                                    <h5 class="mb-1 fw-bold">Accuracy Drift Detection</h5>
                                    <p class="text-muted mb-0 small">Model performance over time</p>
                                </div>
                                <div class="card-body">
                                    <canvas id="accuracyChart" height="100"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Active Alerts Card -->
                    <div class="card border-0 shadow-sm mb-4">
                        <div class="card-header bg-white">
                            <div class="row align-items-center">
                                <div class="col">
                                    <h5 class="mb-1 fw-bold">Active Alerts</h5>
                                    <p class="text-muted mb-0 small">System notifications and recommendations</p>
                                </div>
                                <div class="col-auto">
                                    <button class="btn btn-outline-primary btn-sm" onclick="window.location.reload()">
                                        <i class="bi bi-arrow-clockwise me-1"></i> Refresh
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div class="card-body" id="alerts-list">
                            <div class="text-center py-3">
                                <div class="spinner-border text-primary" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Feedback Loop Configuration Card -->
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-white">
                            <h5 class="mb-1 fw-bold">Feedback Loop Configuration</h5>
                            <p class="text-muted mb-0 small">Automated retraining triggers and data collection</p>
                        </div>
                        <div class="card-body">
                            <div class="row g-4">
                                <div class="col-md-4">
                                    <div class="card h-100 border">
                                        <div class="card-body">
                                            <h6 class="fw-bold mb-3">Accuracy Threshold</h6>
                                            <h3 class="mb-2">93%</h3>
                                            <p class="text-muted small mb-3">Trigger retraining below this value</p>
                                            <button class="btn btn-outline-primary btn-sm w-100">Configure</button>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="card h-100 border">
                                        <div class="card-body">
                                            <h6 class="fw-bold mb-3">Data Collection</h6>
                                            <h3 class="mb-2">Active</h3>
                                            <p class="text-muted small mb-3">Collecting edge cases for improvement</p>
                                            <button class="btn btn-outline-primary btn-sm w-100">View Data</button>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="card h-100 border">
                                        <div class="card-body">
                                            <h6 class="fw-bold mb-3">Auto-Retrain</h6>
                                            <h3 class="mb-2">Enabled</h3>
                                            <p class="text-muted small mb-3">Scheduled weekly with new data</p>
                                            <button class="btn btn-outline-primary btn-sm w-100">Settings</button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    initCharts(data) {
        // Inference Volume Chart (placeholder with sample data)
        const inferenceCtx = document.getElementById('inferenceChart');
        if (inferenceCtx) {
            new Chart(inferenceCtx, {
                type: 'bar',
                data: {
                    labels: ['6h ago', '5h ago', '4h ago', '3h ago', '2h ago', '1h ago'],
                    datasets: [{
                        label: 'Requests',
                        data: [0, 0, 0, 0, 0, data.total_requests || 0],
                        backgroundColor: 'rgba(99, 102, 241, 0.8)',
                        yAxisID: 'y'
                    }, {
                        label: 'Avg Latency (ms)',
                        data: [0, 0, 0, 0, 0, data.avg_latency || 0],
                        borderColor: 'rgb(245, 158, 11)',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        type: 'line',
                        yAxisID: 'y1',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: { display: true, text: 'Requests' }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: { display: true, text: 'Latency (ms)' },
                            grid: { drawOnChartArea: false }
                        }
                    }
                }
            });
        }

        // Accuracy Drift Chart
        const accuracyCtx = document.getElementById('accuracyChart');
        if (accuracyCtx) {
            const accuracy = (data.accuracy || 0.9) * 100;
            new Chart(accuracyCtx, {
                type: 'line',
                data: {
                    labels: ['4 weeks ago', '3 weeks ago', '2 weeks ago', '1 week ago', 'Today'],
                    datasets: [{
                        label: 'Accuracy %',
                        data: [accuracy, accuracy, accuracy, accuracy, accuracy],
                        borderColor: 'rgb(16, 185, 129)',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            min: 85,
                            max: 100,
                            title: { display: true, text: 'Accuracy (%)' }
                        }
                    }
                }
            });
        }
    }
}
