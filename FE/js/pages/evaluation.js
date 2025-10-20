// Evaluation Page Component

class EvaluationPage {
    constructor() {
        this.evaluations = [];
    }

    async init() {
        await this.loadEvaluations();
    }

    async loadEvaluations() {
        try {
            const response = await window.apiService.get('/evaluation');
            this.evaluations = response;
            this.renderEvaluationsContent();
        } catch (error) {
            console.error('Failed to load evaluations:', error);
            this.evaluations = [];
            this.renderEvaluationsContent();
        }
    }

    renderEvaluationsContent() {
        const container = document.getElementById('evaluations-content');
        if (!container) return;

        if (this.evaluations.length === 0) {
            container.innerHTML = `
                <div class="text-center py-5">
                    <i class="bi bi-clipboard-data text-muted" style="font-size: 3rem;"></i>
                    <p class="text-muted mt-3">No evaluation data available</p>
                    <p class="text-muted small">Train and evaluate a model to see metrics here</p>
                    <a href="#/training" class="btn btn-primary mt-2">Start Training</a>
                </div>
            `;
            return;
        }

        // Get the latest evaluation
        const latest = this.evaluations[0];

        container.innerHTML = `
            <!-- Metrics Overview -->
            <div class="row g-4 mb-4">
                <div class="col-md-3">
                    <div class="card border-0 shadow-sm">
                        <div class="card-body">
                            <h6 class="text-muted mb-3">Precision</h6>
                            <h2 class="fw-bold mb-2">${(latest.precision * 100).toFixed(1)}%</h2>
                            <p class="text-muted small mb-0">Overall precision</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card border-0 shadow-sm">
                        <div class="card-body">
                            <h6 class="text-muted mb-3">Recall</h6>
                            <h2 class="fw-bold mb-2">${(latest.recall * 100).toFixed(1)}%</h2>
                            <p class="text-muted small mb-0">Overall recall</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card border-0 shadow-sm">
                        <div class="card-body">
                            <h6 class="text-muted mb-3">F1-Score</h6>
                            <h2 class="fw-bold mb-2">${(latest.f1_score * 100).toFixed(1)}%</h2>
                            <p class="text-muted small mb-0">Harmonic mean</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card border-0 shadow-sm">
                        <div class="card-body">
                            <h6 class="text-muted mb-3">mAP@0.5</h6>
                            <h2 class="fw-bold mb-2">${(latest.map_50 * 100).toFixed(1)}%</h2>
                            <p class="text-muted small mb-0">Mean average precision</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Charts -->
            <div class="row g-4 mb-4">
                <div class="col-lg-6">
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-white">
                            <h5 class="mb-1 fw-bold">Performance Metrics</h5>
                            <p class="text-muted mb-0 small">Model evaluation results</p>
                        </div>
                        <div class="card-body">
                            <canvas id="metricsChart" height="120"></canvas>
                        </div>
                    </div>
                </div>
                <div class="col-lg-6">
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-white">
                            <h5 class="mb-1 fw-bold">Evaluation Details</h5>
                            <p class="text-muted mb-0 small">Additional information</p>
                        </div>
                        <div class="card-body">
                            <div class="mb-3">
                                <div class="d-flex justify-content-between mb-2">
                                    <span class="text-muted">Model Name:</span>
                                    <span class="fw-medium">${latest.model_name || 'N/A'}</span>
                                </div>
                                <div class="d-flex justify-content-between mb-2">
                                    <span class="text-muted">Evaluated At:</span>
                                    <span class="fw-medium">${latest.created_at ? new Date(latest.created_at).toLocaleString() : 'N/A'}</span>
                                </div>
                                <div class="d-flex justify-content-between mb-2">
                                    <span class="text-muted">Status:</span>
                                    <span class="badge badge-success">${latest.status}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Initialize chart after rendering
        setTimeout(() => this.initCharts(latest), 100);
    }

    render() {
        return `
            <div class="min-vh-100 bg-light">
                <div class="container py-4">
                    <!-- Page Header -->
                    <div class="mb-4">
                        <h1 class="display-5 fw-bold mb-2">Model Evaluation</h1>
                        <p class="text-muted">Comprehensive performance metrics and model comparison</p>
                    </div>

                    <div id="evaluations-content">
                        <div class="text-center py-5">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                            <p class="text-muted mt-3">Loading evaluation data...</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    initCharts(evaluation) {
        // Metrics Chart
        const metricsCtx = document.getElementById('metricsChart');
        if (metricsCtx) {
            new Chart(metricsCtx, {
                type: 'bar',
                data: {
                    labels: ['Precision', 'Recall', 'F1-Score', 'mAP@0.5'],
                    datasets: [{
                        label: 'Score',
                        data: [
                            evaluation.precision * 100,
                            evaluation.recall * 100,
                            evaluation.f1_score * 100,
                            evaluation.map_50 * 100
                        ],
                        backgroundColor: [
                            'rgba(99, 102, 241, 0.8)',
                            'rgba(16, 185, 129, 0.8)',
                            'rgba(59, 130, 246, 0.8)',
                            'rgba(245, 158, 11, 0.8)'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Score (%)'
                            }
                        }
                    }
                }
            });
        }
    }
}
