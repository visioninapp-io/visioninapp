// Evaluation Page Component

class EvaluationPage {
    constructor() {
        this.evaluations = [];
        this.completedTrainings = [];
        this.selectedTrainingResults = null;
    }

    async init() {
        await Promise.all([
            this.loadEvaluations(),
            this.loadCompletedTrainings()
        ]);
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

    async loadCompletedTrainings() {
        try {
            const response = await window.apiService.get('/evaluation/completed-trainings');
            this.completedTrainings = response;
            this.renderCompletedTrainings();
        } catch (error) {
            console.error('Failed to load completed trainings:', error);
            this.completedTrainings = [];
            this.renderCompletedTrainings();
        }
    }

    async loadTrainingResults(modelKey) {
        try {
            const response = await window.apiService.get(`/evaluation/results/${modelKey}`);
            this.selectedTrainingResults = response;
            this.renderTrainingResults();
        } catch (error) {
            console.error('Failed to load training results:', error);
            alert('Failed to load training results. The results.csv file may not exist yet.');
        }
    }

    renderCompletedTrainings() {
        const container = document.getElementById('completed-trainings-list');
        if (!container) return;

        if (this.completedTrainings.length === 0) {
            container.innerHTML = `
                <div class="text-center py-3">
                    <p class="text-muted">No completed trainings available</p>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div class="list-group">
                ${this.completedTrainings.map(training => `
                    <button 
                        class="list-group-item list-group-item-action d-flex justify-content-between align-items-center"
                        onclick="evaluationPage.loadTrainingResults('${training.model_key}')"
                    >
                        <div>
                            <h6 class="mb-1">${training.name}</h6>
                            <small class="text-muted">${training.architecture || 'N/A'}</small>
                        </div>
                        <span class="badge bg-success">Completed</span>
                    </button>
                `).join('')}
            </div>
        `;
    }

    renderTrainingResults() {
        const container = document.getElementById('training-results-container');
        if (!container) return;

        if (!this.selectedTrainingResults) {
            container.innerHTML = '';
            return;
        }

        const { model_key, rows, columns } = this.selectedTrainingResults;

        if (!rows || rows.length === 0) {
            container.innerHTML = `
                <div class="alert alert-info">
                    No data available in results.csv
                </div>
            `;
            return;
        }

        // epoch 컬럼 찾기 (보통 첫 번째 컬럼)
        const epochCol = columns.find(c => c.toLowerCase().includes('epoch')) || columns[0];
        const epochs = rows.map(row => Math.round(row[epochCol]));

        // 메트릭 컬럼들 (숫자 컬럼만, epoch 제외)
        const metricCols = columns.filter(col => {
            return col !== epochCol && typeof rows[0][col] === 'number';
        });

        // 메트릭 카테고리 분류
        const lossMetrics = metricCols.filter(col => col.toLowerCase().includes('loss'));
        const evalMetrics = metricCols.filter(col => 
            col.toLowerCase().includes('precision') || 
            col.toLowerCase().includes('recall') || 
            col.toLowerCase().includes('map') ||
            col.toLowerCase().includes('metrics/')
        );
        const lrMetrics = metricCols.filter(col => col.toLowerCase().includes('lr'));
        const timeMetrics = metricCols.filter(col => col.toLowerCase().includes('time'));
        // 나머지 모든 메트릭 (어떤 카테고리에도 속하지 않는 것들, time은 테이블에만 표시)
        const otherMetrics = metricCols.filter(col => 
            !lossMetrics.includes(col) && 
            !evalMetrics.includes(col) && 
            !lrMetrics.includes(col) &&
            !timeMetrics.includes(col)
        );

        container.innerHTML = `
            <div class="card border-0 shadow-sm mb-4">
                <div class="card-header bg-white">
                    <h5 class="mb-1 fw-bold">Training Results: ${model_key}</h5>
                    <p class="text-muted mb-0 small">Epoch-by-epoch metrics from training</p>
                </div>
                <div class="card-body">
                    <!-- 탭 네비게이션 -->
                    <ul class="nav nav-tabs mb-3" id="metricsTabs" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="metrics-tab" data-bs-toggle="tab" data-bs-target="#metrics-pane" type="button" role="tab">
                                Evaluation Metrics
                            </button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="loss-tab" data-bs-toggle="tab" data-bs-target="#loss-pane" type="button" role="tab">
                                Loss
                            </button>
                        </li>
                        ${lrMetrics.length > 0 ? `
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="lr-tab" data-bs-toggle="tab" data-bs-target="#lr-pane" type="button" role="tab">
                                Learning Rate
                            </button>
                        </li>
                        ` : ''}
                        ${otherMetrics.length > 0 ? `
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="other-tab" data-bs-toggle="tab" data-bs-target="#other-pane" type="button" role="tab">
                                Other
                            </button>
                        </li>
                        ` : ''}
                    </ul>

                    <!-- 탭 컨텐츠 -->
                    <div class="tab-content" id="metricsTabContent">
                        <!-- Evaluation Metrics 탭 -->
                        <div class="tab-pane fade show active" id="metrics-pane" role="tabpanel">
                            <div class="mb-3">
                                <small class="text-muted">Select metrics to display:</small>
                                <div class="mt-2 d-flex flex-wrap gap-2" id="metrics-checkboxes">
                                    ${evalMetrics.map(col => `
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input metric-checkbox" type="checkbox" 
                                                value="${col}" id="check-${col}" 
                                                ${['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'].includes(col) ? 'checked' : ''}>
                                            <label class="form-check-label small" for="check-${col}">
                                                ${col.replace('metrics/', '').replace('(B)', '')}
                                            </label>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                            <div style="height: 400px;">
                                <canvas id="metricsChart"></canvas>
                            </div>
                        </div>

                        <!-- Loss 탭 -->
                        <div class="tab-pane fade" id="loss-pane" role="tabpanel">
                            <div class="mb-3">
                                <small class="text-muted">Select loss metrics to display:</small>
                                <div class="mt-2 d-flex flex-wrap gap-2" id="loss-checkboxes">
                                    ${lossMetrics.map(col => `
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input loss-checkbox" type="checkbox" 
                                                value="${col}" id="check-${col}" checked>
                                            <label class="form-check-label small" for="check-${col}">
                                                ${col}
                                            </label>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                            <div style="height: 400px;">
                                <canvas id="lossChart"></canvas>
                            </div>
                        </div>

                        ${lrMetrics.length > 0 ? `
                        <!-- Learning Rate 탭 -->
                        <div class="tab-pane fade" id="lr-pane" role="tabpanel">
                            <div class="mb-3">
                                <small class="text-muted">Select learning rate metrics to display:</small>
                                <div class="mt-2 d-flex flex-wrap gap-2" id="lr-checkboxes">
                                    ${lrMetrics.map(col => `
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input lr-checkbox" type="checkbox" 
                                                value="${col}" id="check-${col}" checked>
                                            <label class="form-check-label small" for="check-${col}">
                                                ${col}
                                            </label>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                            <div style="height: 400px;">
                                <canvas id="lrChart"></canvas>
                            </div>
                        </div>
                        ` : ''}

                        ${otherMetrics.length > 0 ? `
                        <!-- Other 탭 -->
                        <div class="tab-pane fade" id="other-pane" role="tabpanel">
                            <div class="mb-3">
                                <small class="text-muted">Select metrics to display:</small>
                                <div class="mt-2 d-flex flex-wrap gap-2" id="other-checkboxes">
                                    ${otherMetrics.map(col => `
                                        <div class="form-check form-check-inline">
                                            <input class="form-check-input other-checkbox" type="checkbox" 
                                                value="${col}" id="check-${col}" checked>
                                            <label class="form-check-label small" for="check-${col}">
                                                ${col}
                                            </label>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                            <div style="height: 400px;">
                                <canvas id="otherChart"></canvas>
                            </div>
                        </div>
                        ` : ''}
                    </div>
                    
                    <!-- 테이블 -->
                    <div class="table-responsive mt-4">
                        <table class="table table-sm table-hover">
                            <thead>
                                <tr>
                                    ${columns.map(col => {
                                        // 컬럼 헤더를 깔끔하게 포맷팅
                                        let header = col
                                            .replace('metrics/', '')
                                            .replace('(B)', '')
                                            .replace('train/', '')
                                            .replace('val/', '');
                                        return `<th>${header}</th>`;
                                    }).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                ${rows.map(row => `
                                    <tr>
                                        ${columns.map(col => {
                                            if (col === epochCol) {
                                                return `<td>${Math.round(row[col])}</td>`;
                                            }
                                            return `<td>${typeof row[col] === 'number' ? row[col].toFixed(4) : row[col]}</td>`;
                                        }).join('')}
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;

        // 차트 렌더링
        setTimeout(() => {
            this.renderResultsCharts(rows, columns, epochCol, {
                evalMetrics,
                lossMetrics,
                lrMetrics,
                otherMetrics
            });
        }, 100);
    }

    renderResultsCharts(rows, columns, epochCol, metricCategories) {
        const epochs = rows.map(row => row[epochCol]);
        
        // 색상 팔레트
        const colors = [
            'rgb(99, 102, 241)',   // indigo
            'rgb(16, 185, 129)',   // green
            'rgb(59, 130, 246)',   // blue
            'rgb(245, 158, 11)',   // yellow
            'rgb(239, 68, 68)',    // red
            'rgb(168, 85, 247)',   // purple
            'rgb(236, 72, 153)',   // pink
            'rgb(14, 165, 233)',   // sky
            'rgb(34, 197, 94)',    // emerald
            'rgb(251, 146, 60)'   // orange
        ];

        // Evaluation Metrics 차트
        this.renderCategoryChart('metricsChart', epochs, rows, metricCategories.evalMetrics, colors, 'Evaluation Metrics');
        
        // Loss 차트
        this.renderCategoryChart('lossChart', epochs, rows, metricCategories.lossMetrics, colors, 'Loss Metrics');
        
        // Learning Rate 차트
        if (metricCategories.lrMetrics.length > 0) {
            this.renderCategoryChart('lrChart', epochs, rows, metricCategories.lrMetrics, colors, 'Learning Rate');
        }
        
        // Other 차트
        if (metricCategories.otherMetrics.length > 0) {
            this.renderCategoryChart('otherChart', epochs, rows, metricCategories.otherMetrics, colors, 'Other Metrics');
        }

        // 체크박스 이벤트 리스너 추가
        this.setupChartCheckboxes(rows, epochs, metricCategories);
    }

    renderCategoryChart(canvasId, epochs, rows, metricCols, colors, title) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        // 기본적으로 체크된 메트릭만 표시
        const defaultMetrics = metricCols.filter(col => {
            const checkbox = document.getElementById(`check-${col}`);
            return checkbox && checkbox.checked;
        });

        if (defaultMetrics.length === 0 && metricCols.length > 0) {
            // 기본값이 없으면 첫 번째 메트릭만 표시
            defaultMetrics.push(metricCols[0]);
            const checkbox = document.getElementById(`check-${metricCols[0]}`);
            if (checkbox) checkbox.checked = true;
        }

        const datasets = defaultMetrics.map((col, idx) => ({
            label: col.replace('metrics/', '').replace('(B)', ''),
            data: rows.map(row => row[col]),
            borderColor: colors[idx % colors.length],
            backgroundColor: colors[idx % colors.length] + '20',
            tension: 0.3,
            fill: false,
            pointRadius: 2,
            pointHoverRadius: 4
        }));

        // 기존 차트가 있으면 제거
        if (window[`${canvasId}Instance`]) {
            window[`${canvasId}Instance`].destroy();
        }

        window[`${canvasId}Instance`] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            boxWidth: 12,
                            padding: 8,
                            font: {
                                size: 11
                            }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: title.includes('Rate') ? 'Learning Rate' : title.includes('Loss') ? 'Loss' : 'Value'
                        }
                    }
                }
            }
        });
    }

    setupChartCheckboxes(rows, epochs, metricCategories) {
        // Evaluation Metrics 체크박스
        document.querySelectorAll('.metric-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.updateChart('metricsChart', rows, epochs, metricCategories.evalMetrics);
            });
        });

        // Loss 체크박스
        document.querySelectorAll('.loss-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.updateChart('lossChart', rows, epochs, metricCategories.lossMetrics);
            });
        });

        // Learning Rate 체크박스
        document.querySelectorAll('.lr-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.updateChart('lrChart', rows, epochs, metricCategories.lrMetrics);
            });
        });

        // Other 체크박스
        document.querySelectorAll('.other-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.updateChart('otherChart', rows, epochs, metricCategories.otherMetrics);
            });
        });
    }

    updateChart(canvasId, rows, epochs, metricCols) {
        const colors = [
            'rgb(99, 102, 241)',
            'rgb(16, 185, 129)',
            'rgb(59, 130, 246)',
            'rgb(245, 158, 11)',
            'rgb(239, 68, 68)',
            'rgb(168, 85, 247)',
            'rgb(236, 72, 153)',
            'rgb(14, 165, 233)',
            'rgb(34, 197, 94)',
            'rgb(251, 146, 60)'
        ];

        const selectedMetrics = metricCols.filter(col => {
            const checkbox = document.getElementById(`check-${col}`);
            return checkbox && checkbox.checked;
        });

        const ctx = document.getElementById(canvasId);
        if (!ctx || !window[`${canvasId}Instance`]) return;

        const datasets = selectedMetrics.map((col, idx) => ({
            label: col.replace('metrics/', '').replace('(B)', ''),
            data: rows.map(row => row[col]),
            borderColor: colors[idx % colors.length],
            backgroundColor: colors[idx % colors.length] + '20',
            tension: 0.3,
            fill: false,
            pointRadius: 2,
            pointHoverRadius: 4
        }));

        window[`${canvasId}Instance`].data.datasets = datasets;
        window[`${canvasId}Instance`].update();
    }

    renderEvaluationsContent() {
        const container = document.getElementById('evaluations-content');
        if (!container) return;

        // 완료된 학습 목록 섹션
        const completedTrainingsSection = `
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-white">
                            <h5 class="mb-1 fw-bold">Completed Training Jobs</h5>
                            <p class="text-muted mb-0 small">Select a training to view detailed results from S3</p>
                        </div>
                        <div class="card-body">
                            <div id="completed-trainings-list">
                                <div class="text-center py-3">
                                    <div class="spinner-border spinner-border-sm text-primary" role="status">
                                        <span class="visually-hidden">Loading...</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div id="training-results-container"></div>
        `;

        if (this.evaluations.length === 0) {
            // 완료된 트레이닝이 있으면 선택 안내 메시지 표시
            if (this.completedTrainings.length > 0) {
                container.innerHTML = completedTrainingsSection + `
                    <div class="alert alert-info border-0 shadow-sm">
                        <div class="d-flex align-items-center">
                            <i class="bi bi-info-circle me-3" style="font-size: 1.5rem;"></i>
                            <div>
                                <h6 class="mb-1 fw-bold">View Training Results</h6>
                                <p class="mb-0 small">Select one of the completed trainings above to view detailed training results and metrics stored in S3.</p>
                            </div>
                        </div>
                    </div>
                `;
                setTimeout(() => this.renderCompletedTrainings(), 100);
                return;
            }
            
            // 완료된 트레이닝도 없으면 안내 메시지 표시
            container.innerHTML = completedTrainingsSection + `
                <div class="text-center py-5">
                    <i class="bi bi-clipboard-data text-muted" style="font-size: 3rem;"></i>
                    <p class="text-muted mt-3">No evaluation data available</p>
                    <p class="text-muted small">Train and evaluate a model to see metrics here</p>
                    <a href="#/training" class="btn btn-primary mt-2">Start Training</a>
                </div>
            `;
            // 완료된 학습 목록 렌더링
            setTimeout(() => this.renderCompletedTrainings(), 100);
            return;
        }

        // Get the latest evaluation
        const latest = this.evaluations[0];

        container.innerHTML = completedTrainingsSection + `
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
                                    <span class="text-muted">Test Dataset:</span>
                                    <span class="fw-medium">${latest.test_dataset_name || 'N/A'} (${latest.test_dataset_size || 0} images)</span>
                                </div>
                                <div class="d-flex justify-content-between mb-2">
                                    <span class="text-muted">Evaluated At:</span>
                                    <span class="fw-medium">${latest.created_at ? new Date(latest.created_at).toLocaleString() : 'N/A'}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Per-Class Metrics -->
            ${latest.class_metrics && latest.class_metrics.length > 0 ? `
                <div class="card border-0 shadow-sm mb-4">
                    <div class="card-header bg-white">
                        <h5 class="mb-1 fw-bold">Per-Class Performance</h5>
                        <p class="text-muted mb-0 small">Detailed metrics for each class</p>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-hover">
                                <thead>
                                    <tr>
                                        <th>Class</th>
                                        <th>Precision</th>
                                        <th>Recall</th>
                                        <th>F1-Score</th>
                                        <th>Support</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${latest.class_metrics.map(cm => `
                                        <tr>
                                            <td><strong>${cm.class || 'N/A'}</strong></td>
                                            <td>${(cm.precision * 100).toFixed(1)}%</td>
                                            <td>${(cm.recall * 100).toFixed(1)}%</td>
                                            <td>${(cm.f1 * 100).toFixed(1)}%</td>
                                            <td>${cm.support || 0}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            ` : ''}

            <!-- Confusion Matrix -->
            ${latest.confusion_matrix ? `
                <div class="card border-0 shadow-sm">
                    <div class="card-header bg-white">
                        <h5 class="mb-1 fw-bold">Confusion Matrix</h5>
                        <p class="text-muted mb-0 small">Prediction vs ground truth</p>
                    </div>
                    <div class="card-body">
                        <div id="confusion-matrix-container"></div>
                    </div>
                </div>
            ` : ''}
        `;

        // Initialize chart after rendering
        setTimeout(() => {
            this.initCharts(latest);
            if (latest.confusion_matrix) {
                this.renderConfusionMatrix(latest.confusion_matrix);
            }
            // 완료된 학습 목록 렌더링
            this.renderCompletedTrainings();
        }, 100);
    }

    renderConfusionMatrix(matrix) {
        const container = document.getElementById('confusion-matrix-container');
        if (!container || !matrix || !Array.isArray(matrix) || matrix.length === 0) return;

        // Assuming matrix is a 2D array
        const numClasses = matrix.length;
        const classes = Array.from({length: numClasses}, (_, i) => `Class ${i}`);

        let html = '<table class="table table-bordered text-center" style="max-width: 600px; margin: 0 auto;">';
        html += '<thead><tr><th></th>';

        // Header row (predicted)
        classes.forEach(cls => {
            html += `<th class="small">${cls}</th>`;
        });
        html += '</tr></thead><tbody>';

        // Data rows
        matrix.forEach((row, i) => {
            html += `<tr><th class="small">Actual ${classes[i]}</th>`;
            row.forEach((val, j) => {
                const color = i === j ? 'bg-success bg-opacity-25' : (val > 0 ? 'bg-danger bg-opacity-10' : '');
                html += `<td class="${color}"><strong>${val}</strong></td>`;
            });
            html += '</tr>';
        });

        html += '</tbody></table>';
        container.innerHTML = html;
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
