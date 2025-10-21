// Training Page Component

class TrainingPage {
    constructor() {
        this.trainingJobs = [];
        this.selectedJob = null;
        this.refreshInterval = null;
    }

    async init() {
        await this.loadTrainingJobs();

        // Re-render the page after data is loaded
        const app = document.getElementById('app');
        if (app) {
            app.innerHTML = this.render();
            this.afterRender();
        }

        this.attachEventListeners();
        this.startAutoRefresh();
    }

    async loadTrainingJobs() {
        try {
            this.trainingJobs = await apiService.getTrainingJobs();
            // Find active training job for metrics display
            this.selectedJob = this.trainingJobs.find(j => j.status === 'running') || this.trainingJobs[0];
        } catch (error) {
            console.error('Error loading training jobs:', error);
            this.trainingJobs = [];
        }
    }

    render() {
        const activeJob = this.trainingJobs.find(j => j.status === 'running');
        const currentAccuracy = activeJob?.current_accuracy || 0;
        const currentLoss = activeJob?.current_loss || 0;
        const currentEpoch = activeJob?.current_epoch || 0;
        const totalEpochs = activeJob?.total_epochs || 0;
        const progress = totalEpochs > 0 ? Math.round((currentEpoch / totalEpochs) * 100) : 0;

        return `
            <div class="min-vh-100 bg-light">
                <div class="container py-4">
                    <!-- Page Header -->
                    <div class="mb-4">
                        <h1 class="display-5 fw-bold mb-2">Model Training</h1>
                        <p class="text-muted">Monitor training progress and performance metrics</p>
                    </div>

                    <div class="row g-4 mb-4">
                        <!-- Chart Card -->
                        <div class="col-lg-8">
                            <div class="card border-0 shadow-sm h-100">
                                <div class="card-header bg-white">
                                    <h5 class="mb-1 fw-bold">Training Metrics</h5>
                                    <p class="text-muted mb-0 small">Real-time loss and accuracy tracking</p>
                                </div>
                                <div class="card-body">
                                    <canvas id="trainingChart" height="100"></canvas>
                                </div>
                            </div>
                        </div>

                        <!-- Stats Cards -->
                        <div class="col-lg-4">
                            <div class="card border-0 shadow-sm mb-3">
                                <div class="card-body">
                                    <h6 class="text-muted mb-3">Current Accuracy</h6>
                                    <h2 class="fw-bold text-success mb-2">${currentAccuracy.toFixed(1)}%</h2>
                                    <div class="progress mb-2" style="height: 8px;">
                                        <div class="progress-bar bg-success" style="width: ${currentAccuracy}%"></div>
                                    </div>
                                    <p class="text-muted small mb-0">Epoch ${currentEpoch}/${totalEpochs}</p>
                                </div>
                            </div>

                            <div class="card border-0 shadow-sm mb-3">
                                <div class="card-body">
                                    <h6 class="text-muted mb-3">Training Loss</h6>
                                    <h2 class="fw-bold mb-2">${currentLoss.toFixed(3)}</h2>
                                    <p class="text-success small mb-0">
                                        <i class="bi bi-arrow-down"></i> Decreasing
                                    </p>
                                </div>
                            </div>

                            <div class="card border-0 shadow-sm">
                                <div class="card-body">
                                    <h6 class="text-muted mb-3">Active Jobs</h6>
                                    <h2 class="fw-bold mb-2">${this.trainingJobs.filter(j => j.status === 'running').length}</h2>
                                    <p class="text-muted small mb-0">Currently training</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Active Training Jobs -->
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-white">
                            <div class="row align-items-center">
                                <div class="col">
                                    <h5 class="mb-1 fw-bold">Training Jobs</h5>
                                    <p class="text-muted mb-0 small">Manage and monitor your model training pipelines</p>
                                </div>
                                <div class="col-auto">
                                    <button class="btn btn-primary" onclick="showStartTrainingModal()">
                                        <i class="bi bi-play-fill me-1"></i> Start New Training
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div class="card-body">
                            ${this.trainingJobs.length > 0 ? this.renderTrainingJobs() : this.renderEmptyState()}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderTrainingJobs() {
        return this.trainingJobs.map(job => {
            const progress = job.total_epochs > 0
                ? Math.round((job.current_epoch / job.total_epochs) * 100)
                : 0;

            const statusBadge = {
                'pending': 'bg-secondary',
                'running': 'bg-primary',
                'paused': 'bg-warning',
                'completed': 'bg-success',
                'failed': 'bg-danger'
            }[job.status] || 'bg-secondary';

            return `
                <div class="card mb-3">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start mb-3">
                            <div>
                                <h5 class="fw-bold mb-1">${job.name}</h5>
                                <p class="text-muted mb-0 small">Architecture: ${job.architecture || 'N/A'}</p>
                            </div>
                            <div class="d-flex gap-2">
                                <span class="badge ${statusBadge}">${job.status}</span>
                                ${job.status === 'running' ? `
                                    <button class="btn btn-outline-warning btn-sm" onclick="pauseTraining(${job.id})">
                                        <i class="bi bi-pause-fill"></i> Pause
                                    </button>
                                    <button class="btn btn-outline-danger btn-sm" onclick="stopTraining(${job.id})">
                                        <i class="bi bi-stop-fill"></i> Stop
                                    </button>
                                ` : job.status === 'paused' ? `
                                    <button class="btn btn-outline-primary btn-sm" onclick="resumeTraining(${job.id})">
                                        <i class="bi bi-play-fill"></i> Resume
                                    </button>
                                ` : job.status === 'completed' ? `
                                    <button class="btn btn-primary btn-sm" onclick="viewTrainingResults(${job.id})">
                                        <i class="bi bi-graph-up"></i> View Results
                                    </button>
                                ` : ''}
                            </div>
                        </div>

                        <div class="mb-3">
                            <div class="d-flex justify-content-between mb-2">
                                <span class="text-muted small">Progress</span>
                                <span class="fw-medium small">${job.current_epoch}/${job.total_epochs} epochs</span>
                            </div>
                            <div class="progress" style="height: 8px;">
                                <div class="progress-bar ${job.status === 'running' ? 'progress-bar-striped progress-bar-animated' : ''}"
                                     style="width: ${progress}%"></div>
                            </div>
                        </div>

                        <div class="row">
                            <div class="col-md-3">
                                <p class="text-muted small mb-1">Accuracy</p>
                                <p class="fw-medium mb-0 text-success">${(job.current_accuracy || 0).toFixed(2)}%</p>
                            </div>
                            <div class="col-md-3">
                                <p class="text-muted small mb-1">Loss</p>
                                <p class="fw-medium mb-0">${(job.current_loss || 0).toFixed(4)}</p>
                            </div>
                            <div class="col-md-3">
                                <p class="text-muted small mb-1">Learning Rate</p>
                                <p class="fw-medium mb-0 font-monospace small">
                                    ${job.hyperparameters ? JSON.parse(job.hyperparameters).learning_rate || '0.001' : '0.001'}
                                </p>
                            </div>
                            <div class="col-md-3">
                                <p class="text-muted small mb-1">Batch Size</p>
                                <p class="fw-medium mb-0">
                                    ${job.hyperparameters ? JSON.parse(job.hyperparameters).batch_size || 16 : 16}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    renderEmptyState() {
        return `
            <div class="text-center py-5">
                <i class="bi bi-robot text-muted mb-3" style="font-size: 5rem;"></i>
                <h5 class="text-muted">No Training Jobs Yet</h5>
                <p class="text-muted">Start training your first model to see it here</p>
                <button class="btn btn-primary" onclick="showStartTrainingModal()">
                    <i class="bi bi-play-fill me-1"></i> Start New Training
                </button>
            </div>
        `;
    }

    attachEventListeners() {
        // Event listeners are attached via onclick handlers in the HTML
    }

    startAutoRefresh() {
        // Refresh training jobs every 5 seconds
        this.refreshInterval = setInterval(async () => {
            if (this.trainingJobs.some(j => j.status === 'running')) {
                await this.loadTrainingJobs();
                this.updateUI();
            }
        }, 5000);
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
    }

    updateUI() {
        const app = document.getElementById('app');
        if (app) {
            app.innerHTML = this.render();
            this.afterRender();
        }
    }

    afterRender() {
        this.initChart();
    }

    async initChart() {
        if (!this.selectedJob) return;

        try {
            const metrics = await apiService.getTrainingMetrics(this.selectedJob.id);

            const ctx = document.getElementById('trainingChart');
            if (!ctx) return;

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: metrics.map((m, idx) => `Epoch ${idx + 1}`),
                    datasets: [{
                        label: 'Loss',
                        data: metrics.map(m => m.loss),
                        borderColor: 'rgb(239, 68, 68)',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        yAxisID: 'y',
                        tension: 0.4
                    }, {
                        label: 'Accuracy (%)',
                        data: metrics.map(m => m.accuracy * 100),
                        borderColor: 'rgb(16, 185, 129)',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        yAxisID: 'y1',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Loss'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Accuracy (%)'
                            },
                            grid: {
                                drawOnChartArea: false,
                            },
                        },
                    }
                }
            });
        } catch (error) {
            console.error('Error loading training metrics:', error);
        }
    }
}

// Start Training Modal
function showStartTrainingModal() {
    const modalHTML = `
        <div class="modal fade" id="startTrainingModal" tabindex="-1" aria-hidden="true">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="bi bi-play-fill me-2"></i>Start New Training
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <form id="start-training-form">
                            <!-- Training Job Name -->
                            <div class="mb-3">
                                <label for="training-name" class="form-label">Training Job Name *</label>
                                <input type="text" class="form-control" id="training-name" required
                                       placeholder="e.g., DefectNet-v1">
                            </div>

                            <!-- Dataset Selection -->
                            <div class="mb-3">
                                <label for="dataset-select" class="form-label">Dataset *</label>
                                <select class="form-select" id="dataset-select" required>
                                    <option value="">-- Select Dataset --</option>
                                </select>
                                <small class="text-muted">Choose the dataset to train on</small>
                            </div>

                            <!-- Model Architecture -->
                            <div class="mb-3">
                                <label for="architecture-select" class="form-label">Model Architecture *</label>
                                <select class="form-select" id="architecture-select" required>
                                    <option value="">-- Select Architecture --</option>
                                    <option value="yolov8n">YOLOv8 Nano (Fastest)</option>
                                    <option value="yolov8s">YOLOv8 Small</option>
                                    <option value="yolov8m">YOLOv8 Medium</option>
                                    <option value="yolov8l">YOLOv8 Large</option>
                                    <option value="yolov8x">YOLOv8 XLarge (Best Accuracy)</option>
                                    <option value="faster-rcnn">Faster R-CNN</option>
                                    <option value="ssd">SSD MobileNet</option>
                                    <option value="efficientdet">EfficientDet</option>
                                </select>
                            </div>

                            <!-- Hyperparameters -->
                            <div class="card bg-light mb-3">
                                <div class="card-header bg-transparent">
                                    <h6 class="mb-0">Hyperparameters</h6>
                                </div>
                                <div class="card-body">
                                    <div class="row g-3">
                                        <div class="col-md-6">
                                            <label for="epochs" class="form-label">Epochs</label>
                                            <input type="number" class="form-control" id="epochs"
                                                   value="100" min="1" max="1000">
                                            <small class="text-muted">Number of training iterations</small>
                                        </div>
                                        <div class="col-md-6">
                                            <label for="batch-size" class="form-label">Batch Size</label>
                                            <select class="form-select" id="batch-size">
                                                <option value="8">8</option>
                                                <option value="16" selected>16</option>
                                                <option value="32">32</option>
                                                <option value="64">64</option>
                                            </select>
                                            <small class="text-muted">Images per batch</small>
                                        </div>
                                        <div class="col-md-6">
                                            <label for="learning-rate" class="form-label">Learning Rate</label>
                                            <input type="number" class="form-control" id="learning-rate"
                                                   value="0.001" step="0.0001" min="0.00001" max="1">
                                            <small class="text-muted">Optimizer learning rate</small>
                                        </div>
                                        <div class="col-md-6">
                                            <label for="image-size" class="form-label">Image Size</label>
                                            <select class="form-select" id="image-size">
                                                <option value="416">416x416</option>
                                                <option value="512">512x512</option>
                                                <option value="640" selected>640x640</option>
                                                <option value="800">800x800</option>
                                                <option value="1024">1024x1024</option>
                                            </select>
                                            <small class="text-muted">Input image resolution</small>
                                        </div>
                                        <div class="col-md-6">
                                            <label for="optimizer" class="form-label">Optimizer</label>
                                            <select class="form-select" id="optimizer">
                                                <option value="adam" selected>Adam</option>
                                                <option value="sgd">SGD</option>
                                                <option value="adamw">AdamW</option>
                                            </select>
                                        </div>
                                        <div class="col-md-6">
                                            <label for="momentum" class="form-label">Momentum</label>
                                            <input type="number" class="form-control" id="momentum"
                                                   value="0.9" step="0.01" min="0" max="1">
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Advanced Options -->
                            <div class="accordion mb-3" id="advancedOptions">
                                <div class="accordion-item">
                                    <h2 class="accordion-header">
                                        <button class="accordion-button collapsed" type="button"
                                                data-bs-toggle="collapse" data-bs-target="#collapseAdvanced">
                                            Advanced Options
                                        </button>
                                    </h2>
                                    <div id="collapseAdvanced" class="accordion-collapse collapse"
                                         data-bs-parent="#advancedOptions">
                                        <div class="accordion-body">
                                            <div class="row g-3">
                                                <div class="col-md-6">
                                                    <label for="weight-decay" class="form-label">Weight Decay</label>
                                                    <input type="number" class="form-control" id="weight-decay"
                                                           value="0.0005" step="0.0001" min="0" max="1">
                                                </div>
                                                <div class="col-md-6">
                                                    <label for="warmup-epochs" class="form-label">Warmup Epochs</label>
                                                    <input type="number" class="form-control" id="warmup-epochs"
                                                           value="3" min="0" max="10">
                                                </div>
                                                <div class="col-12">
                                                    <div class="form-check form-switch">
                                                        <input class="form-check-input" type="checkbox"
                                                               id="use-pretrained" checked>
                                                        <label class="form-check-label" for="use-pretrained">
                                                            Use pre-trained weights
                                                        </label>
                                                    </div>
                                                </div>
                                                <div class="col-12">
                                                    <div class="form-check form-switch">
                                                        <input class="form-check-input" type="checkbox"
                                                               id="data-augmentation" checked>
                                                        <label class="form-check-label" for="data-augmentation">
                                                            Enable data augmentation
                                                        </label>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </form>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" id="start-training-btn">
                            <i class="bi bi-play-fill me-1"></i> Start Training
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    const existingModal = document.getElementById('startTrainingModal');
    if (existingModal) existingModal.remove();

    document.body.insertAdjacentHTML('beforeend', modalHTML);
    const modal = new bootstrap.Modal(document.getElementById('startTrainingModal'));

    // Load datasets
    loadDatasetsForTraining();

    // Handle form submission
    document.getElementById('start-training-btn').addEventListener('click', handleStartTraining);

    modal.show();
}

async function loadDatasetsForTraining() {
    try {
        const datasets = await apiService.getDatasets();
        const select = document.getElementById('dataset-select');

        datasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.id;
            option.textContent = `${dataset.name} (${dataset.total_images || 0} images)`;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading datasets:', error);
    }
}

async function handleStartTraining() {
    const name = document.getElementById('training-name').value;
    const datasetId = document.getElementById('dataset-select').value;
    const architecture = document.getElementById('architecture-select').value;

    if (!name || !datasetId || !architecture) {
        showToast('Please fill in all required fields', 'error');
        return;
    }

    const hyperparameters = {
        epochs: parseInt(document.getElementById('epochs').value),
        batch_size: parseInt(document.getElementById('batch-size').value),
        learning_rate: parseFloat(document.getElementById('learning-rate').value),
        img_size: parseInt(document.getElementById('image-size').value),
        optimizer: document.getElementById('optimizer').value,
        momentum: parseFloat(document.getElementById('momentum').value),
        weight_decay: parseFloat(document.getElementById('weight-decay').value),
        warmup_epochs: parseInt(document.getElementById('warmup-epochs').value),
        use_pretrained: document.getElementById('use-pretrained').checked,
        data_augmentation: document.getElementById('data-augmentation').checked
    };

    try {
        const trainingData = {
            name,
            dataset_id: parseInt(datasetId),
            architecture,
            hyperparameters: hyperparameters  // Send as object, not JSON string
        };

        const result = await apiService.startTraining(trainingData);

        bootstrap.Modal.getInstance(document.getElementById('startTrainingModal')).hide();
        showToast('Training started successfully!', 'success');

        setTimeout(() => window.location.reload(), 1000);
    } catch (error) {
        console.error('Error starting training:', error);
        showToast('Failed to start training: ' + error.message, 'error');
    }
}

// Training control functions
async function pauseTraining(jobId) {
    try {
        await apiService.pauseTraining(jobId);
        showToast('Training paused', 'info');
        setTimeout(() => window.location.reload(), 1000);
    } catch (error) {
        showToast('Failed to pause training: ' + error.message, 'error');
    }
}

async function stopTraining(jobId) {
    if (confirm('Are you sure you want to stop this training? This cannot be undone.')) {
        try {
            await apiService.stopTraining(jobId);
            showToast('Training stopped', 'info');
            setTimeout(() => window.location.reload(), 1000);
        } catch (error) {
            showToast('Failed to stop training: ' + error.message, 'error');
        }
    }
}

function resumeTraining(jobId) {
    showToast('Resume training feature coming soon', 'info');
}

function viewTrainingResults(jobId) {
    window.location.hash = `#/evaluation/${jobId}`;
}
