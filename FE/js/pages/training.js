// Training Page Component - Real-time Training Monitoring

class TrainingPage {
    constructor() {
        this.trainingJobs = [];
        this.selectedJob = null;
        this.metricsInterval = null;
        this.chart = null;
        this.isPolling = false;
    }

    async init() {
        console.log('[Training Page] Initializing...');

        try {
            await this.loadTrainingJobs();

            // Render the page
            const app = document.getElementById('app');
            if (app) {
                app.innerHTML = this.render();
                await this.afterRender();
            }

            this.attachEventListeners();
            this.startConditionalPolling();

            console.log('[Training Page] Initialized successfully');
        } catch (error) {
            console.error('[Training Page] Initialization error:', error);
            showToast('Failed to load training page: ' + error.message, 'error');
        }
    }

    async loadTrainingJobs() {
        try {
            console.log('[Training Page] Loading training jobs...');
            const jobs = await apiService.getTrainingJobs();

            this.trainingJobs = jobs || [];
            console.log('[Training Page] Loaded jobs:', this.trainingJobs.length);

            this.trainingJobs.forEach(job => {
                console.log(`  Job ${job.id}: ${job.name} - Status: ${job.status}, Epoch: ${job.current_epoch || 0}/${job.total_epochs || 0}`);
            });

            // Select a job to display metrics for
            if (!this.selectedJob) {
                this.selectedJob = this.trainingJobs.find(j => j.status === 'running') ||
                                  this.trainingJobs.find(j => j.status === 'completed') ||
                                  this.trainingJobs[0];
            } else {
                // Update selected job data
                const updated = this.trainingJobs.find(j => j.id === this.selectedJob.id);
                if (updated) {
                    this.selectedJob = updated;
                }
            }

            if (this.selectedJob) {
                console.log('[Training Page] Selected job:', this.selectedJob.id, this.selectedJob.name);
            }

        } catch (error) {
            console.error('[Training Page] Error loading jobs:', error);
            this.trainingJobs = [];
            throw error;
        }
    }

    render() {
        console.log('[Training Page] Rendering with', this.trainingJobs.length, 'jobs');

        // Get data for selected job
        const selectedJob = this.selectedJob || this.trainingJobs[0];
        const currentAccuracy = selectedJob?.current_accuracy || 0;
        const currentLoss = selectedJob?.current_loss || 0;
        const currentEpoch = selectedJob?.current_epoch || 0;
        const totalEpochs = selectedJob?.total_epochs || 0;
        const activeJobsCount = this.trainingJobs.filter(j => j.status === 'running').length;

        return `
            <div class="min-vh-100 bg-light">
                <div class="container py-4">
                    <!-- Page Header -->
                    <div class="mb-4">
                        <h1 class="display-5 fw-bold mb-2">Model Training</h1>
                        <p class="text-muted">Monitor training progress and performance metrics in real-time</p>
                    </div>

                    <!-- Job Selector -->
                    ${this.trainingJobs.length > 0 ? `
                    <div class="card border-0 shadow-sm mb-4">
                        <div class="card-body">
                            <div class="row align-items-center">
                                <div class="col-md-6">
                                    <label class="form-label fw-bold mb-2">Select Training Job:</label>
                                    <select class="form-select" id="job-selector" onchange="window.trainingPage.selectJob(this.value)">
                                        ${this.trainingJobs.map(job => `
                                            <option value="${job.id}" ${selectedJob?.id === job.id ? 'selected' : ''}>
                                                ${job.name} - ${job.status.toUpperCase()} (${job.current_epoch || 0}/${job.total_epochs || 0} epochs)
                                            </option>
                                        `).join('')}
                                    </select>
                                </div>
                                <div class="col-md-6 text-end">
                                    <button class="btn btn-primary" onclick="window.trainingPage.showStartTrainingModal()">
                                        <i class="bi bi-play-fill me-1"></i> Start New Training
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    ` : ''}

                    <div class="row g-4 mb-4">
                        <!-- Stats Cards -->
                        <div class="col-lg-4">
                            <div class="card border-0 shadow-sm mb-3">
                                <div class="card-body">
                                    <h6 class="text-muted mb-3">Current Accuracy</h6>
                                    <h2 class="fw-bold text-success mb-2" id="live-accuracy">${currentAccuracy.toFixed(1)}%</h2>
                                    <div class="progress mb-2" style="height: 8px;">
                                        <div class="progress-bar bg-success" id="accuracy-bar" style="width: ${currentAccuracy}%"></div>
                                    </div>
                                    <p class="text-muted small mb-0" id="live-epoch">Epoch ${currentEpoch}/${totalEpochs}</p>
                                </div>
                            </div>

                            <div class="card border-0 shadow-sm mb-3">
                                <div class="card-body">
                                    <h6 class="text-muted mb-3">Training Loss</h6>
                                    <h2 class="fw-bold mb-2" id="live-loss">${currentLoss.toFixed(4)}</h2>
                                    <p class="text-success small mb-0">
                                        <i class="bi bi-arrow-down"></i> ${selectedJob?.status === 'running' ? 'Decreasing' : 'Final'}
                                    </p>
                                </div>
                            </div>

                            <div class="card border-0 shadow-sm">
                                <div class="card-body">
                                    <h6 class="text-muted mb-3">Active Jobs</h6>
                                    <h2 class="fw-bold mb-2" id="active-jobs-count">${activeJobsCount}</h2>
                                    <p class="text-muted small mb-0">Currently training</p>
                                </div>
                            </div>
                        </div>

                        <!-- Chart Card -->
                        <div class="col-lg-8">
                            <div class="card border-0 shadow-sm h-100">
                                <div class="card-header bg-white">
                                    <h5 class="mb-1 fw-bold">Training Metrics - ${selectedJob?.name || 'No Job Selected'}</h5>
                                    <p class="text-muted mb-0 small">
                                        Real-time loss and accuracy tracking
                                        ${selectedJob?.status === 'running' ? '<span class="badge bg-primary ms-2">LIVE</span>' : ''}
                                    </p>
                                </div>
                                <div class="card-body">
                                    <canvas id="trainingChart" height="100"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Training Jobs List -->
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-white">
                            <div class="row align-items-center">
                                <div class="col">
                                    <h5 class="mb-1 fw-bold">All Training Jobs</h5>
                                    <p class="text-muted mb-0 small">View and manage all training pipelines</p>
                                </div>
                            </div>
                        </div>
                        <div class="card-body" id="jobs-container">
                            ${this.trainingJobs.length > 0 ? this.renderTrainingJobs() : this.renderEmptyState()}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderTrainingJobs() {
        if (!this.trainingJobs || this.trainingJobs.length === 0) {
            return this.renderEmptyState();
        }

        return this.trainingJobs.map(job => {
            const progress = job.total_epochs > 0
                ? Math.round((job.current_epoch || 0) / job.total_epochs * 100)
                : 0;

            const statusBadge = {
                'pending': 'bg-secondary',
                'running': 'bg-primary',
                'paused': 'bg-warning',
                'completed': 'bg-success',
                'failed': 'bg-danger'
            }[job.status] || 'bg-secondary';

            const isSelected = this.selectedJob?.id === job.id;

            // Parse hyperparameters safely
            let hyperparams = {};
            if (job.hyperparameters) {
                if (typeof job.hyperparameters === 'string') {
                    try {
                        hyperparams = JSON.parse(job.hyperparameters);
                    } catch (e) {
                        console.warn('Failed to parse hyperparameters:', e);
                    }
                } else {
                    hyperparams = job.hyperparameters;
                }
            }

            return `
                <div class="card mb-3 ${isSelected ? 'border-primary' : ''}" style="cursor: pointer;" onclick="window.trainingPage.selectJob(${job.id})">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start mb-3">
                            <div>
                                <h5 class="fw-bold mb-1">
                                    ${job.name}
                                    ${isSelected ? '<i class="bi bi-check-circle-fill text-primary ms-2"></i>' : ''}
                                </h5>
                                <p class="text-muted mb-0 small">Architecture: ${job.architecture || 'N/A'}</p>
                            </div>
                            <span class="badge ${statusBadge}">${job.status.toUpperCase()}</span>
                        </div>

                        <div class="mb-3">
                            <div class="d-flex justify-content-between mb-2">
                                <span class="text-muted small">Progress</span>
                                <span class="fw-medium small">${job.current_epoch || 0}/${job.total_epochs || 0} epochs</span>
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
                                <p class="fw-medium mb-0 font-monospace small">${hyperparams.learning_rate || '0.001'}</p>
                            </div>
                            <div class="col-md-3">
                                <p class="text-muted small mb-1">Batch Size</p>
                                <p class="fw-medium mb-0">${hyperparams.batch_size || 16}</p>
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
                <button class="btn btn-primary" onclick="window.trainingPage.showStartTrainingModal()">
                    <i class="bi bi-play-fill me-1"></i> Start New Training
                </button>
            </div>
        `;
    }

    attachEventListeners() {
        // Event listeners are handled via onclick in HTML
    }

    selectJob(jobId) {
        const id = parseInt(jobId);
        const job = this.trainingJobs.find(j => j.id === id);

        if (job) {
            console.log('[Training Page] Selected job:', job.id, job.name);
            this.selectedJob = job;

            // Re-render the page
            const app = document.getElementById('app');
            if (app) {
                app.innerHTML = this.render();
                this.afterRender();
            }

            // Restart polling if needed
            this.startConditionalPolling();
        }
    }

    startConditionalPolling() {
        console.log('[Training Page] Starting conditional polling...');

        // Clear any existing interval
        if (this.metricsInterval) {
            clearInterval(this.metricsInterval);
            this.metricsInterval = null;
        }

        // Check if any job is running
        const hasRunningJobs = this.trainingJobs.some(j => j.status === 'running');

        if (hasRunningJobs) {
            console.log('[Training Page] Running jobs detected - starting 1-second polling');
            this.isPolling = true;

            this.metricsInterval = setInterval(async () => {
                try {
                    await this.updateRealTimeData();
                } catch (error) {
                    console.error('[Training Page] Real-time update error:', error);
                }
            }, 1000); // 1 second polling only when training
        } else {
            console.log('[Training Page] No running jobs - polling stopped');
            this.isPolling = false;
        }
    }

    async updateRealTimeData() {
        try {
            // Reload jobs data
            const jobs = await apiService.getTrainingJobs();
            this.trainingJobs = jobs || [];

            // Check if any job is still running
            const hasRunningJobs = this.trainingJobs.some(j => j.status === 'running');

            // If no running jobs, stop polling
            if (!hasRunningJobs && this.isPolling) {
                console.log('[Training Page] All jobs completed - stopping polling');
                this.stopPolling();

                // Reload page to show final state
                const app = document.getElementById('app');
                if (app) {
                    app.innerHTML = this.render();
                    await this.afterRender();
                }
                return;
            }

            // Update selected job if it's running
            if (this.selectedJob) {
                const updatedSelectedJob = this.trainingJobs.find(j => j.id === this.selectedJob.id);
                if (updatedSelectedJob) {
                    this.selectedJob = updatedSelectedJob;

                    // Update live stats
                    const accuracyEl = document.getElementById('live-accuracy');
                    const lossEl = document.getElementById('live-loss');
                    const epochEl = document.getElementById('live-epoch');
                    const accuracyBar = document.getElementById('accuracy-bar');
                    const activeCount = document.getElementById('active-jobs-count');

                    if (accuracyEl) accuracyEl.textContent = `${(this.selectedJob.current_accuracy || 0).toFixed(1)}%`;
                    if (lossEl) lossEl.textContent = (this.selectedJob.current_loss || 0).toFixed(4);
                    if (epochEl) epochEl.textContent = `Epoch ${this.selectedJob.current_epoch || 0}/${this.selectedJob.total_epochs || 0}`;
                    if (accuracyBar) accuracyBar.style.width = `${this.selectedJob.current_accuracy || 0}%`;
                    if (activeCount) activeCount.textContent = this.trainingJobs.filter(j => j.status === 'running').length;

                    // Update chart if selected job is running
                    if (this.selectedJob.status === 'running') {
                        await this.updateChart();
                    }
                }
            }

            // Update jobs list every 5 seconds (less frequent to avoid flicker)
            const now = Date.now();
            if (!this.lastFullUpdate || now - this.lastFullUpdate > 5000) {
                this.lastFullUpdate = now;
                const jobsContainer = document.getElementById('jobs-container');
                if (jobsContainer) {
                    jobsContainer.innerHTML = this.renderTrainingJobs();
                }

                // Update job selector
                const selector = document.getElementById('job-selector');
                if (selector) {
                    const currentValue = selector.value;
                    selector.innerHTML = this.trainingJobs.map(job => `
                        <option value="${job.id}" ${job.id === parseInt(currentValue) ? 'selected' : ''}>
                            ${job.name} - ${job.status.toUpperCase()} (${job.current_epoch || 0}/${job.total_epochs || 0} epochs)
                        </option>
                    `).join('');
                }
            }

        } catch (error) {
            console.error('[Training Page] Error updating real-time data:', error);
        }
    }

    stopPolling() {
        if (this.metricsInterval) {
            clearInterval(this.metricsInterval);
            this.metricsInterval = null;
        }
        this.isPolling = false;
        console.log('[Training Page] Polling stopped');
    }

    async afterRender() {
        await this.initChart();
    }

    async initChart() {
        if (!this.selectedJob) {
            console.log('[Training Page] No selected job for chart');
            return;
        }

        try {
            console.log('[Training Page] Initializing chart for job', this.selectedJob.id);
            const metrics = await apiService.getTrainingMetrics(this.selectedJob.id);
            console.log('[Training Page] Loaded metrics:', metrics?.length || 0);

            const ctx = document.getElementById('trainingChart');
            if (!ctx) {
                console.warn('[Training Page] Chart canvas not found');
                return;
            }

            // Destroy existing chart
            if (this.chart) {
                this.chart.destroy();
                this.chart = null;
            }

            // Create chart only if we have metrics
            if (!metrics || metrics.length === 0) {
                console.log('[Training Page] No metrics to display');
                const context = ctx.getContext('2d');
                context.fillStyle = '#6c757d';
                context.font = '16px sans-serif';
                context.fillText('No metrics available yet. Metrics will appear once training starts.', 50, 100);
                return;
            }

            this.chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: metrics.map((m, idx) => `Epoch ${m.epoch || idx + 1}`),
                    datasets: [{
                        label: 'Loss',
                        data: metrics.map(m => m.train_loss || 0),
                        borderColor: 'rgb(239, 68, 68)',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        yAxisID: 'y',
                        tension: 0.4
                    }, {
                        label: 'Accuracy (%)',
                        data: metrics.map(m => (m.train_accuracy || 0)),
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

            console.log('[Training Page] Chart initialized successfully');

        } catch (error) {
            console.error('[Training Page] Error initializing chart:', error);
        }
    }

    async updateChart() {
        if (!this.chart || !this.selectedJob) return;

        try {
            const metrics = await apiService.getTrainingMetrics(this.selectedJob.id);
            if (!metrics || metrics.length === 0) return;

            // Update chart data
            this.chart.data.labels = metrics.map((m, idx) => `Epoch ${m.epoch || idx + 1}`);
            this.chart.data.datasets[0].data = metrics.map(m => m.train_loss || 0);
            this.chart.data.datasets[1].data = metrics.map(m => (m.train_accuracy || 0));
            this.chart.update('none');
        } catch (error) {
            console.error('[Training Page] Error updating chart:', error);
        }
    }

    cleanup() {
        console.log('[Training Page] Cleaning up...');
        this.stopPolling();
        if (this.chart) {
            this.chart.destroy();
            this.chart = null;
        }
    }

    // Modal and control methods
    showStartTrainingModal() {
        const modalHTML = `
            <div class="modal fade" id="startTrainingModal" tabindex="-1">
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
                                <div class="mb-3">
                                    <label class="form-label">Training Job Name *</label>
                                    <input type="text" class="form-control" id="training-name" required
                                           placeholder="e.g., ProductDefect-Training-v1">
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Dataset *</label>
                                    <select class="form-select" id="dataset-select" required>
                                        <option value="">-- Select Dataset --</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Select Model to Train *</label>
                                    <select class="form-select" id="model-select" required>
                                        <option value="">-- Select Model --</option>
                                    </select>
                                    <small class="text-muted">Choose an existing model or create a new one</small>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Model Architecture *</label>
                                    <select class="form-select" id="architecture-select" required>
                                        <option value="">-- Select Architecture --</option>
                                        <optgroup label="Object Detection (YOLO)">
                                            <option value="yolov8n">YOLOv8 Nano (Fastest, Smallest)</option>
                                            <option value="yolov8s">YOLOv8 Small</option>
                                            <option value="yolov8m">YOLOv8 Medium</option>
                                            <option value="yolov8l">YOLOv8 Large</option>
                                            <option value="yolov8x">YOLOv8 XLarge (Most Accurate)</option>
                                        </optgroup>
                                        <optgroup label="Image Classification (PyTorch)">
                                            <option value="resnet18">ResNet18 (Fast)</option>
                                            <option value="resnet50">ResNet50</option>
                                            <option value="mobilenet_v2">MobileNet V2</option>
                                        </optgroup>
                                    </select>
                                    <small class="text-muted">YOLO for object detection, ResNet/MobileNet for classification</small>
                                </div>
                                <div class="row">
                                    <div class="col-md-3">
                                        <label class="form-label">Epochs</label>
                                        <input type="number" class="form-control" id="epochs" value="20" min="1">
                                    </div>
                                    <div class="col-md-3">
                                        <label class="form-label">Batch Size</label>
                                        <input type="number" class="form-control" id="batch-size" value="16" min="1">
                                    </div>
                                    <div class="col-md-3">
                                        <label class="form-label">Image Size</label>
                                        <input type="number" class="form-control" id="img-size" value="640" step="32" min="320">
                                        <small class="text-muted">For YOLO</small>
                                    </div>
                                    <div class="col-md-3">
                                        <label class="form-label">Learning Rate</label>
                                        <input type="number" class="form-control" id="learning-rate" value="0.001" step="0.0001">
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

        // Load datasets and models
        this.loadDatasetsForModal();
        this.loadModelsForModal();

        // Handle form submission
        document.getElementById('start-training-btn').addEventListener('click', () => this.handleStartTraining());

        modal.show();
    }

    async loadDatasetsForModal() {
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

    async loadModelsForModal() {
        try {
            const models = await apiService.get('/models/');
            const select = document.getElementById('model-select');

            // Add "Create New Model" option
            const newOption = document.createElement('option');
            newOption.value = 'new';
            newOption.textContent = '+ Create New Model';
            select.appendChild(newOption);

            // Add existing models
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = `${model.name} (${model.architecture || 'Unknown'})`;
                select.appendChild(option);
            });
        } catch (error) {
            console.error('Error loading models:', error);
        }
    }

    async handleStartTraining() {
        const name = document.getElementById('training-name').value;
        const datasetId = document.getElementById('dataset-select').value;
        const modelId = document.getElementById('model-select').value;
        const architecture = document.getElementById('architecture-select').value;

        if (!name || !datasetId || !modelId || !architecture) {
            showToast('Please fill in all required fields', 'error');
            return;
        }

        const hyperparameters = {
            epochs: parseInt(document.getElementById('epochs').value),
            batch_size: parseInt(document.getElementById('batch-size').value),
            img_size: parseInt(document.getElementById('img-size').value),
            learning_rate: parseFloat(document.getElementById('learning-rate').value),
            num_classes: 10
        };

        try {
            // If "Create New Model" was selected, create model first
            let actualModelId = modelId;
            if (modelId === 'new') {
                const modelData = {
                    name: `${name}_model`,
                    architecture: architecture,
                    description: `Model for ${name}`
                };
                const createdModel = await apiService.post('/models/', modelData);
                actualModelId = createdModel.id;
                showToast('Model created successfully', 'success');
            }

            // Start training
            await apiService.startTraining({
                name,
                dataset_id: parseInt(datasetId),
                architecture,
                hyperparameters
            });

            bootstrap.Modal.getInstance(document.getElementById('startTrainingModal')).hide();
            showToast('Training started successfully!', 'success');

            // Reload page data
            await this.loadTrainingJobs();
            const app = document.getElementById('app');
            if (app) {
                app.innerHTML = this.render();
                await this.afterRender();
            }

            // Start polling since we have a running job now
            this.startConditionalPolling();

        } catch (error) {
            console.error('Error starting training:', error);
            showToast('Failed to start training: ' + error.message, 'error');
        }
    }
}

// Make instance globally accessible
window.trainingPage = null;
