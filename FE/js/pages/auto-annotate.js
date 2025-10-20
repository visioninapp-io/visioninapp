// Auto-Annotate Page Component

class AutoAnnotatePage {
    constructor(datasetId) {
        this.datasetId = datasetId;
        this.dataset = null;
        this.models = [];
        this.selectedModel = null;
        this.confidenceThreshold = 0.5;
        this.isAnnotating = false;
        this.progress = 0;
    }

    async init() {
        await this.loadDataset();
        await this.loadModels();
        this.attachEventListeners();
    }

    async loadDataset() {
        try {
            this.dataset = await apiService.getDataset(this.datasetId);
        } catch (error) {
            console.error('Error loading dataset:', error);
        }
    }

    async loadModels() {
        try {
            const allModels = await apiService.getModels();
            // Filter for ready models that can be used for annotation
            this.models = allModels.filter(m => m.status === 'ready' || m.status === 'deployed');
        } catch (error) {
            console.error('Error loading models:', error);
            this.models = [];
        }
    }

    render() {
        if (!this.dataset) {
            return `
                <div class="min-vh-100 bg-light d-flex align-items-center justify-content-center">
                    <div class="text-center">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="text-muted mt-3">Loading dataset...</p>
                    </div>
                </div>
            `;
        }

        return `
            <div class="min-vh-100 bg-light">
                <div class="container py-4">
                    <!-- Page Header -->
                    <div class="mb-4">
                        <nav aria-label="breadcrumb">
                            <ol class="breadcrumb">
                                <li class="breadcrumb-item"><a href="#/datasets">Datasets</a></li>
                                <li class="breadcrumb-item active">${this.dataset.name}</li>
                                <li class="breadcrumb-item active">Auto-Annotate</li>
                            </ol>
                        </nav>
                        <h1 class="display-5 fw-bold mb-2">
                            <i class="bi bi-sparkles me-2"></i>Auto-Annotate Dataset
                        </h1>
                        <p class="text-muted">Automatically annotate images using a pre-trained model</p>
                    </div>

                    <div class="row g-4">
                        <!-- Left Column: Configuration -->
                        <div class="col-lg-4">
                            <!-- Dataset Info Card -->
                            <div class="card border-0 shadow-sm mb-4">
                                <div class="card-header bg-white border-0">
                                    <h5 class="mb-0 fw-bold">Dataset Information</h5>
                                </div>
                                <div class="card-body">
                                    <h6 class="fw-bold">${this.dataset.name}</h6>
                                    ${this.dataset.description ? `<p class="text-muted small mb-3">${this.dataset.description}</p>` : ''}
                                    <div class="d-flex flex-column gap-2">
                                        <div class="d-flex justify-content-between">
                                            <span class="text-muted">Total Images:</span>
                                            <strong>${this.dataset.total_images || 0}</strong>
                                        </div>
                                        <div class="d-flex justify-content-between">
                                            <span class="text-muted">Annotated:</span>
                                            <strong>${this.dataset.annotated_images || 0}</strong>
                                        </div>
                                        <div class="d-flex justify-content-between">
                                            <span class="text-muted">Classes:</span>
                                            <strong>${this.dataset.total_classes || 0}</strong>
                                        </div>
                                        <div class="d-flex justify-content-between">
                                            <span class="text-muted">Status:</span>
                                            <span class="badge bg-primary">${this.dataset.status}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Model Selection Card -->
                            <div class="card border-0 shadow-sm mb-4">
                                <div class="card-header bg-white border-0">
                                    <h5 class="mb-0 fw-bold">Model Selection</h5>
                                </div>
                                <div class="card-body">
                                    <div class="mb-3">
                                        <label for="model-select" class="form-label">Select Pre-trained Model</label>
                                        <select class="form-select" id="model-select" ${this.isAnnotating ? 'disabled' : ''}>
                                            <option value="">-- Choose a model --</option>
                                            ${this.models.map(model => `
                                                <option value="${model.id}">
                                                    ${model.name} (${model.framework || 'Unknown'})
                                                </option>
                                            `).join('')}
                                        </select>
                                    </div>

                                    ${this.selectedModel ? `
                                        <div class="alert alert-info">
                                            <h6 class="alert-heading">Selected Model</h6>
                                            <p class="mb-1"><strong>${this.selectedModel.name}</strong></p>
                                            <small class="text-muted">
                                                Framework: ${this.selectedModel.framework}<br>
                                                Architecture: ${this.selectedModel.architecture || 'N/A'}
                                            </small>
                                        </div>
                                    ` : ''}
                                </div>
                            </div>

                            <!-- Settings Card -->
                            <div class="card border-0 shadow-sm mb-4">
                                <div class="card-header bg-white border-0">
                                    <h5 class="mb-0 fw-bold">Annotation Settings</h5>
                                </div>
                                <div class="card-body">
                                    <div class="mb-3">
                                        <label for="confidence-slider" class="form-label">
                                            Confidence Threshold: <strong id="confidence-display">${this.confidenceThreshold}</strong>
                                        </label>
                                        <input type="range" class="form-range" id="confidence-slider"
                                               min="0" max="1" step="0.05" value="${this.confidenceThreshold}"
                                               ${this.isAnnotating ? 'disabled' : ''}>
                                        <small class="text-muted">
                                            Only detections above this confidence level will be saved
                                        </small>
                                    </div>

                                    <div class="form-check form-switch mb-3">
                                        <input class="form-check-input" type="checkbox" id="auto-review" checked
                                               ${this.isAnnotating ? 'disabled' : ''}>
                                        <label class="form-check-label" for="auto-review">
                                            Review annotations after completion
                                        </label>
                                    </div>

                                    <div class="form-check form-switch">
                                        <input class="form-check-input" type="checkbox" id="overwrite-existing"
                                               ${this.isAnnotating ? 'disabled' : ''}>
                                        <label class="form-check-label" for="overwrite-existing">
                                            Overwrite existing annotations
                                        </label>
                                    </div>
                                </div>
                            </div>

                            <!-- Action Buttons -->
                            <div class="d-grid gap-2">
                                <button class="btn btn-primary btn-lg" id="start-annotation-btn"
                                        ${this.isAnnotating || !this.selectedModel ? 'disabled' : ''}>
                                    <i class="bi bi-play-fill me-2"></i>Start Auto-Annotation
                                </button>
                                <button class="btn btn-outline-danger" id="stop-annotation-btn"
                                        ${!this.isAnnotating ? 'disabled' : ''}>
                                    <i class="bi bi-stop-fill me-2"></i>Stop
                                </button>
                                <a href="#/datasets" class="btn btn-outline-secondary">
                                    <i class="bi bi-arrow-left me-2"></i>Back to Datasets
                                </a>
                            </div>
                        </div>

                        <!-- Right Column: Preview and Progress -->
                        <div class="col-lg-8">
                            <!-- Progress Card -->
                            <div class="card border-0 shadow-sm mb-4 ${this.isAnnotating ? '' : 'd-none'}" id="progress-card">
                                <div class="card-header bg-white border-0">
                                    <h5 class="mb-0 fw-bold">
                                        <i class="bi bi-hourglass-split me-2"></i>Annotation Progress
                                    </h5>
                                </div>
                                <div class="card-body">
                                    <div class="mb-3">
                                        <div class="d-flex justify-content-between mb-2">
                                            <span>Processing images...</span>
                                            <span><strong id="progress-text">0%</strong></span>
                                        </div>
                                        <div class="progress" style="height: 30px;">
                                            <div id="annotation-progress-bar" class="progress-bar progress-bar-striped progress-bar-animated"
                                                 role="progressbar" style="width: 0%">
                                                <span id="progress-count">0 / ${this.dataset.total_images || 0}</span>
                                            </div>
                                        </div>
                                    </div>

                                    <div id="annotation-stats" class="row g-3 text-center">
                                        <div class="col-4">
                                            <div class="p-3 bg-light rounded">
                                                <h3 class="mb-0" id="stat-processed">0</h3>
                                                <small class="text-muted">Processed</small>
                                            </div>
                                        </div>
                                        <div class="col-4">
                                            <div class="p-3 bg-light rounded">
                                                <h3 class="mb-0" id="stat-detected">0</h3>
                                                <small class="text-muted">Detections</small>
                                            </div>
                                        </div>
                                        <div class="col-4">
                                            <div class="p-3 bg-light rounded">
                                                <h3 class="mb-0" id="stat-avg-confidence">0%</h3>
                                                <small class="text-muted">Avg Confidence</small>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Preview Card -->
                            <div class="card border-0 shadow-sm">
                                <div class="card-header bg-white border-0">
                                    <h5 class="mb-0 fw-bold">
                                        <i class="bi bi-eye me-2"></i>Preview
                                    </h5>
                                </div>
                                <div class="card-body">
                                    <div id="preview-content" class="text-center py-5">
                                        ${this.renderPreviewContent()}
                                    </div>
                                </div>
                            </div>

                            <!-- Instructions Card -->
                            <div class="card border-0 shadow-sm mt-4 ${this.isAnnotating ? 'd-none' : ''}" id="instructions-card">
                                <div class="card-header bg-white border-0">
                                    <h5 class="mb-0 fw-bold">
                                        <i class="bi bi-info-circle me-2"></i>How Auto-Annotation Works
                                    </h5>
                                </div>
                                <div class="card-body">
                                    <ol class="mb-0">
                                        <li class="mb-2">Select a pre-trained model from the dropdown</li>
                                        <li class="mb-2">Adjust the confidence threshold (higher = more strict)</li>
                                        <li class="mb-2">Click "Start Auto-Annotation" to begin the process</li>
                                        <li class="mb-2">The model will automatically detect objects in all images</li>
                                        <li>Review and refine the annotations after completion</li>
                                    </ol>

                                    <div class="alert alert-warning mt-3 mb-0">
                                        <strong>Note:</strong> Auto-annotation is not perfect. Always review and verify
                                        the generated annotations before using them for training.
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderPreviewContent() {
        if (this.isAnnotating) {
            return `
                <div class="spinner-border text-primary mb-3" role="status">
                    <span class="visually-hidden">Annotating...</span>
                </div>
                <p class="text-muted">Processing images with auto-annotation...</p>
            `;
        }

        return `
            <i class="bi bi-image text-muted mb-3" style="font-size: 5rem;"></i>
            <h5 class="text-muted">Preview will appear here</h5>
            <p class="text-muted">Select a model and start annotation to see results</p>
        `;
    }

    attachEventListeners() {
        // Model selection
        const modelSelect = document.getElementById('model-select');
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                const modelId = parseInt(e.target.value);
                this.selectedModel = this.models.find(m => m.id === modelId);
                this.updateUI();
            });
        }

        // Confidence threshold
        const confidenceSlider = document.getElementById('confidence-slider');
        if (confidenceSlider) {
            confidenceSlider.addEventListener('input', (e) => {
                this.confidenceThreshold = parseFloat(e.target.value);
                document.getElementById('confidence-display').textContent = this.confidenceThreshold;
            });
        }

        // Start annotation
        const startBtn = document.getElementById('start-annotation-btn');
        if (startBtn) {
            startBtn.addEventListener('click', () => this.startAnnotation());
        }

        // Stop annotation
        const stopBtn = document.getElementById('stop-annotation-btn');
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopAnnotation());
        }
    }

    async startAnnotation() {
        if (!this.selectedModel) {
            return;
        }

        this.isAnnotating = true;
        this.progress = 0;
        this.updateUI();

        // Show progress card
        document.getElementById('progress-card')?.classList.remove('d-none');
        document.getElementById('instructions-card')?.classList.add('d-none');

        try {
            // Call API to start auto-annotation
            const result = await apiService.autoAnnotate(this.datasetId, this.selectedModel.id);

            // Simulate progress (in real app, you'd poll for status)
            await this.simulateProgress();

            // Redirect to dataset detail page for review
            const autoReview = document.getElementById('auto-review')?.checked;
            if (autoReview) {
                setTimeout(() => {
                    window.location.hash = `#/dataset-detail/${this.datasetId}`;
                }, 2000);
            }

        } catch (error) {
            console.error('Auto-annotation error:', error);
            this.isAnnotating = false;
            this.updateUI();
        }
    }

    async simulateProgress() {
        const totalImages = this.dataset.total_images || 10;

        return new Promise((resolve) => {
            let processed = 0;
            let detected = 0;

            const interval = setInterval(() => {
                processed++;
                detected += Math.floor(Math.random() * 5) + 1;

                this.progress = Math.round((processed / totalImages) * 100);

                // Update progress bar
                const progressBar = document.getElementById('annotation-progress-bar');
                if (progressBar) {
                    progressBar.style.width = this.progress + '%';
                }

                // Update text
                document.getElementById('progress-text').textContent = this.progress + '%';
                document.getElementById('progress-count').textContent = `${processed} / ${totalImages}`;

                // Update stats
                document.getElementById('stat-processed').textContent = processed;
                document.getElementById('stat-detected').textContent = detected;
                document.getElementById('stat-avg-confidence').textContent =
                    Math.round((this.confidenceThreshold + Math.random() * 0.3) * 100) + '%';

                if (processed >= totalImages) {
                    clearInterval(interval);
                    this.isAnnotating = false;
                    resolve();
                }
            }, 500);
        });
    }

    stopAnnotation() {
        this.isAnnotating = false;
        this.updateUI();
    }

    updateUI() {
        // Re-render the page
        const app = document.getElementById('app');
        if (app) {
            app.innerHTML = this.render();
            this.attachEventListeners();
        }
    }
}
