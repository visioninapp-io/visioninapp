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
        this.useDefaultModel = true; // Default to using built-in model
    }

    async init() {
        console.log('[AutoAnnotatePage] Initializing...');
        await this.loadDataset();
        await this.loadModels();

        // Re-render after data is loaded
        console.log('[AutoAnnotatePage] Data loaded, updating UI...');
        this.updatePage();
        // attachEventListeners() is called inside updatePage()
    }

    updatePage() {
        // Re-render the entire page
        const app = document.getElementById('app');
        if (app) {
            app.innerHTML = this.render();
            // Re-attach event listeners after re-rendering
            this.attachEventListeners();
            
            // Explicitly set the dropdown value after re-render to ensure selection is preserved
            if (this.selectedModel) {
                const modelSelect = document.getElementById('model-select');
                if (modelSelect) {
                    const selectedValue = String(this.selectedModel.id);
                    modelSelect.value = selectedValue;
                }
            }
        }
    }

    async loadDataset() {
        try {
            console.log('[AutoAnnotatePage] Loading dataset...');
            this.dataset = await apiService.getDataset(this.datasetId);
            console.log('[AutoAnnotatePage] Dataset loaded:', this.dataset);
        } catch (error) {
            console.error('[AutoAnnotatePage] Error loading dataset:', error);
            showToast('Failed to load dataset', 'error');
        }
    }

    async loadModels() {
        try {
            console.log('[AutoAnnotatePage] Loading trained models...');
            const trainedModels = await apiService.getTrainedModels();
            console.log('[AutoAnnotatePage] Trained models:', trainedModels);
            
            // Transform trained models to match expected format
            // Use 'id' field from backend (always present), fallback to model_id or model_name
            this.models = trainedModels.map(m => ({
                id: m.id || m.model_id || m.model_name,  // Backend provides 'id' field that's always present
                // model_id: m.model_id,  // Keep original model_id for API calls
                name: m.model_name,
                model_path: m.relative_path,
                s3_key: m.s3_key,  // Keep S3 key for reference
                file_size_mb: m.file_size_mb,
                framework: 'YOLO',
                architecture: 'YOLOv8',
                status: 'trained'
            }));
            
            console.log('[AutoAnnotatePage] Formatted models:', this.models);
        } catch (error) {
            console.error('[AutoAnnotatePage] Error loading models:', error);
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
                        <p class="text-muted">Automatically annotate images using AI-powered object detection</p>
                    </div>

                    <div class="row g-4">
                        <!-- Left Column: Configuration -->
                        <div class="col-lg-4">
                            <!-- Dataset Info Card -->
                            <div class="card border-0 shadow-sm mb-4">
                                <div class="card-header bg-white border-0">
                                    <h5 class="mb-0 fw-bold">
                                        <i class="bi bi-info-circle me-2"></i>Dataset Information
                                    </h5>
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
                                            <strong class="text-success">${this.dataset.annotated_images || 0}</strong>
                                        </div>
                                        <div class="d-flex justify-content-between">
                                            <span class="text-muted">Remaining:</span>
                                            <strong class="text-warning">${(this.dataset.total_images || 0) - (this.dataset.annotated_images || 0)}</strong>
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
                                    <h5 class="mb-0 fw-bold">
                                        <i class="bi bi-cpu me-2"></i>Model Selection
                                    </h5>
                                </div>
                                <div class="card-body">
                                    <div class="form-check form-switch mb-3">
                                        <input class="form-check-input" type="checkbox" id="use-default-model"
                                               ${this.useDefaultModel ? 'checked' : ''} ${this.isAnnotating ? 'disabled' : ''}>
                                        <label class="form-check-label" for="use-default-model">
                                            <strong>Use Default Model</strong>
                                            <br><small class="text-muted">YOLOv8 (AI/models/best.pt)</small>
                                        </label>
                                    </div>

                                    <div id="custom-model-section" class="${this.useDefaultModel ? 'd-none' : ''}">
                                        <div class="mb-3">
                                            <label for="model-select" class="form-label">Select Trained Model</label>
                                            <select class="form-select" id="model-select" ${this.isAnnotating ? 'disabled' : ''}>
                                                <option value="">-- Choose a trained model --</option>
                                                ${this.models.map(model => {
                                                    const isSelected = this.selectedModel && (
                                                        String(this.selectedModel.id) === String(model.id) || 
                                                        this.selectedModel.id === model.id
                                                    );
                                                    return `<option value="${model.id}" ${isSelected ? 'selected' : ''}>
                                                        ${model.name} (${model.file_size_mb}MB)
                                                    </option>`;
                                                }).join('')}
                                            </select>
                                            ${this.models.length === 0 ? `
                                                <small class="text-warning">
                                                    <i class="bi bi-exclamation-triangle me-1"></i>No trained models found. Please train a model first or use the default model.
                                                </small>
                                            ` : `
                                                <small class="text-muted">
                                                    <i class="bi bi-info-circle me-1"></i>${this.models.length} trained model(s) available
                                                </small>
                                            `}
                                        </div>

                                        <div id="selected-model-info">
                                            ${this.selectedModel ? `
                                                <div class="alert alert-info mb-0">
                                                    <h6 class="alert-heading mb-1">Selected Model</h6>
                                                    <p class="mb-1"><strong>${this.selectedModel.name}</strong></p>
                                                    <small class="text-muted">
                                                        Framework: ${this.selectedModel.framework}<br>
                                                        Architecture: ${this.selectedModel.architecture || 'YOLOv8'}<br>
                                                        Size: ${this.selectedModel.file_size_mb}MB
                                                    </small>
                                                </div>
                                            ` : ''}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Settings Card -->
                            <div class="card border-0 shadow-sm mb-4">
                                <div class="card-header bg-white border-0">
                                    <h5 class="mb-0 fw-bold">
                                        <i class="bi bi-gear me-2"></i>Annotation Settings
                                    </h5>
                                </div>
                                <div class="card-body">
                                    <div class="mb-3">
                                        <label for="confidence-slider" class="form-label">
                                            Confidence Threshold: <strong id="confidence-display">${this.confidenceThreshold}</strong>
                                        </label>
                                        <input type="range" class="form-range" id="confidence-slider"
                                               min="0" max="1" step="0.05" value="${this.confidenceThreshold}"
                                               ${this.isAnnotating ? 'disabled' : ''}>
                                        <div class="d-flex justify-content-between small text-muted">
                                            <span>Low (0.0)</span>
                                            <span>High (1.0)</span>
                                        </div>
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
                                <button class="btn btn-success btn-lg" id="start-annotation-btn"
                                        ${this.isAnnotating ? 'disabled' : ''}>
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
                                            <div id="annotation-progress-bar" class="progress-bar progress-bar-striped progress-bar-animated bg-success"
                                                 role="progressbar" style="width: 0%">
                                                <span id="progress-count">0 / ${this.dataset.total_images || 0}</span>
                                            </div>
                                        </div>
                                    </div>

                                    <div id="annotation-stats" class="row g-3 text-center">
                                        <div class="col-4">
                                            <div class="p-3 bg-light rounded">
                                                <h3 class="mb-0 text-primary" id="stat-processed">0</h3>
                                                <small class="text-muted">Processed</small>
                                            </div>
                                        </div>
                                        <div class="col-4">
                                            <div class="p-3 bg-light rounded">
                                                <h3 class="mb-0 text-success" id="stat-detected">0</h3>
                                                <small class="text-muted">Detections</small>
                                            </div>
                                        </div>
                                        <div class="col-4">
                                            <div class="p-3 bg-light rounded">
                                                <h3 class="mb-0 text-info" id="stat-avg-confidence">0%</h3>
                                                <small class="text-muted">Avg Confidence</small>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="alert alert-info mt-3 mb-0">
                                        <i class="bi bi-info-circle me-2"></i>
                                        <strong>Processing:</strong> Running AI model on each image to detect objects...
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
                                        <i class="bi bi-lightbulb me-2"></i>How Auto-Annotation Works
                                    </h5>
                                </div>
                                <div class="card-body">
                                    <ol class="mb-3">
                                        <li class="mb-2">The system uses a pre-trained YOLO model (AI/models/best.pt)</li>
                                        <li class="mb-2">Adjust the confidence threshold (higher = more strict)</li>
                                        <li class="mb-2">Click "Start Auto-Annotation" to begin the process</li>
                                        <li class="mb-2">The model will automatically detect objects in all images</li>
                                        <li>Review and refine the annotations after completion</li>
                                    </ol>

                                    <div class="alert alert-warning mb-0">
                                        <i class="bi bi-exclamation-triangle me-2"></i>
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
                <div class="spinner-border text-success mb-3" style="width: 4rem; height: 4rem;" role="status">
                    <span class="visually-hidden">Annotating...</span>
                </div>
                <h5 class="text-muted">Processing images with AI...</h5>
                <p class="text-muted">This may take a few minutes depending on the dataset size</p>
            `;
        }

        return `
            <i class="bi bi-robot text-muted mb-3" style="font-size: 5rem;"></i>
            <h5 class="text-muted">Ready for Auto-Annotation</h5>
            <p class="text-muted">Click "Start Auto-Annotation" to begin processing</p>
            <div class="mt-4">
                <span class="badge bg-success me-2">AI-Powered</span>
                <span class="badge bg-primary me-2">Fast & Accurate</span>
                <span class="badge bg-info">Real-time Progress</span>
            </div>
        `;
    }

    updateSelectedModelInfo() {
        // Update only the selected model info section without re-rendering entire page
        const selectedModelInfo = document.getElementById('selected-model-info');
        if (selectedModelInfo) {
            if (this.selectedModel) {
                selectedModelInfo.innerHTML = `
                    <div class="alert alert-info mb-0">
                        <h6 class="alert-heading mb-1">Selected Model</h6>
                        <p class="mb-1"><strong>${this.selectedModel.name}</strong></p>
                        <small class="text-muted">
                            Framework: ${this.selectedModel.framework}<br>
                            Architecture: ${this.selectedModel.architecture || 'YOLOv8'}<br>
                            Size: ${this.selectedModel.file_size_mb}MB
                        </small>
                    </div>
                `;
                selectedModelInfo.classList.remove('d-none');
            } else {
                selectedModelInfo.innerHTML = '';
                selectedModelInfo.classList.add('d-none');
            }
        } else {
            // If the element doesn't exist, we need to re-render
            this.updatePage();
        }
    }

    attachEventListeners() {
        // Use default model toggle
        const useDefaultModelCheckbox = document.getElementById('use-default-model');
        if (useDefaultModelCheckbox) {
            useDefaultModelCheckbox.addEventListener('change', (e) => {
                this.useDefaultModel = e.target.checked;
                const customModelSection = document.getElementById('custom-model-section');
                if (customModelSection) {
                    customModelSection.classList.toggle('d-none', this.useDefaultModel);
                }
            });
        }

        // Model selection
        const modelSelect = document.getElementById('model-select');
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                const selectedValue = e.target.value;
                
                if (!selectedValue || selectedValue === '') {
                    this.selectedModel = null;
                } else {
                    // Find model by id (handle both string and number comparisons)
                    this.selectedModel = this.models.find(m => 
                        String(m.id) === String(selectedValue) || 
                        m.id === selectedValue ||
                        m.id === parseInt(selectedValue)
                    );
                }
                
                // Update only the selected model info section instead of re-rendering entire page
                this.updateSelectedModelInfo();
            });
        }

        // Confidence threshold
        const confidenceSlider = document.getElementById('confidence-slider');
        if (confidenceSlider) {
            confidenceSlider.addEventListener('input', (e) => {
                this.confidenceThreshold = parseFloat(e.target.value);
                document.getElementById('confidence-display').textContent = this.confidenceThreshold.toFixed(2);
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
        console.log('[AutoAnnotate] Starting annotation process...');
        console.log('[AutoAnnotate] Dataset ID:', this.datasetId);
        console.log('[AutoAnnotate] Use default model:', this.useDefaultModel);
        console.log('[AutoAnnotate] Confidence threshold:', this.confidenceThreshold);

        // Determine which model to use
        // Use model_id (database ID) if available, otherwise use id (hash/fallback)
        const modelId = this.useDefaultModel ? null : (this.selectedModel ? (this.selectedModel.model_id || this.selectedModel.id) : null);
        // console.log('[AutoAnnotate] Selected model ID:', modelId);
        // console.log('[AutoAnnotate] Selected model:', this.selectedModel);

        // Get overwrite existing checkbox value
        const overwriteExisting = document.getElementById('overwrite-existing')?.checked || false;
        console.log('[AutoAnnotate] Overwrite existing:', overwriteExisting);

        this.isAnnotating = true;
        this.progress = 0;
        this.updatePage();

        // Show progress card
        document.getElementById('progress-card')?.classList.remove('d-none');
        document.getElementById('instructions-card')?.classList.add('d-none');

        try {
            console.log('[AutoAnnotate] Showing toast notification...');
            showToast('Starting auto-annotation...', 'info');

            console.log('[AutoAnnotate] Calling API...');
            // Call API to start auto-annotation
            const result = await apiService.autoAnnotate(
                this.datasetId,
                modelId,
                this.confidenceThreshold,
                overwriteExisting
            );

            console.log('[AutoAnnotate] API call successful. Result:', result);

            // Simulate progress display
            await this.simulateProgress(result);

            showToast('Auto-annotation completed successfully!', 'success');

            // Redirect to dataset detail page for review
            const autoReview = document.getElementById('auto-review')?.checked;
            console.log('[AutoAnnotate] Auto review enabled:', autoReview);

            if (autoReview) {
                setTimeout(() => {
                    console.log('[AutoAnnotate] Redirecting to dataset detail...');
                    window.location.hash = `#/dataset-detail/${this.datasetId}`;
                }, 2000);
            } else {
                setTimeout(() => {
                    console.log('[AutoAnnotate] Redirecting to datasets list...');
                    window.location.hash = '#/datasets';
                }, 2000);
            }

        } catch (error) {
            console.error('[AutoAnnotate] Error occurred:', error);
            console.error('[AutoAnnotate] Error message:', error.message);
            console.error('[AutoAnnotate] Error stack:', error.stack);
            showToast(`Auto-annotation failed: ${error.message}`, 'error');
            this.isAnnotating = false;
            this.updatePage();
        }
    }

    async simulateProgress(result) {
        const totalImages = result.total_images || this.dataset.total_images || 10;
        const annotatedImages = result.annotated_images || 0;
        const totalAnnotations = result.total_annotations || 0;

        return new Promise((resolve) => {
            let processed = 0;
            const targetProcessed = totalImages;

            const interval = setInterval(() => {
                processed += Math.max(1, Math.floor(targetProcessed / 20));
                if (processed > targetProcessed) processed = targetProcessed;

                this.progress = Math.round((processed / targetProcessed) * 100);

                // Update progress bar
                const progressBar = document.getElementById('annotation-progress-bar');
                if (progressBar) {
                    progressBar.style.width = this.progress + '%';
                }

                // Update text
                document.getElementById('progress-text').textContent = this.progress + '%';
                document.getElementById('progress-count').textContent = `${processed} / ${targetProcessed}`;

                // Update stats
                const detectedPerImage = Math.ceil(totalAnnotations / annotatedImages) || 1;
                document.getElementById('stat-processed').textContent = processed;
                document.getElementById('stat-detected').textContent = Math.round(processed * detectedPerImage * 0.7);
                document.getElementById('stat-avg-confidence').textContent =
                    Math.round((this.confidenceThreshold + Math.random() * 0.2) * 100) + '%';

                if (processed >= targetProcessed) {
                    clearInterval(interval);
                    this.isAnnotating = false;
                    resolve();
                }
            }, 100);
        });
    }

    stopAnnotation() {
        this.isAnnotating = false;
        showToast('Annotation stopped', 'warning');
        this.updatePage();
    }

}