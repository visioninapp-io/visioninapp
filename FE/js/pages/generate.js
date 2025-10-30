// Generate (Augmentation & Versions) Page Component

class GeneratePage {
    constructor() {
        console.log('[GeneratePage] Initializing...');
        this.datasets = [];
        this.selectedDataset = null;
        this.versions = [];
        this.isLoading = true;
    }

    async init() {
        console.log('[GeneratePage] Starting initialization...');
        try {
            await Promise.all([
                this.loadDatasets()
            ]);
            this.isLoading = false;
        } catch (error) {
            console.error('[GeneratePage] Initialization failed:', error);
            this.isLoading = false;
        }

        // Re-render
        const app = document.getElementById('app');
        if (app) {
            app.innerHTML = this.render();
            this.attachEventListeners();
        }
    }

    async loadDatasets() {
        try {
            this.datasets = await apiService.getDatasets();
            if (this.datasets.length > 0) {
                this.selectedDataset = this.datasets[0];
                await this.loadVersions(this.selectedDataset.id);
            }
        } catch (error) {
            console.error('Error loading datasets:', error);
            this.datasets = [];
        }
    }

    async loadVersions(datasetId) {
        try {
            this.versions = await apiService.getDatasetVersions(datasetId);
        } catch (error) {
            console.error('Error loading versions:', error);
            this.versions = [];
        }
    }

    render() {
        if (this.isLoading) {
            return `
                <div class="min-vh-100 bg-light d-flex align-items-center justify-content-center">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            `;
        }

        return `
            <div class="min-vh-100 bg-light">
                <div class="container py-4">
                    <!-- Header -->
                    <div class="mb-4">
                        <h1 class="display-5 fw-bold mb-2">Generate & Versions</h1>
                        <p class="text-muted">Create augmented dataset versions with preprocessing and augmentation</p>
                    </div>

                    <!-- Dataset Selector -->
                    <div class="row mb-4">
                        <div class="col-md-6">
                            <div class="card border-0 shadow-sm">
                                <div class="card-body">
                                    <label class="form-label fw-bold">Select Dataset</label>
                                    <select class="form-select form-select-lg" id="dataset-selector">
                                        ${this.datasets.length === 0 ?
                                            '<option value="">No datasets available</option>' :
                                            this.datasets.map(dataset => `
                                                <option value="${dataset.id}" ${this.selectedDataset && this.selectedDataset.id === dataset.id ? 'selected' : ''}>
                                                    ${dataset.name} (${dataset.total_images || 0} images)
                                                </option>
                                            `).join('')
                                        }
                                    </select>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card border-0 shadow-sm">
                                <div class="card-body">
                                    <label class="form-label fw-bold">Actions</label>
                                    <button class="btn btn-primary btn-lg w-100"
                                            onclick="window.currentPageInstance.showCreateVersionModal()"
                                            ${!this.selectedDataset ? 'disabled' : ''}>
                                        <i class="bi bi-plus-circle me-2"></i>Create New Version
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Versions List -->
                    ${this.renderVersionsList()}
                </div>
            </div>
        `;
    }

    renderVersionsList() {
        if (!this.selectedDataset) {
            return `
                <div class="card border-0 shadow-sm">
                    <div class="card-body text-center py-5">
                        <i class="bi bi-folder text-muted" style="font-size: 5rem;"></i>
                        <h4 class="mt-4">No Dataset Selected</h4>
                        <p class="text-muted">Select a dataset to view its versions</p>
                    </div>
                </div>
            `;
        }

        if (this.versions.length === 0) {
            return `
                <div class="card border-0 shadow-sm">
                    <div class="card-body text-center py-5">
                        <i class="bi bi-layers text-muted" style="font-size: 5rem;"></i>
                        <h4 class="mt-4">No Versions Yet</h4>
                        <p class="text-muted">Create your first augmented version of this dataset</p>
                        <button class="btn btn-primary btn-lg mt-3" onclick="window.currentPageInstance.showCreateVersionModal()">
                            <i class="bi bi-plus-circle me-2"></i>Create Version
                        </button>
                    </div>
                </div>
            `;
        }

        return `
            <div class="row g-4">
                ${this.versions.map(version => this.renderVersionCard(version)).join('')}
            </div>
        `;
    }

    renderVersionCard(version) {
        const statusBadge = {
            'pending': 'bg-secondary',
            'generating': 'bg-warning',
            'completed': 'bg-success',
            'failed': 'bg-danger'
        }[version.status] || 'bg-secondary';

        return `
            <div class="col-md-6 col-lg-4">
                <div class="card border-0 shadow-sm h-100">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start mb-3">
                            <div>
                                <h5 class="fw-bold mb-1">${version.name}</h5>
                                <p class="text-muted small mb-0">Version ${version.version_number}</p>
                            </div>
                            <span class="badge ${statusBadge}">${version.status}</span>
                        </div>

                        ${version.description ? `<p class="text-muted small mb-3">${version.description}</p>` : ''}

                        <!-- Split Info -->
                        <div class="mb-3">
                            <h6 class="fw-bold small mb-2">Data Split</h6>
                            <div class="d-flex gap-2">
                                <span class="badge bg-primary">${Math.round(version.train_split * 100)}% Train</span>
                                <span class="badge bg-info">${Math.round(version.valid_split * 100)}% Valid</span>
                                <span class="badge bg-secondary">${Math.round(version.test_split * 100)}% Test</span>
                            </div>
                        </div>

                        <!-- Stats -->
                        <div class="border-top pt-3">
                            <div class="row text-center">
                                <div class="col-4">
                                    <div class="fw-bold">${version.total_images || 0}</div>
                                    <small class="text-muted">Total</small>
                                </div>
                                <div class="col-4">
                                    <div class="fw-bold">${version.train_images || 0}</div>
                                    <small class="text-muted">Train</small>
                                </div>
                                <div class="col-4">
                                    <div class="fw-bold">${version.valid_images || 0}</div>
                                    <small class="text-muted">Valid</small>
                                </div>
                            </div>
                        </div>

                        ${version.status === 'generating' ? `
                            <div class="progress mt-3" style="height: 8px;">
                                <div class="progress-bar progress-bar-striped progress-bar-animated"
                                     style="width: ${version.generation_progress}%"></div>
                            </div>
                            <small class="text-muted">${version.generation_progress}% complete</small>
                        ` : ''}

                        <!-- Actions -->
                        <div class="d-flex gap-2 mt-3">
                            ${version.status === 'completed' ? `
                                <button class="btn btn-sm btn-outline-primary flex-fill"
                                        onclick="window.currentPageInstance.exportVersion(${version.id})">
                                    <i class="bi bi-download"></i> Export
                                </button>
                            ` : ''}
                            <button class="btn btn-sm btn-outline-danger flex-fill"
                                    onclick="window.currentPageInstance.deleteVersion(${version.id})">
                                <i class="bi bi-trash"></i> Delete
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    attachEventListeners() {
        const datasetSelector = document.getElementById('dataset-selector');
        if (datasetSelector) {
            datasetSelector.addEventListener('change', async (e) => {
                const datasetId = parseInt(e.target.value);
                this.selectedDataset = this.datasets.find(d => d.id === datasetId);
                await this.loadVersions(datasetId);
                const app = document.getElementById('app');
                if (app) {
                    app.innerHTML = this.render();
                    this.attachEventListeners();
                }
            });
        }
    }

    showCreateVersionModal() {
        if (!this.selectedDataset) {
            showToast('Please select a dataset first', 'error');
            return;
        }

        const modalHTML = `
            <div class="modal fade" id="createVersionModal" tabindex="-1" aria-hidden="true">
                <div class="modal-dialog modal-xl">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">
                                <i class="bi bi-plus-circle me-2"></i>Create New Dataset Version
                            </h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body bg-light">
                            <!-- Nav Tabs -->
                            <ul class="nav nav-tabs mb-4 bg-white p-2 rounded" id="versionTabs" role="tablist">
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link active" id="basic-tab" data-bs-toggle="tab"
                                            data-bs-target="#basic" type="button" role="tab"
                                            aria-controls="basic" aria-selected="true">
                                        <i class="bi bi-info-circle me-1"></i>Basic Info
                                    </button>
                                </li>
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link" id="split-tab" data-bs-toggle="tab"
                                            data-bs-target="#split" type="button" role="tab"
                                            aria-controls="split" aria-selected="false">
                                        <i class="bi bi-pie-chart me-1"></i>Data Split
                                    </button>
                                </li>
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link" id="preprocessing-tab" data-bs-toggle="tab"
                                            data-bs-target="#preprocessing" type="button" role="tab"
                                            aria-controls="preprocessing" aria-selected="false">
                                        <i class="bi bi-sliders me-1"></i>Preprocessing
                                    </button>
                                </li>
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link" id="augmentation-tab" data-bs-toggle="tab"
                                            data-bs-target="#augmentation" type="button" role="tab"
                                            aria-controls="augmentation" aria-selected="false">
                                        <i class="bi bi-shuffle me-1"></i>Augmentation
                                    </button>
                                </li>
                            </ul>

                            <!-- Tab Content -->
                            <div class="tab-content" id="versionTabsContent">
                                <!-- Basic Info Tab -->
                                <div class="tab-pane fade show active p-4 bg-white rounded" id="basic"
                                     role="tabpanel" aria-labelledby="basic-tab">
                                    ${this.renderBasicInfoTab()}
                                </div>

                                <!-- Data Split Tab -->
                                <div class="tab-pane fade p-4 bg-white rounded" id="split"
                                     role="tabpanel" aria-labelledby="split-tab">
                                    ${this.renderDataSplitTab()}
                                </div>

                                <!-- Preprocessing Tab -->
                                <div class="tab-pane fade p-4 bg-white rounded" id="preprocessing"
                                     role="tabpanel" aria-labelledby="preprocessing-tab">
                                    ${this.renderPreprocessingTab()}
                                </div>

                                <!-- Augmentation Tab -->
                                <div class="tab-pane fade p-4 bg-white rounded" id="augmentation"
                                     role="tabpanel" aria-labelledby="augmentation-tab">
                                    ${this.renderAugmentationTab()}
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" id="create-version-btn">
                                <i class="bi bi-check-circle me-1"></i>Create Version
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Remove existing modal
        const existingModal = document.getElementById('createVersionModal');
        if (existingModal) existingModal.remove();

        // Add modal to DOM
        document.body.insertAdjacentHTML('beforeend', modalHTML);

        // Get modal element
        const modalElement = document.getElementById('createVersionModal');

        // Show modal
        const modal = new bootstrap.Modal(modalElement);
        modal.show();

        // Attach event listeners after modal is shown
        modalElement.addEventListener('shown.bs.modal', () => {
            this.attachModalEventListeners();
        }, { once: true });
    }

    renderBasicInfoTab() {
        return `
            <div class="mb-3">
                <label for="version-name" class="form-label fw-bold">Version Name *</label>
                <input type="text" class="form-control" id="version-name" required
                       placeholder="e.g., v1.0-augmented">
                <small class="text-muted">Give your version a descriptive name</small>
            </div>
            <div class="mb-3">
                <label for="version-description" class="form-label fw-bold">Description</label>
                <textarea class="form-control" id="version-description" rows="3"
                          placeholder="Describe the changes in this version (e.g., applied horizontal flip and rotation augmentation)"></textarea>
            </div>
            <div class="alert alert-info">
                <i class="bi bi-info-circle me-2"></i>
                <strong>Dataset:</strong> ${this.selectedDataset.name} (${this.selectedDataset.total_images} images)
            </div>
        `;
    }

    renderDataSplitTab() {
        return `
            <p class="text-muted mb-4">Configure how to split your dataset into training, validation, and test sets</p>

            <div class="row g-3">
                <div class="col-md-4">
                    <label class="form-label fw-bold">Train Split</label>
                    <div class="input-group">
                        <input type="number" class="form-control" id="train-split" value="70" min="0" max="100">
                        <span class="input-group-text">%</span>
                    </div>
                    <small class="text-muted">Training data</small>
                </div>
                <div class="col-md-4">
                    <label class="form-label fw-bold">Validation Split</label>
                    <div class="input-group">
                        <input type="number" class="form-control" id="valid-split" value="20" min="0" max="100">
                        <span class="input-group-text">%</span>
                    </div>
                    <small class="text-muted">Validation data</small>
                </div>
                <div class="col-md-4">
                    <label class="form-label fw-bold">Test Split</label>
                    <div class="input-group">
                        <input type="number" class="form-control" id="test-split" value="10" min="0" max="100">
                        <span class="input-group-text">%</span>
                    </div>
                    <small class="text-muted">Test data</small>
                </div>
            </div>

            <div class="mt-3">
                <div class="alert alert-warning" id="split-warning" style="display: none;">
                    <i class="bi bi-exclamation-triangle me-2"></i>
                    Total must equal 100%
                </div>
            </div>

            <div class="mt-4">
                <h6 class="fw-bold mb-3">Estimated Distribution</h6>
                <div class="progress" style="height: 30px;">
                    <div class="progress-bar bg-primary" id="train-bar" style="width: 70%">
                        <span id="train-count">Train: 70%</span>
                    </div>
                    <div class="progress-bar bg-info" id="valid-bar" style="width: 20%">
                        <span id="valid-count">Valid: 20%</span>
                    </div>
                    <div class="progress-bar bg-secondary" id="test-bar" style="width: 10%">
                        <span id="test-count">Test: 10%</span>
                    </div>
                </div>
            </div>
        `;
    }

    renderPreprocessingTab() {
        return `
            <p class="text-muted mb-4">Configure preprocessing operations to apply to all images</p>

            <!-- Resize -->
            <div class="card mb-3">
                <div class="card-header">
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" id="enable-resize">
                        <label class="form-check-label fw-bold" for="enable-resize">
                            Resize Images
                        </label>
                    </div>
                </div>
                <div class="card-body" id="resize-options" style="display: none;">
                    <div class="row g-3">
                        <div class="col-md-4">
                            <label class="form-label">Width</label>
                            <input type="number" class="form-control" id="resize-width" value="640">
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">Height</label>
                            <input type="number" class="form-control" id="resize-height" value="640">
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">Mode</label>
                            <select class="form-select" id="resize-mode">
                                <option value="fit">Fit (maintain aspect ratio)</option>
                                <option value="stretch">Stretch</option>
                                <option value="pad">Pad with black</option>
                            </select>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Normalization -->
            <div class="card mb-3">
                <div class="card-header">
                    <h6 class="mb-0 fw-bold">Normalization</h6>
                </div>
                <div class="card-body">
                    <div class="form-check mb-2">
                        <input class="form-check-input" type="checkbox" id="normalize-contrast">
                        <label class="form-check-label" for="normalize-contrast">
                            Normalize Contrast
                        </label>
                    </div>
                    <div class="form-check mb-2">
                        <input class="form-check-input" type="checkbox" id="normalize-exposure">
                        <label class="form-check-label" for="normalize-exposure">
                            Normalize Exposure
                        </label>
                    </div>
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" id="auto-orient">
                        <label class="form-check-label" for="auto-orient">
                            Auto-orient (based on EXIF)
                        </label>
                    </div>
                </div>
            </div>

            <!-- Filtering -->
            <div class="card">
                <div class="card-header">
                    <h6 class="mb-0 fw-bold">Filtering</h6>
                </div>
                <div class="card-body">
                    <div class="form-check mb-2">
                        <input class="form-check-input" type="checkbox" id="filter-null" checked>
                        <label class="form-check-label" for="filter-null">
                            Remove images with no annotations
                        </label>
                    </div>
                </div>
            </div>
        `;
    }

    renderAugmentationTab() {
        return `
            <p class="text-muted mb-4">Configure data augmentation to increase dataset variety</p>

            <!-- Output Count -->
            <div class="mb-4">
                <label class="form-label fw-bold">Augmentations per Image</label>
                <input type="number" class="form-control" id="output-count" value="1" min="1" max="10">
                <small class="text-muted">Number of augmented versions to generate for each image</small>
            </div>

            <!-- Geometric Transformations -->
            <div class="card mb-3">
                <div class="card-header">
                    <h6 class="mb-0 fw-bold">Geometric Transformations</h6>
                </div>
                <div class="card-body">
                    <div class="row g-3">
                        <div class="col-md-6">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="flip-horizontal">
                                <label class="form-check-label" for="flip-horizontal">
                                    Horizontal Flip
                                </label>
                            </div>
                            <input type="range" class="form-range" id="flip-horizontal-prob" min="0" max="100" value="50" disabled>
                            <small class="text-muted">Probability: <span id="flip-h-val">50</span>%</small>
                        </div>
                        <div class="col-md-6">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="flip-vertical">
                                <label class="form-check-label" for="flip-vertical">
                                    Vertical Flip
                                </label>
                            </div>
                            <input type="range" class="form-range" id="flip-vertical-prob" min="0" max="100" value="50" disabled>
                            <small class="text-muted">Probability: <span id="flip-v-val">50</span>%</small>
                        </div>
                    </div>

                    <div class="row g-3 mt-3">
                        <div class="col-md-12">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="enable-rotation">
                                <label class="form-check-label fw-bold" for="enable-rotation">
                                    Rotation
                                </label>
                            </div>
                            <div id="rotation-options" style="display: none;">
                                <div class="row g-2 mt-2">
                                    <div class="col-md-6">
                                        <label class="form-label small">Min Degrees</label>
                                        <input type="number" class="form-control" id="rotate-min" value="-15">
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label small">Max Degrees</label>
                                        <input type="number" class="form-control" id="rotate-max" value="15">
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Color Adjustments -->
            <div class="card mb-3">
                <div class="card-header">
                    <h6 class="mb-0 fw-bold">Color Adjustments</h6>
                </div>
                <div class="card-body">
                    <div class="row g-3">
                        <div class="col-md-6">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="enable-brightness">
                                <label class="form-check-label" for="enable-brightness">
                                    Brightness
                                </label>
                            </div>
                            <input type="range" class="form-range" id="brightness-range" min="0" max="50" value="25" disabled>
                            <small class="text-muted">±<span id="brightness-val">25</span>%</small>
                        </div>
                        <div class="col-md-6">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="enable-saturation">
                                <label class="form-check-label" for="enable-saturation">
                                    Saturation
                                </label>
                            </div>
                            <input type="range" class="form-range" id="saturation-range" min="0" max="50" value="25" disabled>
                            <small class="text-muted">±<span id="saturation-val">25</span>%</small>
                        </div>
                    </div>

                    <div class="row g-3 mt-3">
                        <div class="col-md-6">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="enable-hue">
                                <label class="form-check-label" for="enable-hue">
                                    Hue
                                </label>
                            </div>
                            <input type="range" class="form-range" id="hue-range" min="0" max="50" value="25" disabled>
                            <small class="text-muted">±<span id="hue-val">25</span>%</small>
                        </div>
                        <div class="col-md-6">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="enable-exposure">
                                <label class="form-check-label" for="enable-exposure">
                                    Exposure
                                </label>
                            </div>
                            <input type="range" class="form-range" id="exposure-range" min="0" max="50" value="25" disabled>
                            <small class="text-muted">±<span id="exposure-val">25</span>%</small>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Noise and Effects -->
            <div class="card">
                <div class="card-header">
                    <h6 class="mb-0 fw-bold">Noise & Effects</h6>
                </div>
                <div class="card-body">
                    <div class="row g-3">
                        <div class="col-md-6">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="enable-blur">
                                <label class="form-check-label" for="enable-blur">
                                    Gaussian Blur
                                </label>
                            </div>
                            <input type="range" class="form-range" id="blur-range" min="0" max="5" step="0.5" value="2.5" disabled>
                            <small class="text-muted">Max: <span id="blur-val">2.5</span>px</small>
                        </div>
                        <div class="col-md-6">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="enable-noise">
                                <label class="form-check-label" for="enable-noise">
                                    Add Noise
                                </label>
                            </div>
                            <input type="range" class="form-range" id="noise-range" min="0" max="10" value="5" disabled>
                            <small class="text-muted">Max: <span id="noise-val">5</span>%</small>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    attachModalEventListeners() {
        // Bootstrap tabs work automatically with data-bs-toggle="tab"
        // No manual initialization needed

        // Split inputs - update preview
        const updateSplitPreview = () => {
            const train = parseInt(document.getElementById('train-split')?.value || 70);
            const valid = parseInt(document.getElementById('valid-split')?.value || 20);
            const test = parseInt(document.getElementById('test-split')?.value || 10);
            const total = train + valid + test;

            // Update progress bars
            document.getElementById('train-bar').style.width = train + '%';
            document.getElementById('valid-bar').style.width = valid + '%';
            document.getElementById('test-bar').style.width = test + '%';

            document.getElementById('train-count').textContent = `Train: ${train}%`;
            document.getElementById('valid-count').textContent = `Valid: ${valid}%`;
            document.getElementById('test-count').textContent = `Test: ${test}%`;

            // Show warning if not 100%
            const warning = document.getElementById('split-warning');
            if (warning) {
                if (total !== 100) {
                    warning.style.display = 'block';
                    warning.innerHTML = `<i class="bi bi-exclamation-triangle me-2"></i>Total is ${total}% (must be 100%)`;
                } else {
                    warning.style.display = 'none';
                }
            }
        };

        ['train-split', 'valid-split', 'test-split'].forEach(id => {
            const elem = document.getElementById(id);
            if (elem) elem.addEventListener('input', updateSplitPreview);
        });

        // Resize toggle
        const enableResize = document.getElementById('enable-resize');
        const resizeOptions = document.getElementById('resize-options');
        if (enableResize && resizeOptions) {
            enableResize.addEventListener('change', (e) => {
                resizeOptions.style.display = e.target.checked ? 'block' : 'none';
            });
        }

        // Rotation toggle
        const enableRotation = document.getElementById('enable-rotation');
        const rotationOptions = document.getElementById('rotation-options');
        if (enableRotation && rotationOptions) {
            enableRotation.addEventListener('change', (e) => {
                rotationOptions.style.display = e.target.checked ? 'block' : 'none';
            });
        }

        // Flip checkboxes enable/disable sliders
        const setupFlipToggle = (checkId, sliderId, valId) => {
            const check = document.getElementById(checkId);
            const slider = document.getElementById(sliderId);
            if (check && slider) {
                check.addEventListener('change', (e) => {
                    slider.disabled = !e.target.checked;
                });
                slider.addEventListener('input', (e) => {
                    const val = document.getElementById(valId);
                    if (val) val.textContent = e.target.value;
                });
            }
        };

        setupFlipToggle('flip-horizontal', 'flip-horizontal-prob', 'flip-h-val');
        setupFlipToggle('flip-vertical', 'flip-vertical-prob', 'flip-v-val');

        // Color adjustment toggles
        ['brightness', 'saturation', 'hue', 'exposure', 'blur', 'noise'].forEach(name => {
            const check = document.getElementById(`enable-${name}`);
            const slider = document.getElementById(`${name}-range`);
            const valSpan = document.getElementById(`${name}-val`);

            if (check && slider) {
                check.addEventListener('change', (e) => {
                    slider.disabled = !e.target.checked;
                });
            }

            if (slider && valSpan) {
                slider.addEventListener('input', (e) => {
                    valSpan.textContent = e.target.value;
                });
            }
        });

        // Create button
        const createBtn = document.getElementById('create-version-btn');
        if (createBtn) {
            createBtn.addEventListener('click', () => this.handleCreateVersion());
        }
    }

    async handleCreateVersion() {
        const name = document.getElementById('version-name')?.value;
        const description = document.getElementById('version-description')?.value;

        if (!name) {
            showToast('Please enter a version name', 'error');
            return;
        }

        // Validate splits
        const trainSplit = parseInt(document.getElementById('train-split')?.value || 70) / 100;
        const validSplit = parseInt(document.getElementById('valid-split')?.value || 20) / 100;
        const testSplit = parseInt(document.getElementById('test-split')?.value || 10) / 100;

        if (Math.abs(trainSplit + validSplit + testSplit - 1.0) > 0.001) {
            showToast('Data splits must sum to 100%', 'error');
            return;
        }

        // Build preprocessing config
        const preprocessingConfig = {
            auto_orient: document.getElementById('auto-orient')?.checked || false,
            normalize_contrast: document.getElementById('normalize-contrast')?.checked || false,
            normalize_exposure: document.getElementById('normalize-exposure')?.checked || false,
            filter_null_images: document.getElementById('filter-null')?.checked || true,
        };

        if (document.getElementById('enable-resize')?.checked) {
            preprocessingConfig.resize = {
                width: parseInt(document.getElementById('resize-width')?.value || 640),
                height: parseInt(document.getElementById('resize-height')?.value || 640),
                mode: document.getElementById('resize-mode')?.value || 'fit'
            };
        }

        // Build augmentation config
        const augmentationConfig = {
            output_count: parseInt(document.getElementById('output-count')?.value || 1)
        };

        if (document.getElementById('flip-horizontal')?.checked) {
            augmentationConfig.flip_horizontal = parseInt(document.getElementById('flip-horizontal-prob')?.value || 50) / 100;
        }

        if (document.getElementById('flip-vertical')?.checked) {
            augmentationConfig.flip_vertical = parseInt(document.getElementById('flip-vertical-prob')?.value || 50) / 100;
        }

        if (document.getElementById('enable-rotation')?.checked) {
            augmentationConfig.rotate = {
                min: parseInt(document.getElementById('rotate-min')?.value || -15),
                max: parseInt(document.getElementById('rotate-max')?.value || 15)
            };
        }

        if (document.getElementById('enable-brightness')?.checked) {
            const val = parseInt(document.getElementById('brightness-range')?.value || 25);
            augmentationConfig.brightness = { min: -val, max: val };
        }

        if (document.getElementById('enable-saturation')?.checked) {
            const val = parseInt(document.getElementById('saturation-range')?.value || 25);
            augmentationConfig.saturation = { min: -val, max: val };
        }

        if (document.getElementById('enable-hue')?.checked) {
            const val = parseInt(document.getElementById('hue-range')?.value || 25);
            augmentationConfig.hue = { min: -val, max: val };
        }

        if (document.getElementById('enable-exposure')?.checked) {
            const val = parseInt(document.getElementById('exposure-range')?.value || 25);
            augmentationConfig.exposure = { min: -val, max: val };
        }

        if (document.getElementById('enable-blur')?.checked) {
            augmentationConfig.blur = {
                max: parseFloat(document.getElementById('blur-range')?.value || 2.5)
            };
        }

        if (document.getElementById('enable-noise')?.checked) {
            augmentationConfig.noise = {
                max: parseInt(document.getElementById('noise-range')?.value || 5)
            };
        }

        const versionData = {
            dataset_id: this.selectedDataset.id,
            name,
            description,
            train_split: trainSplit,
            valid_split: validSplit,
            test_split: testSplit,
            preprocessing_config: preprocessingConfig,
            augmentation_config: augmentationConfig
        };

        console.log('[GeneratePage] Creating version:', versionData);

        try {
            const createBtn = document.getElementById('create-version-btn');
            createBtn.disabled = true;
            createBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Creating...';

            await apiService.createVersion(this.selectedDataset.id, versionData);

            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('createVersionModal'));
            if (modal) modal.hide();

            showToast('Version creation started! Processing in background...', 'success');

            // Reload versions
            await this.loadVersions(this.selectedDataset.id);
            const app = document.getElementById('app');
            if (app) {
                app.innerHTML = this.render();
                this.attachEventListeners();
            }

        } catch (error) {
            console.error('[GeneratePage] Error creating version:', error);
            showToast('Failed to create version: ' + error.message, 'error');

            const createBtn = document.getElementById('create-version-btn');
            if (createBtn) {
                createBtn.disabled = false;
                createBtn.innerHTML = '<i class="bi bi-check-circle me-1"></i>Create Version';
            }
        }
    }

    async exportVersion(versionId) {
        console.log('Export version', versionId);
        showToast('Export functionality - to be implemented', 'info');
    }

    async deleteVersion(versionId) {
        if (!confirm('Are you sure you want to delete this version?')) return;

        try {
            await apiService.deleteVersion(versionId);
            showToast('Version deleted successfully', 'success');
            await this.loadVersions(this.selectedDataset.id);
            const app = document.getElementById('app');
            if (app) {
                app.innerHTML = this.render();
                this.attachEventListeners();
            }
        } catch (error) {
            showToast('Failed to delete version: ' + error.message, 'error');
        }
    }
}

console.log('[GeneratePage] Module loaded');
