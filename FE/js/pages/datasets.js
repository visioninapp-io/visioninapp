// Datasets Page Component
// Enhanced with robust error handling and debugging

class DatasetsPage {
    constructor() {
        console.log('[DatasetsPage] Initializing...');
        this.datasets = [];
        this.selectedDataset = null;
        this.datasetImages = [];
        this.stats = {
            totalImages: 0,
            totalDatasets: 0,
            totalClasses: 0,
            autoAnnotatedPercent: 0
        };
        this.searchQuery = '';
        this.isLoading = true;
        this.loadError = null;
    }

    async init() {
        console.log('[DatasetsPage] Starting initialization...');
        this.isLoading = true;
        this.loadError = null;

        // Clear selected dataset to prevent showing deleted datasets
        this.selectedDataset = null;
        this.datasetImages = [];

        try {
            // Load data in parallel
            await Promise.all([
                this.loadDatasets(),
                this.loadStats()
            ]);

            console.log('[DatasetsPage] Data loaded successfully');
            this.isLoading = false;

        } catch (error) {
            console.error('[DatasetsPage] Initialization failed:', error);
            this.loadError = error.message;
            this.isLoading = false;
        }

        // Re-render the page after data is loaded
        const app = document.getElementById('app');
        if (app) {
            app.innerHTML = this.render();
            this.attachEventListeners();

            // Load presigned URLs for images after rendering
            if (this.datasetImages.length > 0) {
                console.log('[DatasetsPage] Loading presigned URLs after initial render...');
                this.loadImageUrls();
            }
        }
    }

    async loadDatasets() {
        try {
            console.log('[DatasetsPage] Loading datasets from API...');
            const datasets = await apiService.getDatasets();

            if (!datasets || !Array.isArray(datasets)) {
                console.warn('[DatasetsPage] Invalid datasets response:', datasets);
                this.datasets = [];
                this.selectedDataset = null;
                this.datasetImages = [];
                return;
            }

            this.datasets = datasets;
            console.log(`[DatasetsPage] Loaded ${this.datasets.length} datasets:`, this.datasets);

            // Load label classes for each dataset
            await Promise.all(this.datasets.map(async (dataset) => {
                try {
                    dataset.label_classes = await apiService.getDatasetLabelClasses(dataset.id);
                } catch (error) {
                    console.error(`[DatasetsPage] Failed to load label classes for dataset ${dataset.id}:`, error);
                    dataset.label_classes = [];
                }
            }));

            // Auto-select first dataset if available
            if (this.datasets.length > 0) {
                // If selected dataset exists, check if it's still in the list
                if (this.selectedDataset) {
                    const stillExists = this.datasets.find(d => d.id === this.selectedDataset.id);
                    if (!stillExists) {
                        console.log('[DatasetsPage] Previously selected dataset no longer exists, selecting first dataset');
                        this.selectedDataset = this.datasets[0];
                    }
                } else {
                    this.selectedDataset = this.datasets[0];
                    console.log('[DatasetsPage] Auto-selected dataset:', this.selectedDataset);
                }

                await this.loadDatasetImages(this.selectedDataset.id);
            } else {
                // No datasets available
                this.selectedDataset = null;
                this.datasetImages = [];
            }

        } catch (error) {
            console.error('[DatasetsPage] Error loading datasets:', error);
            this.datasets = [];
            this.selectedDataset = null;
            this.datasetImages = [];
            showToast('Failed to load datasets. Using empty dataset list.', 'warning');
        }
    }

    async loadDatasetImages(datasetId) {
        try {
            console.log(`[DatasetsPage] Loading images for dataset ${datasetId}...`);
            const images = await apiService.getDatasetImages(datasetId);

            if (!images || !Array.isArray(images)) {
                console.warn('[DatasetsPage] Invalid images response:', images);
                this.datasetImages = [];
                return;
            }

            this.datasetImages = images;
            console.log(`[DatasetsPage] Loaded ${this.datasetImages.length} images`);

            // Update the image grid if we're already rendered
            this.updateImageGrid();

        } catch (error) {
            console.error(`[DatasetsPage] Error loading images for dataset ${datasetId}:`, error);
            this.datasetImages = [];
            this.updateImageGrid();
        }
    }

    async loadStats() {
        try {
            console.log('[DatasetsPage] Loading dataset stats...');
            const statsData = await apiService.getDatasetStats();

            if (statsData) {
                this.stats = {
                    totalImages: statsData.total_images || 0,
                    totalDatasets: statsData.total_datasets || 0,
                    totalClasses: statsData.total_classes || 0,
                    autoAnnotatedPercent: Math.round(statsData.auto_annotation_rate || 0)
                };
                console.log('[DatasetsPage] Loaded stats:', this.stats);
            }

        } catch (error) {
            console.error('[DatasetsPage] Error loading stats:', error);
            // Calculate stats from datasets if API fails
            this.calculateStatsFromDatasets();
        }
    }

    calculateStatsFromDatasets() {
        console.log('[DatasetsPage] Calculating stats from datasets...');
        this.stats.totalDatasets = this.datasets.length;
        this.stats.totalImages = this.datasets.reduce((sum, d) => sum + (d.total_images || 0), 0);
        this.stats.totalClasses = this.datasets.reduce((sum, d) => sum + (d.total_classes || 0), 0);

        const annotatedImages = this.datasets.reduce((sum, d) => sum + (d.annotated_images || 0), 0);
        this.stats.autoAnnotatedPercent = this.stats.totalImages > 0
            ? Math.round((annotatedImages / this.stats.totalImages) * 100)
            : 0;

        console.log('[DatasetsPage] Calculated stats:', this.stats);
    }

    render() {
        console.log('[DatasetsPage] Rendering page...');
        console.log('[DatasetsPage] Current state:', {
            isLoading: this.isLoading,
            datasetsCount: this.datasets.length,
            selectedDataset: this.selectedDataset?.name,
            loadError: this.loadError
        });

        // Show loading state
        if (this.isLoading) {
            return this.renderLoadingState();
        }

        // Show error state
        if (this.loadError) {
            return this.renderErrorState();
        }

        return `
            <div class="min-vh-100 bg-light">
                <div class="container-fluid py-4">
                    <!-- Page Header -->
                    <div class="mb-4">
                        <h1 class="display-5 fw-bold mb-2">Dataset Management</h1>
                        <p class="text-muted">Upload, preprocess, and auto-annotate your training data</p>
                    </div>

                    <!-- Stats Cards -->
                    <div class="row g-4 mb-4">
                        <div class="col-md-3">
                            <div class="card border-0 shadow-sm">
                                <div class="card-body">
                                    <h6 class="text-muted mb-3">Total Images</h6>
                                    <h2 class="fw-bold mb-2" id="stat-total-images">${this.stats.totalImages.toLocaleString()}</h2>
                                    <p class="text-muted small mb-0">Across all datasets</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card border-0 shadow-sm">
                                <div class="card-body">
                                    <h6 class="text-muted mb-3">Datasets</h6>
                                    <h2 class="fw-bold mb-2" id="stat-datasets">${this.stats.totalDatasets}</h2>
                                    <p class="text-muted small mb-0">Active projects</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card border-0 shadow-sm">
                                <div class="card-body">
                                    <h6 class="text-muted mb-3">Classes</h6>
                                    <h2 class="fw-bold mb-2" id="stat-classes">${this.stats.totalClasses}</h2>
                                    <p class="text-muted small mb-0">Across all datasets</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card border-0 shadow-sm">
                                <div class="card-body">
                                    <h6 class="text-muted mb-3">Annotated</h6>
                                    <h2 class="fw-bold mb-2" id="stat-annotated">${this.stats.autoAnnotatedPercent}%</h2>
                                    <p class="text-info small mb-0">Completion rate</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Dataset Selection and Actions -->
                    <div class="row mb-4">
                        <div class="col-md-6">
                            <div class="card border-0 shadow-sm">
                                <div class="card-body">
                                    <label class="form-label fw-bold">
                                        <i class="bi bi-database me-2"></i>Select Dataset
                                    </label>
                                    <select class="form-select form-select-lg" id="dataset-selector">
                                        ${this.datasets.length === 0 ?
                                            '<option value="">No datasets available</option>' :
                                            this.datasets.map(dataset => `
                                                <option value="${dataset.id}" ${this.selectedDataset && this.selectedDataset.id === dataset.id ? 'selected' : ''}>
                                                    ${dataset.name} (${dataset.total_assets || dataset.total_images || 0} images)
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
                                    <label class="form-label fw-bold">
                                        <i class="bi bi-gear me-2"></i>Actions
                                    </label>
                                    <div class="d-flex gap-2">
                                        <button class="btn btn-outline-primary flex-fill" onclick="showConfigureDatasetModal()">
                                            <i class="bi bi-plus-circle me-1"></i> Create New
                                        </button>
                                        <button class="btn btn-primary flex-fill" onclick="showUploadDatasetModal()">
                                            <i class="bi bi-upload me-1"></i> Upload Images
                                        </button>
                                        <button class="btn btn-success flex-fill" onclick="window.currentPageInstance.triggerAutoAnnotate()" ${!this.selectedDataset ? 'disabled' : ''}>
                                            <i class="bi bi-sparkles me-1"></i> Auto-Annotate
                                        </button>
                                        <button class="btn btn-warning flex-fill" onclick="window.currentPageInstance.triggerSelfAnnotate()" ${!this.selectedDataset ? 'disabled' : ''}>
                                            <i class="bi bi-pencil-square me-1"></i> Self-Annotate
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Dataset Info and Images -->
                    ${this.datasets.length > 0 && this.selectedDataset ? this.renderDatasetDetail() : this.renderEmptyState()}
                </div>
            </div>
        `;
    }

    renderLoadingState() {
        return `
            <div class="min-vh-100 bg-light d-flex align-items-center justify-content-center">
                <div class="text-center">
                    <div class="spinner-border text-primary mb-3" style="width: 4rem; height: 4rem;" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <h4 class="text-muted">Loading datasets...</h4>
                    <p class="text-muted small">Please wait while we fetch your data</p>
                </div>
            </div>
        `;
    }

    renderErrorState() {
        return `
            <div class="min-vh-100 bg-light d-flex align-items-center justify-content-center">
                <div class="text-center">
                    <i class="bi bi-exclamation-triangle text-danger mb-3" style="font-size: 4rem;"></i>
                    <h4 class="text-danger">Error Loading Datasets</h4>
                    <p class="text-muted">${this.loadError || 'Unknown error occurred'}</p>
                    <button class="btn btn-primary" onclick="window.location.reload()">
                        <i class="bi bi-arrow-clockwise me-1"></i> Retry
                    </button>
                </div>
            </div>
        `;
    }

    renderDatasetDetail() {
        const dataset = this.selectedDataset;
        const statusMap = {
            'created': { class: 'bg-secondary', text: 'Created' },
            'uploading': { class: 'bg-info', text: 'Uploading' },
            'processing': { class: 'bg-warning', text: 'Processing' },
            'ready': { class: 'bg-success', text: 'Ready' },
            'READY': { class: 'bg-success', text: 'Ready' },
            'annotated': { class: 'bg-primary', text: 'Annotated' },
            'error': { class: 'bg-danger', text: 'Error' }
        };
        const status = statusMap[dataset.status] || statusMap['created'];

        return `
            <div class="row">
                <!-- Dataset Info -->
                <div class="col-md-4">
                    <div class="card border-0 shadow-sm mb-4">
                        <div class="card-header bg-white">
                            <h5 class="mb-0 fw-bold">
                                <i class="bi bi-info-circle me-2"></i>Dataset Information
                            </h5>
                        </div>
                        <div class="card-body">
                            <h4 class="fw-bold mb-3">${dataset.name}</h4>
                            ${dataset.description ? `<p class="text-muted mb-3">${dataset.description}</p>` : ''}

                            <div class="mb-3">
                                <span class="badge ${status.class} mb-2">${status.text}</span>
                            </div>

                            <div class="border-top pt-3">
                                <div class="row mb-2">
                                    <div class="col-6 text-muted">Total Images:</div>
                                    <div class="col-6 fw-bold text-end">${dataset.total_images || 0}</div>
                                </div>
                                <div class="row mb-2">
                                    <div class="col-6 text-muted">Annotated:</div>
                                    <div class="col-6 fw-bold text-end text-success">${dataset.annotated_images || 0}</div>
                                </div>
                                <div class="row mb-2">
                                    <div class="col-6 text-muted">Classes:</div>
                                    <div class="col-6 fw-bold text-end">${dataset.total_classes || 0}</div>
                                </div>
                                <div class="row mb-2">
                                    <div class="col-6 text-muted">Created:</div>
                                    <div class="col-6 text-end small">${this.formatDate(dataset.created_at)}</div>
                                </div>
                                <div class="row">
                                    <div class="col-6 text-muted">Updated:</div>
                                    <div class="col-6 text-end small">${this.formatDate(dataset.updated_at)}</div>
                                </div>
                            </div>

                            ${dataset.label_classes && dataset.label_classes.length > 0 ? `
                                <div class="border-top pt-3 mt-3">
                                    <h6 class="fw-bold mb-2">Classes (${dataset.label_classes.length}):</h6>
                                    <div class="d-flex flex-wrap gap-1">
                                        ${dataset.label_classes.map(labelClass =>
                                            `<span class="badge" style="background-color: ${labelClass.color}">${labelClass.display_name}</span>`
                                        ).join('')}
                                    </div>
                                </div>
                            ` : ''}

                            <div class="d-grid gap-2 mt-3">
                                <button class="btn btn-outline-primary" onclick="navigateToDatasetDetail(${dataset.id})">
                                    <i class="bi bi-eye me-1"></i> View Full Details
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Images Grid -->
                <div class="col-md-8">
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-white">
                            <div class="row align-items-center">
                                <div class="col">
                                    <h5 class="mb-0 fw-bold">
                                        <i class="bi bi-images me-2"></i>Dataset Images
                                        <span class="badge bg-primary ms-2">${this.datasetImages.length}</span>
                                    </h5>
                                </div>
                                <div class="col-auto">
                                    <button class="btn btn-outline-primary btn-sm" onclick="window.currentPageInstance.refreshImages()">
                                        <i class="bi bi-arrow-clockwise me-1"></i> Refresh
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div class="card-body" id="images-grid-container" style="max-height: 600px; overflow-y: auto;">
                            ${this.renderImageGrid()}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderImageGrid() {
        if (this.datasetImages.length === 0) {
            return `
                <div class="text-center py-5">
                    <i class="bi bi-image text-muted" style="font-size: 4rem;"></i>
                    <h5 class="mt-3 text-muted">No Images in Dataset</h5>
                    <p class="text-muted">Upload images to this dataset to get started</p>
                    <button class="btn btn-primary" onclick="showUploadDatasetModal()">
                        <i class="bi bi-upload me-1"></i> Upload Images
                    </button>
                </div>
            `;
        }

        return `
            <div class="row g-3">
                ${this.datasetImages.map(image => this.renderImageCard(image)).join('')}
            </div>
        `;
    }

    renderImageCard(image) {
        const isAnnotated = image.is_annotated;
        const imageId = `image-${image.id}`;

        return `
            <div class="col-md-4 col-lg-3">
                <div class="card border ${isAnnotated ? 'border-success' : ''} hover-shadow">
                    <div class="position-relative">
                        <img id="${imageId}" class="card-img-top" alt="${image.filename}"
                             style="height: 200px; object-fit: cover;"
                             src="https://via.placeholder.com/300x200?text=Loading..."
                             data-asset-id="${image.id}">
                        ${isAnnotated ? `
                            <span class="position-absolute top-0 end-0 m-2">
                                <span class="badge bg-success">
                                    <i class="bi bi-check-circle-fill"></i> Annotated
                                </span>
                            </span>
                        ` : ''}
                    </div>
                    <div class="card-body p-2">
                        <p class="mb-1 small text-truncate" title="${image.filename}">
                            <i class="bi bi-file-image me-1"></i>${image.filename}
                        </p>
                        <p class="mb-0 text-muted" style="font-size: 0.75rem;">
                            ${image.width || 'N/A'}x${image.height || 'N/A'} • ${this.formatFileSize(image.file_size)}
                        </p>
                    </div>
                </div>
            </div>
        `;
    }

    renderEmptyState() {
        return `
            <div class="card border-0 shadow-sm">
                <div class="card-body text-center py-5">
                    <i class="bi bi-database text-muted" style="font-size: 5rem;"></i>
                    <h3 class="mt-4 mb-3">No Datasets Yet</h3>
                    <p class="text-muted mb-4">Create your first dataset to start managing training data</p>
                    <div class="d-flex gap-2 justify-content-center">
                        <button class="btn btn-primary btn-lg" onclick="showConfigureDatasetModal()">
                            <i class="bi bi-plus-circle me-2"></i> Create Dataset
                        </button>
                        <button class="btn btn-outline-primary btn-lg" onclick="showUploadDatasetModal()">
                            <i class="bi bi-upload me-2"></i> Upload Images
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    formatDate(dateString) {
        if (!dateString) return 'N/A';
        try {
            const date = new Date(dateString);
            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } catch {
            return 'N/A';
        }
    }

    formatFileSize(bytes) {
        if (!bytes || bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    attachEventListeners() {
        console.log('[DatasetsPage] Attaching event listeners...');

        // Dataset selector
        const datasetSelector = document.getElementById('dataset-selector');
        if (datasetSelector) {
            datasetSelector.addEventListener('change', async (e) => {
                const datasetId = parseInt(e.target.value);
                const dataset = this.datasets.find(d => d.id === datasetId);

                if (dataset) {
                    console.log(`[DatasetsPage] Dataset changed to: ${dataset.name}`);
                    this.selectedDataset = dataset;
                    await this.loadDatasetImages(datasetId);

                    // Re-render the page
                    const app = document.getElementById('app');
                    if (app) {
                        app.innerHTML = this.render();
                        this.attachEventListeners();
                    }
                }
            });
        }
    }

    updateImageGrid() {
        const container = document.getElementById('images-grid-container');
        if (container) {
            console.log('[DatasetsPage] Updating image grid...');
            container.innerHTML = this.renderImageGrid();

            // Load presigned URLs for all images
            this.loadImageUrls();
        }
    }

    async loadImageUrls() {
        if (this.datasetImages.length === 0) return;

        console.log(`[DatasetsPage] Loading presigned URLs for ${this.datasetImages.length} images...`);

        // Load URLs in parallel
        const urlPromises = this.datasetImages.map(async (image) => {
            try {
                const response = await apiService.get(`/datasets/assets/${image.id}/presigned-download`);

                if (response && response.download_url) {
                    const imgElement = document.getElementById(`image-${image.id}`);
                    if (imgElement) {
                        imgElement.src = response.download_url;
                    }
                }
            } catch (error) {
                console.error(`[DatasetsPage] Failed to load URL for image ${image.id}:`, error);
                const imgElement = document.getElementById(`image-${image.id}`);
                if (imgElement) {
                    imgElement.src = 'https://via.placeholder.com/300x200?text=Load+Failed';
                }
            }
        });

        await Promise.all(urlPromises);
        console.log('[DatasetsPage] All image URLs loaded');
    }

    async refreshImages() {
        if (this.selectedDataset) {
            console.log(`[DatasetsPage] Refreshing images for dataset ${this.selectedDataset.id}...`);
            await this.loadDatasetImages(this.selectedDataset.id);
        }
    }

    async triggerAutoAnnotate() {
        if (!this.selectedDataset) {
            showToast('Please select a dataset first', 'error');
            return;
        }

        console.log(`[DatasetsPage] Navigating to auto-annotate for dataset ${this.selectedDataset.id}...`);
        navigateToAutoAnnotate(this.selectedDataset.id);
    }

    async triggerSelfAnnotate() {
        if (!this.selectedDataset) {
            showToast('Please select a dataset first', 'error');
            return;
        }

        console.log(`[DatasetsPage] Opening self-annotation modal for dataset ${this.selectedDataset.id}...`);
        // Show self-annotation modal
        if (typeof showSelfAnnotationModal === 'function') {
            showSelfAnnotationModal(this.selectedDataset.id, this.selectedDataset.name);
        } else {
            console.error('[DatasetsPage] showSelfAnnotationModal function not found');
            showToast('Self-annotation feature not loaded', 'error');
        }
    }

    async reloadDatasets(datasetIdToSelect = null) {
        console.log('[DatasetsPage] Reloading datasets...', datasetIdToSelect ? `Select: ${datasetIdToSelect}` : '');

        // Reload datasets and stats
        await Promise.all([
            this.loadDatasets(),
            this.loadStats()
        ]);

        // If a specific dataset ID was provided, select it
        if (datasetIdToSelect) {
            const dataset = this.datasets.find(d => d.id === datasetIdToSelect);
            if (dataset) {
                this.selectedDataset = dataset;
                await this.loadDatasetImages(dataset.id);
            }
        }

        // Re-render the page
        const app = document.getElementById('app');
        if (app) {
            app.innerHTML = this.render();
            this.attachEventListeners();
        }
    }
}

// Configure Dataset Modal
function showConfigureDatasetModal() {
    console.log('[DatasetsPage] Showing configure dataset modal...');
    const modalHTML = `
        <div class="modal fade" id="configureDatasetModal" tabindex="-1" aria-hidden="true">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="bi bi-gear me-2"></i>Configure Dataset
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <form id="configure-dataset-form">
                            <div class="mb-3">
                                <label for="dataset-name" class="form-label">Dataset Name *</label>
                                <input type="text" class="form-control" id="dataset-name" required
                                       placeholder="e.g., Factory Defect Detection">
                            </div>

                            <div class="mb-3">
                                <label for="dataset-description" class="form-label">Description</label>
                                <textarea class="form-control" id="dataset-description" rows="3"
                                          placeholder="Describe the purpose and contents of this dataset"></textarea>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Class Definitions</label>
                                <div id="class-definitions-container">
                                    <div class="input-group mb-2">
                                        <input type="text" class="form-control" placeholder="Class name (e.g., defect)"
                                               data-class-input>
                                        <button class="btn btn-outline-danger" type="button" onclick="this.parentElement.remove()">
                                            <i class="bi bi-trash"></i>
                                        </button>
                                    </div>
                                </div>
                                <button type="button" class="btn btn-outline-primary btn-sm" id="add-class-btn">
                                    <i class="bi bi-plus-circle me-1"></i> Add Class
                                </button>
                            </div>
                        </form>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" id="save-dataset-config">
                            <i class="bi bi-check-circle me-1"></i> Save Configuration
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    const existingModal = document.getElementById('configureDatasetModal');
    if (existingModal) existingModal.remove();

    document.body.insertAdjacentHTML('beforeend', modalHTML);
    const modal = new bootstrap.Modal(document.getElementById('configureDatasetModal'));

    document.getElementById('add-class-btn').addEventListener('click', () => {
        const container = document.getElementById('class-definitions-container');
        const inputGroup = `
            <div class="input-group mb-2">
                <input type="text" class="form-control" placeholder="Class name" data-class-input>
                <button class="btn btn-outline-danger" type="button" onclick="this.parentElement.remove()">
                    <i class="bi bi-trash"></i>
                </button>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', inputGroup);
    });

    document.getElementById('save-dataset-config').addEventListener('click', async () => {
        const name = document.getElementById('dataset-name').value;
        const description = document.getElementById('dataset-description').value;

        const classInputs = document.querySelectorAll('[data-class-input]');
        const classNames = Array.from(classInputs)
            .map(input => input.value.trim())
            .filter(name => name !== '');

        if (!name) {
            showToast('Please enter a dataset name', 'error');
            return;
        }

        if (classNames.length === 0) {
            showToast('Please define at least one class', 'error');
            return;
        }

        try {
            const datasetData = {
                name,
                description,
                total_classes: classNames.length,
                class_names: classNames,
                status: 'created'
            };

            console.log('[DatasetsPage] Creating dataset:', datasetData);
            const createdDataset = await apiService.createDataset(datasetData);

            modal.hide();
            showToast('Dataset configured successfully!', 'success');

            // Reload datasets without page refresh and select the new dataset
            if (window.currentPageInstance && typeof window.currentPageInstance.reloadDatasets === 'function') {
                await window.currentPageInstance.reloadDatasets(createdDataset.id);
            }
        } catch (error) {
            console.error('[DatasetsPage] Error creating dataset:', error);
            showToast('Failed to create dataset: ' + error.message, 'error');
        }
    });

    modal.show();
}

console.log('[DatasetsPage] Module loaded');
