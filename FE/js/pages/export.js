// Export Page Component

class ExportPage {
    constructor() {
        console.log('[ExportPage] Initializing...');
        this.datasets = [];
        this.models = [];
        this.versions = [];
        this.exportJobs = [];
        this.selectedDataset = null;
        this.selectedModel = null;
        this.selectedVersion = null;
        this.isLoading = true;
        this.datasetNameMap = {};
        this.modelNameMap = {};
        this.exportType = 'dataset'; // 'dataset' or 'model'
        this.pendingDeleteExportId = null; // For delete confirmation modal
    }

    async init() {
        console.log('[ExportPage] Starting initialization...');
        try {
            await Promise.all([
                this.loadDatasets(),
                this.loadModels(),
                this.loadExportJobs()
            ]);
            this.isLoading = false;
        } catch (error) {
            console.error('[ExportPage] Initialization failed:', error);
            this.isLoading = false;
        }

        const app = document.getElementById('app');
        if (app) {
            app.innerHTML = this.render();
            this.attachEventListeners();
        }

        // Auto-refresh export jobs
        this.startAutoRefresh();
    }

    async loadDatasets() {
        try {
            this.datasets = await apiService.getDatasets();

            this.datasetNameMap = {};
            if (Array.isArray(this.datasets)) {
                this.datasets.forEach(d => {
                    const name = d.name || d.display_name || `Dataset #${d.id}`;
                    this.datasetNameMap[d.id] = name;
                });
            }
        } catch (error) {
            console.error('Error loading datasets:', error);
            this.datasets = [];
        }
    }

    async loadModels() {
        try {
            this.models = await apiService.getModels();

            this.modelNameMap = {};
            if (Array.isArray(this.models)) {
                this.models.forEach(m => {
                    const name = m.name || `Model #${m.id}`;
                    this.modelNameMap[m.id] = name;
                });
            }
        } catch (error) {
            console.error('Error loading models:', error);
            this.models = [];
        }
    }

    async loadExportJobs() {
        try {
            this.exportJobs = await apiService.getExportJobs();
        } catch (error) {
            console.error('Error loading export jobs:', error);
            this.exportJobs = [];
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
                        <h1 class="display-5 fw-bold mb-2">Export Datasets & Models</h1>
                        <p class="text-muted">Export your datasets and models as ZIP files</p>
                    </div>

                    <!-- Export Form -->
                    <div class="row mb-4">
                        <div class="col-lg-8">
                            <div class="card border-0 shadow-sm">
                                <div class="card-header bg-white">
                                    <h5 class="mb-0 fw-bold">Create Export</h5>
                                </div>
                                <div class="card-body">
                                    <form id="export-form">
                                        <div class="row g-3">
                                            <!-- Export Type Selection -->
                                            <div class="col-12">
                                                <label class="form-label fw-bold">Export Type</label>
                                                <div class="btn-group w-100" role="group">
                                                    <input type="radio" class="btn-check" name="export-type" id="type-dataset" value="dataset" checked>
                                                    <label class="btn btn-outline-primary" for="type-dataset">
                                                        <i class="bi bi-folder2 me-2"></i>Dataset
                                                    </label>
                                                    
                                                    <input type="radio" class="btn-check" name="export-type" id="type-model" value="model">
                                                    <label class="btn btn-outline-primary" for="type-model">
                                                        <i class="bi bi-cpu me-2"></i>Model
                                                    </label>
                                                </div>
                                            </div>

                                            <!-- Dataset Selection (shown when dataset type is selected) -->
                                            <div class="col-md-12" id="dataset-selection">
                                                <label class="form-label fw-bold">Dataset</label>
                                                <select class="form-select" id="export-dataset">
                                                    <option value="">-- Select Dataset --</option>
                                                    ${this.datasets.map(d => {
                                                        const name = d.name || d.display_name || `Dataset #${d.id}`;
                                                        return `<option value="${d.id}">${name}</option>`;
                                                    }).join('')}
                                                </select>
                                            </div>

                                            <!-- Model Selection (hidden by default) -->
                                            <div class="col-md-12" id="model-selection" style="display: none;">
                                                <label class="form-label fw-bold">Model</label>
                                                <select class="form-select" id="export-model">
                                                    <option value="">-- Select Model --</option>
                                                    ${this.models.map(m => `
                                                        <option value="${m.id}">${m.name}</option>
                                                    `).join('')}
                                                </select>
                                            </div>

                                            <!-- Include Images (only for datasets) -->
                                            <div class="col-12" id="include-images-option">
                                                <div class="form-check">
                                                    <input class="form-check-input" type="checkbox" id="include-images" checked>
                                                    <label class="form-check-label" for="include-images">
                                                        Include images in export (annotations only if unchecked)
                                                    </label>
                                                </div>
                                            </div>

                                            <!-- Submit Button -->
                                            <div class="col-12">
                                                <button type="submit" class="btn btn-primary btn-lg w-100">
                                                    <i class="bi bi-file-earmark-zip me-2"></i>Create Export
                                                </button>
                                            </div>
                                        </div>
                                    </form>
                                </div>
                            </div>
                        </div>

                        <!-- Export Info -->
                        <div class="col-lg-4">
                            <div class="card border-0 shadow-sm h-100">
                                <div class="card-header bg-white">
                                    <h5 class="mb-0 fw-bold">Export Information</h5>
                                </div>
                                <div class="card-body d-flex flex-column">
                                    <h6 class="fw-bold mb-3">What gets exported?</h6>
                                    <ul class="small">
                                        <li><strong>Dataset</strong>: All files including images, annotations, and metadata in YOLO format</li>
                                        <li><strong>Model</strong>: All model files including weights, configs, and related artifacts</li>
                                    </ul>

                                    <div class="alert alert-info small mb-0 mt-auto">
                                        <i class="bi bi-info-circle me-1"></i>
                                        Exports are generated as ZIP files and can be downloaded once complete.
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Export Jobs List -->
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-white">
                            <h5 class="mb-0 fw-bold">Export Jobs</h5>
                        </div>
                        <div class="card-body">
                            ${this.renderExportJobs()}
                        </div>
                    </div>
                </div>
            </div>

            <!-- Delete Export Confirmation Modal -->
            <div class="modal fade" id="deleteExportModal" tabindex="-1" aria-labelledby="deleteExportModalLabel" aria-hidden="true">
                <div class="modal-dialog modal-dialog-centered">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="deleteExportModalLabel">
                                <i class="bi bi-exclamation-triangle-fill text-warning me-2"></i>Delete Export
                            </h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <p>Are you sure you want to delete this export?</p>
                            <p class="text-muted small mb-0">This action cannot be undone. The export file will be permanently deleted.</p>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-danger" id="confirmDeleteExportBtn">
                                <i class="bi bi-trash me-1"></i>Delete
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderExportJobs() {
        if (this.exportJobs.length === 0) {
            return `
                <div class="text-center py-5">
                    <i class="bi bi-inbox text-muted" style="font-size: 5rem;"></i>
                    <h5 class="mt-3 text-muted">No Export Jobs Yet</h5>
                    <p class="text-muted">Create an export to see it here</p>
                </div>
            `;
        }

        return `
            <div class="table-responsive">
                <table class="table table-hover">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Type</th>
                            <th>Status</th>
                            <th>Size</th>
                            <th>Created</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${this.exportJobs.map(job => this.renderExportJobRow(job)).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    renderExportJobRow(job) {
        const statusBadge = {
            'pending': 'bg-secondary',
            'processing': 'bg-warning',
            'completed': 'bg-success',
            'failed': 'bg-danger'
        }[job.status] || 'bg-secondary';

        const formatFileSize = (bytes) => {
            if (!bytes) return 'N/A';
            const mb = bytes / (1024 * 1024);
            return `${mb.toFixed(2)} MB`;
        };

        const formatDate = (dateString) => {
            if (!dateString) return 'N/A';
            return new Date(dateString).toLocaleString();
        };

        // Determine if this is a dataset or model export
        let itemName = '-';
        let itemType = 'Dataset';
        
        if (job.model_id) {
            // Model export
            itemType = 'Model';
            itemName = (this.modelNameMap && this.modelNameMap[job.model_id]) || `Model #${job.model_id}`;
        } else if (job.dataset_id || job.version_id) {
            // Dataset export
            itemType = 'Dataset';
            const datasetId = job.dataset_id || job.version_id;
            itemName = (this.datasetNameMap && this.datasetNameMap[datasetId]) || `Dataset #${datasetId}`;
        }

        return `
            <tr>
                <td>${itemName}</td>
                <td><span class="badge ${itemType === 'Model' ? 'bg-info' : 'bg-primary'}">${itemType}</span></td>
                <td><span class="badge ${statusBadge}">${job.status}</span></td>
                <td>${formatFileSize(job.file_size)}</td>
                <td>${formatDate(job.created_at)}</td>
                <td>
                    ${job.status === 'completed' ? `
                        <button class="btn btn-sm btn-success" onclick="window.currentPageInstance.downloadExport(${job.id})">
                            <i class="bi bi-download"></i> Download
                        </button>
                    ` : job.status === 'processing' ? `
                        <button class="btn btn-sm btn-warning" disabled>
                            <span class="spinner-border spinner-border-sm me-1"></span>Processing
                        </button>
                    ` : job.status === 'failed' ? `
                        <button class="btn btn-sm btn-danger" disabled>Failed</button>
                    ` : `
                        <button class="btn btn-sm btn-secondary" disabled>Pending</button>
                    `}
                    <button class="btn btn-sm btn-outline-danger" onclick="window.currentPageInstance.deleteExport(${job.id})">
                        <i class="bi bi-trash"></i>
                    </button>
                </td>
            </tr>
        `;
    }

    attachEventListeners() {
        const form = document.getElementById('export-form');
        if (form) {
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                await this.handleExportSubmit();
            });
        }

        // Handle export type change - use both change and click events for Bootstrap btn-check
        const typeDataset = document.getElementById('type-dataset');
        const typeModel = document.getElementById('type-model');
        const labelDataset = document.querySelector('label[for="type-dataset"]');
        const labelModel = document.querySelector('label[for="type-model"]');
        
        const handleDatasetSelect = () => {
            if (typeDataset && typeDataset.checked) {
                this.updateExportTypeUI('dataset');
            }
        };
        
        const handleModelSelect = () => {
            if (typeModel && typeModel.checked) {
                this.updateExportTypeUI('model');
            }
        };
        
        if (typeDataset) {
            typeDataset.addEventListener('change', handleDatasetSelect);
            typeDataset.addEventListener('click', handleDatasetSelect);
        }
        
        if (labelDataset) {
            labelDataset.addEventListener('click', () => {
                setTimeout(handleDatasetSelect, 0);
            });
        }
        
        if (typeModel) {
            typeModel.addEventListener('change', handleModelSelect);
            typeModel.addEventListener('click', handleModelSelect);
        }
        
        if (labelModel) {
            labelModel.addEventListener('click', () => {
                setTimeout(handleModelSelect, 0);
            });
        }

        // Delete export confirmation modal
        const confirmDeleteBtn = document.getElementById('confirmDeleteExportBtn');
        if (confirmDeleteBtn) {
            confirmDeleteBtn.addEventListener('click', () => {
                this.confirmDeleteExport();
            });
        }
    }

    updateExportTypeUI(type) {
        this.exportType = type;
        
        const datasetSelection = document.getElementById('dataset-selection');
        const modelSelection = document.getElementById('model-selection');
        const includeImagesOption = document.getElementById('include-images-option');
        
        if (type === 'dataset') {
            datasetSelection.style.display = 'block';
            modelSelection.style.display = 'none';
            includeImagesOption.style.display = 'block';
        } else {
            datasetSelection.style.display = 'none';
            modelSelection.style.display = 'block';
            includeImagesOption.style.display = 'none';
        }
    }

    async handleExportSubmit() {
        const exportType = document.querySelector('input[name="export-type"]:checked').value;
        
        try {
            let exportData = {};
            
            if (exportType === 'dataset') {
                const datasetId = parseInt(document.getElementById('export-dataset').value);
                const includeImages = document.getElementById('include-images').checked;
                
                if (!datasetId) {
                    showToast('Please select a dataset', 'error');
                    return;
                }
                
                exportData = {
                    dataset_id: datasetId,
                    include_images: includeImages
                };
            } else {
                const modelId = parseInt(document.getElementById('export-model').value);
                
                if (!modelId) {
                    showToast('Please select a model', 'error');
                    return;
                }
                
                exportData = {
                    model_id: modelId,
                    include_images: false // Not relevant for models
                };
            }

            const result = await apiService.createExport(exportData);
            showToast(`${exportType === 'dataset' ? 'Dataset' : 'Model'} export job created successfully! It will be processed in the background.`, 'success');

            // Reload export jobs
            await this.loadExportJobs();
            const app = document.getElementById('app');
            if (app) {
                app.innerHTML = this.render();
                this.attachEventListeners();
            }

        } catch (error) {
            showToast('Failed to create export: ' + error.message, 'error');
        }
    }

    async downloadExport(exportId) {
        try {
            await apiService.downloadExport(exportId);
            showToast('Download started', 'success');
        } catch (error) {
            showToast('Failed to download: ' + error.message, 'error');
        }
    }

    async deleteExport(exportId) {
        // Store exportId for modal confirmation
        this.pendingDeleteExportId = exportId;
        
        // Show Bootstrap modal
        const modal = new bootstrap.Modal(document.getElementById('deleteExportModal'));
        modal.show();
    }

    async confirmDeleteExport() {
        const exportId = this.pendingDeleteExportId;
        if (!exportId) return;

        try {
            await apiService.deleteExport(exportId);
            showToast('Export deleted successfully', 'success');

            // Close modal
            const modalElement = document.getElementById('deleteExportModal');
            const modal = bootstrap.Modal.getInstance(modalElement);
            if (modal) {
                modal.hide();
            }

            // Clear pending delete
            this.pendingDeleteExportId = null;

            // Reload export jobs
            await this.loadExportJobs();
            const app = document.getElementById('app');
            if (app) {
                app.innerHTML = this.render();
                this.attachEventListeners();
            }
        } catch (error) {
            showToast('Failed to delete export: ' + error.message, 'error');
        }
    }

    startAutoRefresh() {
        // Refresh every 5 seconds if there are pending/processing jobs
        this.refreshInterval = setInterval(async () => {
            const hasActiveJobs = this.exportJobs.some(j => j.status === 'pending' || j.status === 'processing');
            if (hasActiveJobs) {
                await this.loadExportJobs();
                const app = document.getElementById('app');
                if (app && window.location.hash === '#/export') {
                    app.innerHTML = this.render();
                    this.attachEventListeners();
                }
            }
        }, 5000);
    }

    destroy() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
    }

    cleanup() {
        // Alias for destroy() to match app.js expectations
        this.destroy();
    }
}

console.log('[ExportPage] Module loaded');
