// Export Page Component

class ExportPage {
    constructor() {
        console.log('[ExportPage] Initializing...');
        this.datasets = [];
        this.versions = [];
        this.exportJobs = [];
        this.selectedDataset = null;
        this.selectedVersion = null;
        this.isLoading = true;
    }

    async init() {
        console.log('[ExportPage] Starting initialization...');
        try {
            await Promise.all([
                this.loadDatasets(),
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
        } catch (error) {
            console.error('Error loading datasets:', error);
            this.datasets = [];
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
                        <h1 class="display-5 fw-bold mb-2">Export Datasets</h1>
                        <p class="text-muted">Export your datasets in various formats (YOLO, COCO, Pascal VOC, etc.)</p>
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
                                            <!-- Dataset Selection -->
                                            <div class="col-md-6">
                                                <label class="form-label fw-bold">Dataset</label>
                                                <select class="form-select" id="export-dataset" required>
                                                    <option value="">-- Select Dataset --</option>
                                                    ${this.datasets.map(d => `
                                                        <option value="${d.id}">${d.name}</option>
                                                    `).join('')}
                                                </select>
                                            </div>

                                            <!-- Include Images -->
                                            <div class="col-12">
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
                            <div class="card border-0 shadow-sm">
                                <div class="card-header bg-white">
                                    <h6 class="mb-0 fw-bold">Export Information</h6>
                                </div>
                                <div class="card-body">
                                    <h6 class="fw-bold mb-3">Supported Formats</h6>
                                    <ul class="small">
                                        <li><strong>YOLOv12/v11/v8</strong>: Standard YOLO format with data.yaml</li>
                                    </ul>

                                    <div class="alert alert-info small mb-0 mt-3">
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
                            <th>ID</th>
                            <th>Dataset</th>
                            <th>Format</th>
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

        return `
            <tr>
                <td>${job.id}</td>
                <td>Dataset #${job.dataset_id || job.version_id}</td>
                <td><span class="badge bg-primary">YOLO</span></td>
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
    }

    async handleExportSubmit() {
        const datasetId = parseInt(document.getElementById('export-dataset').value);
        const includeImages = document.getElementById('include-images').checked;

        if (!datasetId) {
            showToast('Please select dataset', 'error');
            return;
        }

        try {
            const exportData = {
                dataset_id: datasetId,
                include_images: includeImages
            };

            const result = await apiService.createExport(exportData);
            showToast('Export job created successfully! It will be processed in the background.', 'success');

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
        if (!confirm('Are you sure you want to delete this export?')) return;

        try {
            await apiService.deleteExport(exportId);
            showToast('Export deleted successfully', 'success');

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
}

console.log('[ExportPage] Module loaded');
