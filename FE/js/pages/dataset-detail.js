// Dataset Detail Page Component

class DatasetDetailPage {
    constructor(datasetId) {
        this.datasetId = datasetId;
        this.dataset = null;
        this.images = [];
        this.currentPage = 1;
        this.imagesPerPage = 12;
        this.selectedImage = null;
        this.filterStatus = 'all'; // all, annotated, unannotated
    }

    async init() {
        await this.loadDataset();
        await this.loadImages();
        this.attachEventListeners();
    }

    async loadDataset() {
        try {
            this.dataset = await apiService.getDataset(this.datasetId);
        } catch (error) {
            console.error('Error loading dataset:', error);
        }
    }

    async loadImages() {
        try {
            console.log('Loading images for dataset:', this.datasetId);
            const images = await apiService.getDatasetImages(this.datasetId);
            console.log('Received images:', images);

            // Add base URL for image paths
            this.images = images.map(img => {
                // file_path format: "datasets/1/image.jpg"
                const imageUrl = `http://localhost:8000/uploads/${img.file_path}`;
                console.log('Image URL:', imageUrl);
                return {
                    ...img,
                    file_url: imageUrl
                };
            });
            console.log('Processed images with URLs:', this.images);
        } catch (error) {
            console.error('Error loading images:', error);
            this.images = [];
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

        const classNames = Array.isArray(this.dataset.class_names)
            ? this.dataset.class_names
            : (typeof this.dataset.class_names === 'string'
                ? JSON.parse(this.dataset.class_names)
                : []);

        return `
            <div class="min-vh-100 bg-light">
                <div class="container-fluid py-4">
                    <!-- Page Header -->
                    <div class="mb-4">
                        <nav aria-label="breadcrumb">
                            <ol class="breadcrumb">
                                <li class="breadcrumb-item"><a href="#/datasets">Datasets</a></li>
                                <li class="breadcrumb-item active">${this.dataset.name}</li>
                            </ol>
                        </nav>
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <h1 class="display-5 fw-bold mb-2">${this.dataset.name}</h1>
                                <p class="text-muted">${this.dataset.description || 'No description'}</p>
                            </div>
                            <div class="d-flex gap-2">
                                <button class="btn btn-outline-primary" onclick="window.location.hash='#/auto-annotate/${this.datasetId}'">
                                    <i class="bi bi-sparkles me-1"></i> Auto-Annotate
                                </button>
                                <button class="btn btn-outline-secondary" onclick="showExportModal(${this.datasetId})">
                                    <i class="bi bi-download me-1"></i> Export
                                </button>
                                <button class="btn btn-danger" onclick="confirmDeleteDataset(${this.datasetId})">
                                    <i class="bi bi-trash me-1"></i> Delete
                                </button>
                            </div>
                        </div>
                    </div>

                    <div class="row g-4">
                        <!-- Left Sidebar: Dataset Info and Filters -->
                        <div class="col-lg-3">
                            <!-- Dataset Statistics -->
                            <div class="card border-0 shadow-sm mb-3">
                                <div class="card-header bg-white border-0">
                                    <h6 class="mb-0 fw-bold">Dataset Statistics</h6>
                                </div>
                                <div class="card-body">
                                    <div class="d-flex flex-column gap-3">
                                        <div>
                                            <small class="text-muted">Total Images</small>
                                            <h4 class="mb-0">${this.dataset.total_images || 0}</h4>
                                        </div>
                                        <div>
                                            <small class="text-muted">Annotated</small>
                                            <h4 class="mb-0 text-success">${this.dataset.annotated_images || 0}</h4>
                                            <div class="progress mt-2" style="height: 6px;">
                                                <div class="progress-bar bg-success" style="width: ${this.calculateAnnotationProgress()}%"></div>
                                            </div>
                                        </div>
                                        <div>
                                            <small class="text-muted">Unannotated</small>
                                            <h4 class="mb-0 text-warning">
                                                ${(this.dataset.total_images || 0) - (this.dataset.annotated_images || 0)}
                                            </h4>
                                        </div>
                                        <div>
                                            <small class="text-muted">Total Classes</small>
                                            <h4 class="mb-0">${this.dataset.total_classes || 0}</h4>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Class Labels -->
                            <div class="card border-0 shadow-sm mb-3">
                                <div class="card-header bg-white border-0">
                                    <h6 class="mb-0 fw-bold">Class Labels</h6>
                                </div>
                                <div class="card-body">
                                    ${classNames.length > 0 ? `
                                        <div class="d-flex flex-wrap gap-2">
                                            ${classNames.map((name, idx) => `
                                                <span class="badge bg-primary">${idx}: ${name}</span>
                                            `).join('')}
                                        </div>
                                    ` : `
                                        <p class="text-muted small mb-0">No classes defined</p>
                                    `}
                                </div>
                            </div>

                            <!-- Filters -->
                            <div class="card border-0 shadow-sm">
                                <div class="card-header bg-white border-0">
                                    <h6 class="mb-0 fw-bold">Filters</h6>
                                </div>
                                <div class="card-body">
                                    <div class="mb-3">
                                        <label class="form-label small">Status</label>
                                        <select class="form-select form-select-sm" id="filter-status">
                                            <option value="all">All Images</option>
                                            <option value="annotated">Annotated Only</option>
                                            <option value="unannotated">Unannotated Only</option>
                                        </select>
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label small">Class</label>
                                        <select class="form-select form-select-sm" id="filter-class">
                                            <option value="all">All Classes</option>
                                            ${classNames.map((name, idx) => `
                                                <option value="${idx}">${name}</option>
                                            `).join('')}
                                        </select>
                                    </div>
                                    <button class="btn btn-outline-secondary btn-sm w-100" id="reset-filters">
                                        <i class="bi bi-arrow-clockwise me-1"></i> Reset Filters
                                    </button>
                                </div>
                            </div>
                        </div>

                        <!-- Main Content: Image Gallery -->
                        <div class="col-lg-9">
                            <div class="card border-0 shadow-sm">
                                <div class="card-header bg-white border-0">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <h5 class="mb-0 fw-bold">
                                            <i class="bi bi-images me-2"></i>Images
                                        </h5>
                                        <div class="d-flex gap-2 align-items-center">
                                            <div class="input-group input-group-sm" style="width: 250px;">
                                                <span class="input-group-text">
                                                    <i class="bi bi-search"></i>
                                                </span>
                                                <input type="text" class="form-control" id="image-search"
                                                       placeholder="Search images...">
                                            </div>
                                            <div class="btn-group btn-group-sm" role="group">
                                                <button type="button" class="btn btn-outline-secondary active" id="grid-view">
                                                    <i class="bi bi-grid-3x3"></i>
                                                </button>
                                                <button type="button" class="btn btn-outline-secondary" id="list-view">
                                                    <i class="bi bi-list"></i>
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="card-body">
                                    ${this.images.length > 0 ? this.renderImageGallery() : this.renderEmptyState()}
                                </div>

                                ${this.images.length > 0 ? `
                                    <div class="card-footer bg-white">
                                        ${this.renderPagination()}
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Image Viewer Modal -->
                ${this.renderImageViewerModal()}
            </div>
        `;
    }

    renderImageGallery() {
        const startIdx = (this.currentPage - 1) * this.imagesPerPage;
        const endIdx = startIdx + this.imagesPerPage;
        const pageImages = this.images.slice(startIdx, endIdx);

        return `
            <div class="row g-3" id="image-gallery">
                ${pageImages.map(image => `
                    <div class="col-md-4 col-lg-3">
                        <div class="card h-100 hover-shadow" style="cursor: pointer;"
                             onclick="viewImage(${image.id})">
                            <img src="${image.file_url}" class="card-img-top" alt="${image.filename}"
                                 style="height: 200px; object-fit: cover;"
                                 onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iI2VlZSIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LXNpemU9IjE4IiBmaWxsPSIjOTk5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+SW1hZ2UgTm90IEZvdW5kPC90ZXh0Pjwvc3ZnPg==';">
                            <div class="card-body p-2">
                                <p class="small mb-1 text-truncate" title="${image.filename}">${image.filename}</p>
                                <div class="d-flex justify-content-between align-items-center">
                                    <small class="text-muted">${image.width || '?'}x${image.height || '?'}</small>
                                    ${image.is_annotated ?
                                        '<span class="badge bg-success badge-sm">Annotated</span>' :
                                        '<span class="badge bg-warning badge-sm">Pending</span>'
                                    }
                                </div>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    renderEmptyState() {
        return `
            <div class="text-center py-5">
                <i class="bi bi-images text-muted mb-3" style="font-size: 5rem;"></i>
                <h5 class="text-muted">No Images Found</h5>
                <p class="text-muted">Upload images to this dataset to get started</p>
                <button class="btn btn-primary" onclick="showUploadDatasetModal()">
                    <i class="bi bi-upload me-1"></i> Upload Images
                </button>
            </div>
        `;
    }

    renderPagination() {
        const totalPages = Math.ceil(this.images.length / this.imagesPerPage);

        if (totalPages <= 1) return '';

        let paginationHTML = `
            <nav>
                <ul class="pagination pagination-sm mb-0 justify-content-center">
                    <li class="page-item ${this.currentPage === 1 ? 'disabled' : ''}">
                        <a class="page-link" href="#" onclick="changePage(${this.currentPage - 1}); return false;">
                            Previous
                        </a>
                    </li>
        `;

        for (let i = 1; i <= totalPages; i++) {
            if (i === 1 || i === totalPages || (i >= this.currentPage - 2 && i <= this.currentPage + 2)) {
                paginationHTML += `
                    <li class="page-item ${i === this.currentPage ? 'active' : ''}">
                        <a class="page-link" href="#" onclick="changePage(${i}); return false;">${i}</a>
                    </li>
                `;
            } else if (i === this.currentPage - 3 || i === this.currentPage + 3) {
                paginationHTML += `<li class="page-item disabled"><span class="page-link">...</span></li>`;
            }
        }

        paginationHTML += `
                    <li class="page-item ${this.currentPage === totalPages ? 'disabled' : ''}">
                        <a class="page-link" href="#" onclick="changePage(${this.currentPage + 1}); return false;">
                            Next
                        </a>
                    </li>
                </ul>
            </nav>
        `;

        return paginationHTML;
    }

    renderImageViewerModal() {
        return `
            <div class="modal fade" id="imageViewerModal" tabindex="-1" aria-hidden="true">
                <div class="modal-dialog modal-xl">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Image Viewer & Annotation Editor</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="row">
                                <div class="col-lg-8">
                                    <canvas id="annotation-canvas" class="border w-100"></canvas>
                                </div>
                                <div class="col-lg-4">
                                    <h6 class="fw-bold mb-3">Annotations</h6>
                                    <div id="annotation-list" class="list-group">
                                        <div class="text-muted text-center py-3">No annotations</div>
                                    </div>
                                    <div class="mt-3">
                                        <button class="btn btn-sm btn-primary w-100">
                                            <i class="bi bi-plus-circle me-1"></i> Add Annotation
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            <button type="button" class="btn btn-primary">Save Changes</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    calculateAnnotationProgress() {
        if (!this.dataset.total_images || this.dataset.total_images === 0) return 0;
        return Math.round((this.dataset.annotated_images / this.dataset.total_images) * 100);
    }

    attachEventListeners() {
        // Filter status
        const filterStatus = document.getElementById('filter-status');
        if (filterStatus) {
            filterStatus.addEventListener('change', (e) => {
                this.filterStatus = e.target.value;
                this.applyFilters();
            });
        }

        // Reset filters
        const resetFilters = document.getElementById('reset-filters');
        if (resetFilters) {
            resetFilters.addEventListener('click', () => {
                this.filterStatus = 'all';
                document.getElementById('filter-status').value = 'all';
                document.getElementById('filter-class').value = 'all';
                this.applyFilters();
            });
        }
    }

    applyFilters() {
        // Filter logic would go here
        this.loadImages().then(() => {
            this.updateGallery();
        });
    }

    updateGallery() {
        const gallery = document.getElementById('image-gallery');
        if (gallery) {
            gallery.innerHTML = this.renderImageGallery();
        }
    }

    changePage(page) {
        this.currentPage = page;
        this.updateGallery();
    }
}

// Global functions for onclick handlers
function viewImage(imageId) {
    const modal = new bootstrap.Modal(document.getElementById('imageViewerModal'));
    modal.show();
}

function changePage(page) {
    if (window.currentDatasetDetailPage) {
        window.currentDatasetDetailPage.changePage(page);
    }
}

function showExportModal(datasetId) {
    const modalHTML = `
        <div class="modal fade" id="exportDatasetModal" tabindex="-1" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="bi bi-download me-2"></i>Export Dataset
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="mb-3">
                            <label class="form-label">Export Format</label>
                            <select class="form-select" id="export-format">
                                <option value="yolo">YOLO (txt files)</option>
                                <option value="coco">COCO JSON</option>
                                <option value="pascal-voc">Pascal VOC (XML)</option>
                                <option value="tfrecord">TFRecord</option>
                            </select>
                        </div>
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="include-images" checked>
                            <label class="form-check-label" for="include-images">
                                Include images in export
                            </label>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" onclick="handleExport(${datasetId})">
                            <i class="bi bi-download me-1"></i> Export
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    const existingModal = document.getElementById('exportDatasetModal');
    if (existingModal) existingModal.remove();

    document.body.insertAdjacentHTML('beforeend', modalHTML);
    const modal = new bootstrap.Modal(document.getElementById('exportDatasetModal'));
    modal.show();
}

function handleExport(datasetId) {
    const format = document.getElementById('export-format').value;
    const includeImages = document.getElementById('include-images').checked;

    showToast('Preparing export...', 'info');

    // In real app, this would call the API
    setTimeout(() => {
        showToast('Dataset exported successfully!', 'success');
        bootstrap.Modal.getInstance(document.getElementById('exportDatasetModal')).hide();
    }, 2000);
}

function confirmDeleteDataset(datasetId) {
    if (confirm('Are you sure you want to delete this dataset? This action cannot be undone.')) {
        apiService.delete(`/datasets/${datasetId}`)
            .then(() => {
                window.location.hash = '#/datasets';
            })
            .catch(error => {
                console.error('Failed to delete dataset:', error);
            });
    }
}
