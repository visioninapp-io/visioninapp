// Dataset Detail Page Component

class DatasetDetailPage {
    constructor(datasetId) {
        this.datasetId = datasetId;
        this.dataset = null;
        this.images = [];
        this.labelClasses = []; // Store label classes
        this.currentPage = 1;
        this.imagesPerPage = 12;
        this.selectedImage = null;
        this.filterStatus = 'all'; // all, annotated, unannotated
        this.filterClass = 'all'; // all, or specific class name
        this.minConfidence = 0; // 0-100, for annotation display threshold
    }

    async init() {
        await this.loadDataset();
        await this.loadLabelClasses();
        await this.loadImages();

        // Re-render after data is loaded
        this.updateUI();
        this.attachEventListeners();
    }

    updateUI() {
        // Re-render the page
        const app = document.getElementById('app');
        if (app) {
            app.innerHTML = this.render();
        }
    }

    async loadDataset() {
        try {
            this.dataset = await apiService.getDataset(this.datasetId);
        } catch (error) {
            console.error('Error loading dataset:', error);
        }
    }

    async loadLabelClasses() {
        try {
            console.log('Loading label classes for dataset:', this.datasetId);
            this.labelClasses = await apiService.getDatasetLabelClasses(this.datasetId);
            console.log('Loaded label classes:', this.labelClasses);
        } catch (error) {
            console.error('Error loading label classes:', error);
            this.labelClasses = [];
        }
    }

    async loadImages() {
        try {
            console.log('Loading images for dataset:', this.datasetId);
            const images = await apiService.getDatasetImages(this.datasetId);
            console.log('Received images:', images);

            // Store images without file_url (will be loaded on demand)
            this.images = images.map(img => ({
                ...img,
                file_url: null // Will be loaded via presigned URL
            }));
            console.log('Processed images:', this.images);
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

        // Use labelClasses loaded from API
        const classNames = this.labelClasses.map(lc => lc.display_name);

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
                                    <h6 class="mb-0 fw-bold">Class Labels (${this.labelClasses.length})</h6>
                                </div>
                                <div class="card-body">
                                    ${this.labelClasses.length > 0 ? `
                                        <div class="d-flex flex-wrap gap-2">
                                            ${this.labelClasses.map((labelClass) => `
                                                <span class="badge" style="background-color: ${labelClass.color}">
                                                    ${labelClass.display_name}
                                                </span>
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
                                            ${classNames.map(name => `
                                                <option value="${name}">${name}</option>
                                            `).join('')}
                                        </select>
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label small">Min Confidence: <span id="confidence-value">0%</span></label>
                                        <input type="range" class="form-range" id="filter-confidence"
                                               min="0" max="100" value="0" step="5">
                                        <div class="d-flex justify-content-between small text-muted">
                                            <span>0%</span>
                                            <span>100%</span>
                                        </div>
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
        // Use filtered images if available, otherwise use all images
        const displayImages = this.filteredImages || this.images;

        const startIdx = (this.currentPage - 1) * this.imagesPerPage;
        const endIdx = startIdx + this.imagesPerPage;
        const pageImages = displayImages.slice(startIdx, endIdx);

        if (displayImages.length === 0) {
            return `
                <div class="text-center py-5">
                    <i class="bi bi-filter text-muted" style="font-size: 3rem;"></i>
                    <h5 class="mt-3 text-muted">No images match the current filters</h5>
                    <p class="text-muted">Try adjusting your filter settings</p>
                </div>
            `;
        }

        // Schedule loading of presigned URLs after render
        setTimeout(() => this.loadImageUrlsForGallery(pageImages), 0);

        return `
            <div class="row g-3" id="image-gallery">
                ${pageImages.map(image => `
                    <div class="col-md-4 col-lg-3">
                        <div class="card h-100 hover-shadow" style="cursor: pointer;" onclick="viewImage(${image.id})">
                            <img id="gallery-img-${image.id}"
                                 src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iI2YwZjBmMCIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LXNpemU9IjE0IiBmaWxsPSIjOTk5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+TG9hZGluZy4uLjwvdGV4dD48L3N2Zz4="
                                 class="card-img-top" alt="${image.filename}"
                                 style="height: 200px; object-fit: cover;"
                                 data-asset-id="${image.id}">
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
        // Use filtered images if available, otherwise use all images
        const displayImages = this.filteredImages || this.images;
        const totalPages = Math.ceil(displayImages.length / this.imagesPerPage);

        if (totalPages <= 1) return '';

        let paginationHTML = `
            <nav>
                <ul class="pagination pagination-sm mb-0 justify-content-center">
                    <li class="page-item ${this.currentPage === 1 ? 'disabled' : ''}">
        `;

        // Previous button - only add onclick if not disabled
        if (this.currentPage === 1) {
            paginationHTML += `<span class="page-link">Previous</span>`;
        } else {
            paginationHTML += `<a class="page-link" href="#" onclick="changePage(${this.currentPage - 1}); return false;">Previous</a>`;
        }

        paginationHTML += `</li>`;

        // Page numbers
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

        // Next button - only add onclick if not disabled
        paginationHTML += `<li class="page-item ${this.currentPage === totalPages ? 'disabled' : ''}">`;
        if (this.currentPage === totalPages) {
            paginationHTML += `<span class="page-link">Next</span>`;
        } else {
            paginationHTML += `<a class="page-link" href="#" onclick="changePage(${this.currentPage + 1}); return false;">Next</a>`;
        }

        paginationHTML += `
                    </li>
                </ul>
            </nav>
        `;

        return paginationHTML;
    }

    renderImageViewerModal() {
        return `
            <div class="modal fade" id="imageViewerModal" tabindex="-1" aria-hidden="true" data-bs-backdrop="static">
                <div class="modal-dialog modal-xl">
                    <div class="modal-content">
                        <div class="modal-header bg-primary text-white">
                            <h5 class="modal-title">
                                <i class="bi bi-image me-2"></i>Image Viewer & Annotation Editor
                            </h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="alert alert-info d-flex align-items-center mb-3" role="alert">
                                <i class="bi bi-info-circle-fill me-2"></i>
                                <div>
                                    <strong>Draw mode:</strong> Click and drag to create bounding boxes.
                                    Click on annotations in the list to delete them.
                                </div>
                            </div>
                            <div class="row">
                                <div class="col-lg-8">
                                    <div class="border rounded p-2" style="background-color: #f8f9fa; max-height: 600px; overflow: auto;">
                                        <canvas id="annotation-canvas" style="max-width: 100%; display: block; cursor: crosshair;"></canvas>
                                    </div>
                                </div>
                                <div class="col-lg-4">
                                    <div class="d-flex justify-content-between align-items-center mb-3">
                                        <h6 class="fw-bold mb-0">
                                            <i class="bi bi-tags me-2"></i>Annotations (<span id="viewer-annotation-count">0</span>)
                                        </h6>
                                        <button class="btn btn-sm btn-outline-danger" onclick="clearAllViewerAnnotations()">
                                            <i class="bi bi-trash"></i> Clear All
                                        </button>
                                    </div>
                                    <div id="annotation-list" class="list-group" style="max-height: 500px; overflow-y: auto;">
                                        <div class="text-muted text-center py-3">Loading...</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            <button type="button" class="btn btn-success" onclick="saveViewerAnnotations()">
                                <i class="bi bi-check-circle me-1"></i>Save All Changes
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Class Input Modal for Viewer -->
            <div class="modal fade" id="viewerClassInputModal" tabindex="-1" data-bs-backdrop="true">
                <div class="modal-dialog modal-dialog-centered">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Enter Class Name</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <input type="text"
                                   id="viewer-class-name-input"
                                   class="form-control form-control-lg"
                                   placeholder="e.g., person, car, dog"
                                   autocomplete="off">
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" id="viewer-confirm-class-btn">Add Annotation</button>
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
                console.log('[Filter] Status changed to:', this.filterStatus);
                this.applyFilters();
            });
        }

        // Filter class
        const filterClass = document.getElementById('filter-class');
        if (filterClass) {
            filterClass.addEventListener('change', (e) => {
                this.filterClass = e.target.value;
                console.log('[Filter] Class changed to:', this.filterClass);
                this.applyFilters();
            });
        }

        // Filter confidence threshold
        const filterConfidence = document.getElementById('filter-confidence');
        const confidenceValue = document.getElementById('confidence-value');
        if (filterConfidence && confidenceValue) {
            filterConfidence.addEventListener('input', (e) => {
                this.minConfidence = parseInt(e.target.value);
                confidenceValue.textContent = `${this.minConfidence}%`;
                console.log('[Filter] Min confidence changed to:', this.minConfidence);
                // Don't apply filters on every slider movement, wait for change event
            });

            filterConfidence.addEventListener('change', (e) => {
                this.minConfidence = parseInt(e.target.value);
                console.log('[Filter] Min confidence set to:', this.minConfidence);
                this.applyFilters();
            });
        }

        // Reset filters
        const resetFilters = document.getElementById('reset-filters');
        if (resetFilters) {
            resetFilters.addEventListener('click', () => {
                this.filterStatus = 'all';
                this.filterClass = 'all';
                this.minConfidence = 0;
                document.getElementById('filter-status').value = 'all';
                document.getElementById('filter-class').value = 'all';
                document.getElementById('filter-confidence').value = '0';
                document.getElementById('confidence-value').textContent = '0%';
                console.log('[Filter] Reset to all');
                this.applyFilters();
            });
        }
    }

    async applyFilters() {
        console.log('[Filter] Applying filters - Status:', this.filterStatus, 'Class:', this.filterClass);

        // Get all images (reload from backend to ensure we have latest data)
        await this.loadImages();

        // Apply filters to the loaded images
        let filteredImages = [...this.images];

        // Filter by annotation status
        if (this.filterStatus === 'annotated') {
            filteredImages = filteredImages.filter(img => img.is_annotated === true);
            console.log('[Filter] After annotated filter:', filteredImages.length, 'images');
        } else if (this.filterStatus === 'unannotated') {
            filteredImages = filteredImages.filter(img => img.is_annotated === false);
            console.log('[Filter] After unannotated filter:', filteredImages.length, 'images');
        }

        // Filter by class (need to check annotations for each image)
        if (this.filterClass !== 'all') {
            const imagesWithClass = [];
            const minConf = this.minConfidence > 0 ? this.minConfidence / 100 : null;
            for (const img of filteredImages) {
                if (img.is_annotated) {
                    // Get annotations for this image (filtered by confidence if set)
                    const annotations = await apiService.getImageAnnotations(img.id, minConf);
                    // Check if any annotation has the selected class
                    const hasClass = annotations.some(ann => {
                        const className = ann.label_class ? ann.label_class.display_name : null;
                        return className === this.filterClass;
                    });
                    if (hasClass) {
                        imagesWithClass.push(img);
                    }
                }
            }
            filteredImages = imagesWithClass;
            console.log('[Filter] After class filter:', filteredImages.length, 'images');
        }

        // Store filtered images temporarily for display
        this.filteredImages = filteredImages;
        console.log('[Filter] Final filtered count:', filteredImages.length);

        // Reset to first page when filters change
        this.currentPage = 1;

        // Update gallery display
        this.updateGallery();
    }

    async loadImageUrlsForGallery(images) {
        if (!images || images.length === 0) return;

        console.log(`[DatasetDetail] Loading presigned URLs for ${images.length} images...`);

        // Load URLs in parallel
        const urlPromises = images.map(async (image) => {
            try {
                const response = await apiService.get(`/datasets/assets/${image.id}/presigned-download`);

                if (response && response.download_url) {
                    const imgElement = document.getElementById(`gallery-img-${image.id}`);
                    if (imgElement) {
                        imgElement.src = response.download_url;
                        // Store URL in image object for later use (e.g., in modal)
                        image.file_url = response.download_url;
                    }
                }
            } catch (error) {
                console.error(`[DatasetDetail] Failed to load URL for image ${image.id}:`, error);
                const imgElement = document.getElementById(`gallery-img-${image.id}`);
                if (imgElement) {
                    imgElement.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iI2VlZSIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LXNpemU9IjE0IiBmaWxsPSIjZGQwMDAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+TG9hZCBGYWlsZWQ8L3RleHQ+PC9zdmc+';
                }
            }
        });

        await Promise.all(urlPromises);
        console.log('[DatasetDetail] All gallery image URLs loaded');
    }

    updateGallery() {
        const gallery = document.getElementById('image-gallery');
        if (gallery) {
            gallery.innerHTML = this.renderImageGallery();
        }
    }

    changePage(page) {
        console.log('[DatasetDetail] Changing to page:', page);

        // Calculate total pages
        const displayImages = this.filteredImages || this.images;
        const totalPages = Math.ceil(displayImages.length / this.imagesPerPage);

        // Validate page number
        if (page < 1 || page > totalPages) {
            console.warn('[DatasetDetail] Invalid page number:', page);
            return;
        }

        this.currentPage = page;

        // Re-render the entire page to update both gallery and pagination
        const app = document.getElementById('app');
        if (app) {
            app.innerHTML = this.render();
            this.attachEventListeners();
        }
    }
}

// Global state for Image Viewer annotation editing
let viewerState = {
    currentImage: null,
    annotations: [],
    deletedAnnotations: [], // Track deleted annotations for backend deletion
    canvas: null,
    ctx: null,
    img: null,
    scale: 1,
    originalWidth: 0,
    originalHeight: 0,
    displayWidth: 0,
    displayHeight: 0,
    drawing: false,
    startX: 0,
    startY: 0,
    currentX: 0,
    currentY: 0
};

// Global functions for onclick handlers
async function viewImage(imageId) {
    console.log('[ViewImage] Loading image:', imageId);

    // Find image data
    const page = window.currentDatasetDetailPage;
    if (!page) {
        console.error('[ViewImage] No page instance found');
        return;
    }

    const image = page.images.find(img => img.id === imageId);
    if (!image) {
        console.error('[ViewImage] Image not found:', imageId);
        return;
    }

    console.log('[ViewImage] Image data:', image);
    viewerState.currentImage = image;
    viewerState.deletedAnnotations = []; // Reset deleted annotations

    // Load annotations (filtered by min confidence if set)
    const minConf = page.minConfidence > 0 ? page.minConfidence / 100 : null;
    console.log('[ViewImage] Using min confidence:', minConf);
    const annotations = await apiService.getImageAnnotations(imageId, minConf);
    console.log('[ViewImage] Loaded annotations:', annotations);

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('imageViewerModal'));
    modal.show();

    // Wait for modal to be shown, then setup canvas and draw
    setTimeout(() => {
        setupViewerCanvas();
        drawImageWithAnnotations(image, annotations);
    }, 100);
}

// Setup canvas event listeners for drawing
function setupViewerCanvas() {
    const canvas = document.getElementById('annotation-canvas');
    if (!canvas) {
        console.error('[SetupCanvas] Canvas not found');
        return;
    }

    viewerState.canvas = canvas;
    viewerState.ctx = canvas.getContext('2d');

    // Remove existing listeners
    canvas.onmousedown = null;
    canvas.onmousemove = null;
    canvas.onmouseup = null;

    // Add new listeners
    canvas.addEventListener('mousedown', handleViewerMouseDown);
    canvas.addEventListener('mousemove', handleViewerMouseMove);
    canvas.addEventListener('mouseup', handleViewerMouseUp);

    console.log('[SetupCanvas] Canvas event listeners attached');
}

async function drawImageWithAnnotations(imageData, annotations) {
    console.log('[DrawImage] Drawing image with annotations');
    const canvas = document.getElementById('annotation-canvas');
    if (!canvas) {
        console.error('[DrawImage] Canvas not found');
        return;
    }

    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = function() {
        console.log('[DrawImage] Image loaded:', img.width, 'x', img.height);

        // Store image and dimensions in viewerState
        viewerState.img = img;
        viewerState.originalWidth = img.width;
        viewerState.originalHeight = img.height;

        // Calculate scale to fit canvas in modal
        const maxWidth = 800;
        const maxHeight = 600;
        let scale = 1;

        if (img.width > maxWidth || img.height > maxHeight) {
            const scaleX = maxWidth / img.width;
            const scaleY = maxHeight / img.height;
            scale = Math.min(scaleX, scaleY);
        }

        viewerState.scale = scale;
        viewerState.displayWidth = img.width * scale;
        viewerState.displayHeight = img.height * scale;

        // Set canvas size
        canvas.width = viewerState.displayWidth;
        canvas.height = viewerState.displayHeight;

        // Convert annotations from normalized coordinates to display coordinates
        viewerState.annotations = annotations.map(ann => {
            if (!ann.geometry || !ann.geometry.bbox) {
                return null;
            }

            const bbox = ann.geometry.bbox;

            // Convert to original pixel coordinates
            const origX = bbox.x_center * img.width - (bbox.width * img.width / 2);
            const origY = bbox.y_center * img.height - (bbox.height * img.height / 2);
            const origWidth = bbox.width * img.width;
            const origHeight = bbox.height * img.height;

            // Convert to display coordinates
            return {
                id: ann.id,
                className: ann.label_class ? ann.label_class.display_name : 'Unknown',
                label_class_id: ann.label_class_id,
                x: origX * scale,
                y: origY * scale,
                width: origWidth * scale,
                height: origHeight * scale,
                confidence: ann.confidence || 1.0,
                source: ann.source || 'human',
                saved: true
            };
        }).filter(ann => ann !== null);

        console.log('[DrawImage] Converted annotations:', viewerState.annotations);

        // Draw image and annotations
        redrawViewerCanvas();

        // Update annotation list
        updateViewerAnnotationList();
    };

    img.onerror = function() {
        console.error('[DrawImage] Failed to load image');
        showToast('Failed to load image', 'error');
    };

    // Load image from presigned URL
    try {
        // Check if we already have the URL cached
        if (imageData.file_url) {
            img.src = imageData.file_url;
        } else {
            // Load presigned URL
            const response = await apiService.get(`/datasets/assets/${imageData.id}/presigned-download`);
            if (response && response.download_url) {
                img.src = response.download_url;
                // Cache the URL
                imageData.file_url = response.download_url;
            } else {
                console.error('[DrawImage] No download URL available');
                showToast('No image URL available', 'error');
            }
        }
    } catch (error) {
        console.error('[DrawImage] Failed to get presigned URL:', error);
        showToast('Failed to load image URL', 'error');
    }
}

function getColorForClass(classId) {
    // Try to get color from label classes if available
    const page = window.currentDatasetDetailPage;
    if (page && page.labelClasses) {
        const labelClass = page.labelClasses.find(lc => lc.id === classId);
        if (labelClass && labelClass.color) {
            return labelClass.color;
        }
    }

    // Fallback to default colors
    const colors = [
        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
        '#FF9F40', '#FF6384', '#C9CBCF', '#4BC0C0', '#FF6384'
    ];
    return colors[classId % colors.length];
}

function updateAnnotationList(annotations) {
    const listContainer = document.getElementById('annotation-list');
    if (!listContainer) return;

    if (annotations.length === 0) {
        listContainer.innerHTML = '<div class="text-muted text-center py-3">No annotations</div>';
        return;
    }

    listContainer.innerHTML = annotations.map((ann, idx) => {
        const className = ann.label_class ? ann.label_class.display_name : 'Unknown';
        const isManual = ann.source === 'human';
        return `
        <div class="list-group-item">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <span class="badge" style="background-color: ${getColorForClass(ann.label_class_id)}">${ann.label_class_id}</span>
                    <strong>${className}</strong>
                </div>
                <small class="text-muted">${(ann.confidence * 100).toFixed(1)}%</small>
            </div>
            <small class="text-muted">
                ${isManual ? '<i class="bi bi-pencil"></i> Manual' : '<i class="bi bi-robot"></i> Auto'}
            </small>
        </div>
        `;
    }).join('');
}

function changePage(page) {
    if (window.currentDatasetDetailPage) {
        window.currentDatasetDetailPage.changePage(page);
    }
}

function confirmDeleteDataset(datasetId) {
    const modalHTML = `
        <div class="modal fade" id="deleteDatasetModal" tabindex="-1" aria-hidden="true">
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content border-0 shadow-lg">
                    <div class="modal-header border-0 pb-0">
                        <h5 class="modal-title fw-bold">
                            <i class="bi bi-exclamation-triangle-fill text-warning me-2"></i>Delete Dataset
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body pt-4">
                        <p class="mb-3">Are you sure you want to delete this dataset?</p>
                        <div class="d-flex align-items-start text-muted">
                            <i class="bi bi-info-circle me-2 mt-1"></i>
                            <small>This action cannot be undone. All images and annotations in this dataset will be permanently deleted.</small>
                        </div>
                    </div>
                    <div class="modal-footer border-0 pt-0">
                        <button type="button" class="btn btn-outline-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-danger" id="confirm-delete-btn">
                            <i class="bi bi-trash me-1"></i> Delete
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    const existingModal = document.getElementById('deleteDatasetModal');
    if (existingModal) existingModal.remove();

    document.body.insertAdjacentHTML('beforeend', modalHTML);
    const modal = new bootstrap.Modal(document.getElementById('deleteDatasetModal'));

    document.getElementById('confirm-delete-btn').addEventListener('click', () => {
        modal.hide();
        apiService.delete(`/datasets/${datasetId}`)
            .then(() => {
                showToast('Dataset deleted successfully', 'success');
                // Force page reload to update dataset list
                setTimeout(() => {
                    window.location.hash = '#/datasets';
                    window.location.reload();
                }, 500);
            })
            .catch(error => {
                console.error('Failed to delete dataset:', error);
                showToast('Failed to delete dataset: ' + error.message, 'error');
            });
    });

    modal.show();
}

// Open single image annotation modal
async function openSingleImageAnnotation(imageId) {
    console.log(`[OpenSingleImageAnnotation] Opening annotation for image ${imageId}`);

    const page = window.currentDatasetDetailPage;
    if (!page) {
        console.error('[OpenSingleImageAnnotation] No page instance found');
        showToast('Error: Page not initialized', 'error');
        return;
    }

    const image = page.images.find(img => img.id === imageId);
    if (!image) {
        console.error('[OpenSingleImageAnnotation] Image not found:', imageId);
        showToast('Error: Image not found', 'error');
        return;
    }

    // Call the self-annotation modal with single image
    // We'll modify the self-annotation modal to support single image mode
    await showSingleImageSelfAnnotation(page.datasetId, page.dataset.name, imageId);
}

// ========== Image Viewer Annotation Functions ==========

// Redraw canvas with image and all annotations
function redrawViewerCanvas() {
    if (!viewerState.ctx || !viewerState.img) return;

    const ctx = viewerState.ctx;
    const img = viewerState.img;

    // Clear and draw image
    ctx.clearRect(0, 0, viewerState.canvas.width, viewerState.canvas.height);
    ctx.drawImage(img, 0, 0, viewerState.displayWidth, viewerState.displayHeight);

    // Draw all annotations
    viewerState.annotations.forEach((ann, index) => {
        // Draw bbox
        ctx.strokeStyle = ann.saved ? '#4caf50' : '#ff9800';
        ctx.lineWidth = 3;
        ctx.strokeRect(ann.x, ann.y, ann.width, ann.height);

        // Add subtle fill
        ctx.fillStyle = ann.saved ? 'rgba(76, 175, 80, 0.1)' : 'rgba(255, 152, 0, 0.1)';
        ctx.fillRect(ann.x, ann.y, ann.width, ann.height);

        // Draw label background
        const labelText = ann.className;
        ctx.font = 'bold 14px Arial';
        const textWidth = ctx.measureText(labelText).width;

        ctx.fillStyle = ann.saved ? '#4caf50' : '#ff9800';
        ctx.fillRect(ann.x, ann.y - 22, textWidth + 12, 22);

        // Draw label text
        ctx.fillStyle = '#fff';
        ctx.fillText(labelText, ann.x + 6, ann.y - 6);

        // Draw index number
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(ann.x + ann.width - 25, ann.y, 25, 20);
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 12px Arial';
        ctx.fillText(`#${index + 1}`, ann.x + ann.width - 20, ann.y + 14);
    });
}

// Update annotation list in sidebar
function updateViewerAnnotationList() {
    const listContainer = document.getElementById('annotation-list');
    const countElement = document.getElementById('viewer-annotation-count');

    if (!listContainer || !countElement) return;

    countElement.textContent = viewerState.annotations.length;

    if (viewerState.annotations.length === 0) {
        listContainer.innerHTML = `
            <div class="text-muted text-center py-4">
                <i class="bi bi-inbox display-4"></i>
                <p class="mt-2">No annotations yet.<br>Draw boxes on the image.</p>
            </div>
        `;
        return;
    }

    let html = '<div class="list-group list-group-flush">';

    viewerState.annotations.forEach((ann, index) => {
        html += `
            <div class="list-group-item">
                <div class="d-flex justify-content-between align-items-start">
                    <div class="flex-grow-1">
                        <div class="d-flex align-items-center mb-1">
                            <span class="badge bg-primary me-2">${index + 1}</span>
                            <input type="text" class="form-control form-control-sm"
                                value="${ann.className}"
                                onchange="updateViewerAnnotationClass(${index}, this.value)"
                                placeholder="Class name">
                        </div>
                        <div class="small text-muted">
                            Box: (${Math.round(ann.x)}, ${Math.round(ann.y)})
                            ${Math.round(ann.width)}×${Math.round(ann.height)}px
                        </div>
                        ${ann.saved ? '<span class="badge bg-success badge-sm">Saved</span>' : '<span class="badge bg-warning badge-sm">Unsaved</span>'}
                    </div>
                    <button class="btn btn-sm btn-outline-danger" onclick="deleteViewerAnnotation(${index})">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
        `;
    });

    html += '</div>';
    listContainer.innerHTML = html;
}

// Mouse event handlers
function handleViewerMouseDown(e) {
    const rect = viewerState.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    viewerState.startX = mouseX;
    viewerState.startY = mouseY;
    viewerState.currentX = mouseX;
    viewerState.currentY = mouseY;
    viewerState.drawing = true;

    console.log(`[ViewerDrawing] Start drawing at (${mouseX}, ${mouseY})`);
}

function handleViewerMouseMove(e) {
    if (!viewerState.drawing) return;

    const rect = viewerState.canvas.getBoundingClientRect();
    viewerState.currentX = e.clientX - rect.left;
    viewerState.currentY = e.clientY - rect.top;

    // Redraw image and existing annotations
    redrawViewerCanvas();

    // Draw current box
    const ctx = viewerState.ctx;
    const x = Math.min(viewerState.startX, viewerState.currentX);
    const y = Math.min(viewerState.startY, viewerState.currentY);
    const width = Math.abs(viewerState.currentX - viewerState.startX);
    const height = Math.abs(viewerState.currentY - viewerState.startY);

    // Draw yellow box while drawing
    ctx.strokeStyle = '#ffeb3b';
    ctx.lineWidth = 3;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(x, y, width, height);
    ctx.setLineDash([]);

    // Draw size info
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(x, y - 25, 100, 20);
    ctx.fillStyle = '#fff';
    ctx.font = '12px Arial';
    ctx.fillText(`${Math.round(width)} x ${Math.round(height)}`, x + 5, y - 10);
}

function handleViewerMouseUp(e) {
    if (!viewerState.drawing) return;

    const rect = viewerState.canvas.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;

    viewerState.drawing = false;

    // Calculate bbox
    const x = Math.min(viewerState.startX, endX);
    const y = Math.min(viewerState.startY, endY);
    const width = Math.abs(endX - viewerState.startX);
    const height = Math.abs(endY - viewerState.startY);

    // Ignore very small boxes
    if (width < 10 || height < 10) {
        console.log('[ViewerDrawing] Box too small, ignoring');
        redrawViewerCanvas();
        return;
    }

    console.log(`[ViewerDrawing] Created box: (${x}, ${y}, ${width}, ${height})`);

    // Show class input dialog
    showViewerClassInputDialog(x, y, width, height);
}

// Show class input dialog
function showViewerClassInputDialog(x, y, width, height) {
    console.log('[ViewerClassInput] Opening class input modal');

    const modalElement = document.getElementById('viewerClassInputModal');
    const inputElement = document.getElementById('viewer-class-name-input');
    const confirmBtn = document.getElementById('viewer-confirm-class-btn');

    if (!modalElement || !inputElement || !confirmBtn) {
        console.error('[ViewerClassInput] Modal elements not found');
        return;
    }

    // Clear previous input
    inputElement.value = '';

    // Store box coordinates
    window._viewerPendingAnnotation = { x, y, width, height };

    // Get or create modal instance
    let modal = bootstrap.Modal.getInstance(modalElement);
    if (!modal) {
        modal = new bootstrap.Modal(modalElement);
    }

    // Remove old listeners
    const newConfirmBtn = confirmBtn.cloneNode(true);
    confirmBtn.parentNode.replaceChild(newConfirmBtn, confirmBtn);

    const newInputElement = inputElement.cloneNode(true);
    inputElement.parentNode.replaceChild(newInputElement, inputElement);
    newInputElement.value = '';

    // Add confirm button listener
    newConfirmBtn.addEventListener('click', () => {
        const className = newInputElement.value.trim();
        if (className) {
            confirmViewerClassInput(window._viewerPendingAnnotation.x, window._viewerPendingAnnotation.y,
                                    window._viewerPendingAnnotation.width, window._viewerPendingAnnotation.height,
                                    className);
        }
        modal.hide();
    }, { once: true });

    // Handle Enter key
    newInputElement.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            const className = newInputElement.value.trim();
            if (className) {
                confirmViewerClassInput(window._viewerPendingAnnotation.x, window._viewerPendingAnnotation.y,
                                        window._viewerPendingAnnotation.width, window._viewerPendingAnnotation.height,
                                        className);
            }
            modal.hide();
        }
    });

    // Focus input when modal shown
    modalElement.addEventListener('shown.bs.modal', () => {
        setTimeout(() => {
            newInputElement.focus();
        }, 100);
    }, { once: true });

    // Clean up on hide
    modalElement.addEventListener('hidden.bs.modal', () => {
        redrawViewerCanvas();
        window._viewerPendingAnnotation = null;
    }, { once: true });

    modal.show();
}

// Confirm class input and add annotation
function confirmViewerClassInput(x, y, width, height, className) {
    console.log('[ViewerClassInput] Confirming with class name:', className);

    if (!className) {
        console.log('[ViewerAnnotation] No class name entered');
        return;
    }

    // Add annotation
    const annotation = {
        id: null,
        className: className,
        label_class_id: null,
        x,
        y,
        width,
        height,
        confidence: 1.0,
        source: 'human',
        saved: false
    };

    viewerState.annotations.push(annotation);

    // Update display
    redrawViewerCanvas();
    updateViewerAnnotationList();

    console.log(`[ViewerAnnotation] Added annotation:`, annotation);
    showToast(`Added annotation: ${className}`, 'success');
}

// Update annotation class name
function updateViewerAnnotationClass(index, newClassName) {
    if (index < 0 || index >= viewerState.annotations.length) return;

    const annotation = viewerState.annotations[index];
    annotation.className = newClassName.trim().toLowerCase();
    annotation.saved = false;

    console.log(`[ViewerAnnotation] Updated annotation ${index} class to "${newClassName}"`);
    redrawViewerCanvas();
    updateViewerAnnotationList();
}

// Delete annotation
function deleteViewerAnnotation(index) {
    if (index < 0 || index >= viewerState.annotations.length) return;

    if (!confirm('Delete this annotation?')) return;

    const annotation = viewerState.annotations[index];
    console.log(`[ViewerAnnotation] Deleting annotation ${index}:`, annotation);

    // If saved annotation with an ID, track it for backend deletion
    if (annotation.saved && annotation.id) {
        viewerState.deletedAnnotations.push(annotation);
        console.log(`[ViewerAnnotation] Added to deletedAnnotations:`, annotation.id);
    }

    viewerState.annotations.splice(index, 1);
    redrawViewerCanvas();
    updateViewerAnnotationList();
    showToast('Annotation deleted', 'info');
}

// Clear all annotations
function clearAllViewerAnnotations() {
    if (!confirm('Clear all annotations for this image?')) return;

    console.log(`[ViewerAnnotation] Clearing all annotations`);

    // Track saved annotations for deletion
    viewerState.annotations.forEach(ann => {
        if (ann.saved && ann.id) {
            viewerState.deletedAnnotations.push(ann);
        }
    });

    viewerState.annotations = [];
    redrawViewerCanvas();
    updateViewerAnnotationList();
    showToast('All annotations cleared', 'info');
}

// Save all annotations
async function saveViewerAnnotations() {
    console.log('[ViewerAnnotation] Saving all annotations...');

    if (!viewerState.currentImage) {
        showToast('No image loaded', 'error');
        return;
    }

    const unsavedAnnotations = viewerState.annotations.filter(ann => !ann.saved);
    const deletedAnnotations = viewerState.deletedAnnotations;

    if (unsavedAnnotations.length === 0 && deletedAnnotations.length === 0) {
        showToast('No changes to save', 'info');
        return;
    }

    console.log(`[ViewerAnnotation] Changes - New/Modified: ${unsavedAnnotations.length}, Deleted: ${deletedAnnotations.length}`);

    try {
        // Get page to access datasetId
        const page = window.currentDatasetDetailPage;
        if (!page) {
            showToast('Error: Page not initialized', 'error');
            return;
        }

        // First, create or get label classes
        const uniqueClassNames = [...new Set(unsavedAnnotations.map(ann => ann.className.trim().toLowerCase()))];
        const labelClassMap = new Map();

        // Get existing label classes
        const existingClasses = await apiService.get(`/datasets/${page.datasetId}/label-classes`);
        if (existingClasses && Array.isArray(existingClasses)) {
            existingClasses.forEach(cls => {
                labelClassMap.set(cls.display_name.trim().toLowerCase(), cls);
            });
        }

        // Create missing label classes
        for (const className of uniqueClassNames) {
            const key = className.trim().toLowerCase();

            if (!labelClassMap.has(key)) {
                const newClass = await apiService.post(
                    `/datasets/${page.datasetId}/label-classes`,
                    { display_name: className.trim(), color: getRandomColorForLabel() }
                );

                labelClassMap.set(key, newClass);
            }
        }

        let successCount = 0;
        let failCount = 0;

        // Save each annotation
        for (const ann of unsavedAnnotations) {
            try {
                const clsObj = labelClassMap.get(ann.className.trim().toLowerCase());
                const labelClassId = clsObj?.id;
                if (!labelClassId) {
                    console.error(`[ViewerAnnotation] No label class ID for ${ann.className}`);
                    failCount++;
                    continue;
                }

                // Convert display coordinates back to original image coordinates
                const scale = viewerState.scale;
                const originalWidth = viewerState.originalWidth;
                const originalHeight = viewerState.originalHeight;

                const origX = ann.x / scale;
                const origY = ann.y / scale;
                const origWidth = ann.width / scale;
                const origHeight = ann.height / scale;

                // Normalize bbox coordinates
                const normalizedGeometry = {
                    bbox: {
                        x_center: (origX + origWidth / 2) / originalWidth,
                        y_center: (origY + origHeight / 2) / originalHeight,
                        width: origWidth / originalWidth,
                        height: origHeight / originalHeight
                    }
                };

                const annotationData = {
                    asset_id: viewerState.currentImage.id,
                    label_class_id: labelClassId,
                    geometry_type: 'bbox',
                    geometry: normalizedGeometry,
                    is_normalized: true,
                    source: 'human',
                    confidence: 1.0,
                    annotator_name: 'user'
                };

                const response = await apiService.createAnnotation(annotationData);
                ann.id = response.id;
                ann.label_class_id = labelClassId;
                ann.saved = true;
                successCount++;

            } catch (error) {
                console.error(`[ViewerAnnotation] Failed to save annotation:`, error);
                failCount++;
            }
        }

        // Delete annotations from backend
        let deleteSuccessCount = 0;
        let deleteFailCount = 0;

        for (const ann of deletedAnnotations) {
            try {
                if (ann.id) {
                    await apiService.deleteAnnotation(ann.id);
                    deleteSuccessCount++;
                    console.log(`[ViewerAnnotation] Deleted annotation ${ann.id} from backend`);
                }
            } catch (error) {
                console.error(`[ViewerAnnotation] Failed to delete annotation ${ann.id}:`, error);
                deleteFailCount++;
            }
        }

        // Clear deleted annotations list after processing
        if (deleteSuccessCount > 0) {
            viewerState.deletedAnnotations = [];
        }

        console.log(`[ViewerAnnotation] Save complete: Created/Updated: ${successCount} success, ${failCount} failed, Deleted: ${deleteSuccessCount} success, ${deleteFailCount} failed`);

        // Show appropriate toast messages
        if (successCount > 0 || deleteSuccessCount > 0) {
            let message = '';
            if (successCount > 0) message += `Saved ${successCount} annotation(s)`;
            if (deleteSuccessCount > 0) {
                if (message) message += ', ';
                message += `Deleted ${deleteSuccessCount} annotation(s)`;
            }
            showToast(message, 'success');
            redrawViewerCanvas();
            updateViewerAnnotationList();

            // Update image annotation status
            updateImageAnnotationStatus();
        }

        if (failCount > 0 || deleteFailCount > 0) {
            let message = 'Failed: ';
            if (failCount > 0) message += `${failCount} save error(s)`;
            if (deleteFailCount > 0) {
                if (failCount > 0) message += ', ';
                message += `${deleteFailCount} delete error(s)`;
            }
            showToast(message, 'error');
        }

        // After saving/deleting annotations → regenerate YOLO txt

        // Build class map(display_name → class object)
        const clsMap = new Map();
        const labelClasses = await apiService.get(`/datasets/${page.datasetId}/label-classes`);

        labelClasses.forEach(cls => {
            clsMap.set(cls.display_name.trim().toLowerCase(), cls);
        });

        // Normalize viewer annotations
        const normalized = viewerState.annotations.map(ann => {
            const scale = viewerState.scale;
            const OW = viewerState.originalWidth;
            const OH = viewerState.originalHeight;

            const ox = ann.x / scale;
            const oy = ann.y / scale;
            const ow = ann.width / scale;
            const oh = ann.height / scale;

            return {
                className: ann.className.trim().toLowerCase(),
                x_center: (ox + ow / 2) / OW,
                y_center: (oy + oh / 2) / OH,
                width: ow / OW,
                height: oh / OH
            };
        });

        // Convert to YOLO format
        const yoloTxt = convertToYOLO(normalized, clsMap);

        // Upload to S3
        const txtFile = viewerState.currentImage.filename.replace(/\.[^/.]+$/, ".txt");
        await apiService.uploadLabel(page.datasetId, txtFile, yoloTxt);

        showToast("Label file updated successfully", "success");

    } catch (error) {
        console.error('[ViewerAnnotation] Save error:', error);
        showToast('Failed to save annotations', 'error');
    }
}

// Update image annotation status and refresh gallery
function updateImageAnnotationStatus() {
    if (!viewerState.currentImage) return;

    const page = window.currentDatasetDetailPage;
    if (!page) return;

    // Check if image has any annotations
    const hasAnnotations = viewerState.annotations.length > 0;
    const imageId = viewerState.currentImage.id;

    // Update the image in the page's images array
    const imageIndex = page.images.findIndex(img => img.id === imageId);
    if (imageIndex !== -1) {
        page.images[imageIndex].is_annotated = hasAnnotations;
        console.log(`[UpdateStatus] Updated image ${imageId} is_annotated to ${hasAnnotations}`);
    }

    // Update filteredImages if it exists
    if (page.filteredImages) {
        const filteredIndex = page.filteredImages.findIndex(img => img.id === imageId);
        if (filteredIndex !== -1) {
            page.filteredImages[filteredIndex].is_annotated = hasAnnotations;
        }
    }

    // Update the badge in the gallery card (find by onclick attribute or data attribute)
    const galleryCards = document.querySelectorAll('.card.hover-shadow');
    galleryCards.forEach(card => {
        // Check if this card is for the current image
        const img = card.querySelector(`img[data-asset-id="${imageId}"]`);
        if (img) {
            // Find the badge in the card body
            const badge = card.querySelector('.badge.badge-sm');
            if (badge) {
                if (hasAnnotations) {
                    badge.className = 'badge bg-success badge-sm';
                    badge.textContent = 'Annotated';
                } else {
                    badge.className = 'badge bg-warning badge-sm';
                    badge.textContent = 'Pending';
                }
                console.log(`[UpdateStatus] Updated badge for image ${imageId} to ${hasAnnotations ? 'Annotated' : 'Pending'}`);
            }
        }
    });
}

// Generate random color for label class
function getRandomColorForLabel() {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
}


function convertToYOLO(normalizedAnnotations, classMap) {
    return normalizedAnnotations
        .map(a => {
            const cls = classMap.get(a.className.trim().toLowerCase());
            return `${cls.yolo_index} ${a.x_center} ${a.y_center} ${a.width} ${a.height}`;
        })
        .join("\n");
}