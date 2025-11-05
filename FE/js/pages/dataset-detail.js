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
        this.filterClass = 'all'; // all, or specific class name
        this.minConfidence = 0; // 0-100, for annotation display threshold
    }

    async init() {
        await this.loadDataset();
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
                        <div class="card h-100 hover-shadow" style="cursor: pointer;"
                             onclick="viewImage(${image.id})">
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
                                    <div class="border rounded p-2" style="background-color: #f8f9fa; max-height: 600px; overflow: auto;">
                                        <canvas id="annotation-canvas" style="max-width: 100%; display: block;"></canvas>
                                    </div>
                                </div>
                                <div class="col-lg-4">
                                    <h6 class="fw-bold mb-3">
                                        <i class="bi bi-tags me-2"></i>Annotations
                                    </h6>
                                    <div id="annotation-list" class="list-group" style="max-height: 500px; overflow-y: auto;">
                                        <div class="text-muted text-center py-3">Loading...</div>
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
                    const hasClass = annotations.some(ann => ann.class_name === this.filterClass);
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

    // Load annotations (filtered by min confidence if set)
    const minConf = page.minConfidence > 0 ? page.minConfidence / 100 : null;
    console.log('[ViewImage] Using min confidence:', minConf);
    const annotations = await apiService.getImageAnnotations(imageId, minConf);
    console.log('[ViewImage] Loaded annotations:', annotations);

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('imageViewerModal'));
    modal.show();

    // Wait for modal to be shown
    setTimeout(() => {
        drawImageWithAnnotations(image, annotations);
    }, 100);
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

        // Set canvas size to match image
        canvas.width = img.width;
        canvas.height = img.height;

        // Draw image
        ctx.drawImage(img, 0, 0);

        // Draw annotations
        annotations.forEach((ann, idx) => {
            console.log(`[DrawImage] Drawing annotation ${idx}:`, ann);

            // Convert normalized coordinates to pixel coordinates
            const x_center = ann.x_center * img.width;
            const y_center = ann.y_center * img.height;
            const width = ann.width * img.width;
            const height = ann.height * img.height;

            // Calculate top-left corner
            const x = x_center - width / 2;
            const y = y_center - height / 2;

            // Draw bounding box
            ctx.strokeStyle = getColorForClass(ann.class_id);
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, width, height);

            // Draw label background
            const label = `${ann.class_name} ${(ann.confidence * 100).toFixed(0)}%`;
            ctx.font = '16px Arial';
            const textMetrics = ctx.measureText(label);
            const textHeight = 20;

            ctx.fillStyle = getColorForClass(ann.class_id);
            ctx.fillRect(x, y - textHeight, textMetrics.width + 10, textHeight);

            // Draw label text
            ctx.fillStyle = 'white';
            ctx.fillText(label, x + 5, y - 5);
        });

        // Update annotation list
        updateAnnotationList(annotations);
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

    listContainer.innerHTML = annotations.map((ann, idx) => `
        <div class="list-group-item">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <span class="badge" style="background-color: ${getColorForClass(ann.class_id)}">${ann.class_id}</span>
                    <strong>${ann.class_name}</strong>
                </div>
                <small class="text-muted">${(ann.confidence * 100).toFixed(1)}%</small>
            </div>
            <small class="text-muted">
                ${ann.is_auto_generated ? '<i class="bi bi-robot"></i> Auto' : '<i class="bi bi-pencil"></i> Manual'}
            </small>
        </div>
    `).join('');
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
    }
}
