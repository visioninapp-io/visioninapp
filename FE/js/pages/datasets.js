// Datasets Page Component

class DatasetsPage {
    constructor() {
        this.datasets = [];
        this.stats = {
            totalImages: 0,
            totalDatasets: 0,
            totalClasses: 0,
            autoAnnotatedPercent: 0
        };
        this.searchQuery = '';
    }

    async init() {
        await this.loadDatasets();
        await this.loadStats();
        this.attachEventListeners();
    }

    async loadDatasets() {
        try {
            console.log('Loading datasets from API...');
            this.datasets = await apiService.getDatasets();
            console.log('Loaded datasets:', this.datasets);
        } catch (error) {
            console.error('Error loading datasets:', error);
            console.error('Error details:', error.message, error.stack);
            this.datasets = [];

            // Show error to user
            if (typeof showToast === 'function') {
                showToast('Failed to load datasets. Please check if backend server is running.', 'error');
            }
        }
    }

    async loadStats() {
        try {
            console.log('Loading dataset stats...');
            const statsData = await apiService.getDatasetStats();
            this.stats = statsData;
            console.log('Loaded stats:', this.stats);
        } catch (error) {
            console.error('Error loading stats:', error);
            // Calculate stats from datasets if API fails
            this.calculateStatsFromDatasets();
        }
    }

    calculateStatsFromDatasets() {
        this.stats.totalDatasets = this.datasets.length;
        this.stats.totalImages = this.datasets.reduce((sum, d) => sum + (d.total_images || 0), 0);
        this.stats.totalClasses = this.datasets.reduce((sum, d) => sum + (d.total_classes || 0), 0);

        const annotatedImages = this.datasets.reduce((sum, d) => sum + (d.annotated_images || 0), 0);
        this.stats.autoAnnotatedPercent = this.stats.totalImages > 0
            ? Math.round((annotatedImages / this.stats.totalImages) * 100)
            : 0;
    }

    render() {
        return `
            <div class="min-vh-100 bg-light">
                <div class="container py-4">
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

                    <!-- Datasets Table Card -->
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-white border-0 py-3">
                            <div class="row align-items-center">
                                <div class="col">
                                    <h5 class="mb-1 fw-bold">Datasets</h5>
                                    <p class="text-muted mb-0 small">Manage and organize your training datasets</p>
                                </div>
                                <div class="col-auto">
                                    <button class="btn btn-outline-primary btn-sm me-2" onclick="showConfigureDatasetModal()">
                                        <i class="bi bi-gear me-1"></i> Configure
                                    </button>
                                    <button class="btn btn-primary btn-sm" onclick="showUploadDatasetModal()">
                                        <i class="bi bi-upload me-1"></i> Upload Dataset
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div class="card-body">
                            <!-- Search -->
                            <div class="mb-3">
                                <div class="input-group">
                                    <span class="input-group-text bg-white">
                                        <i class="bi bi-search"></i>
                                    </span>
                                    <input type="text" class="form-control" id="dataset-search" placeholder="Search datasets...">
                                </div>
                            </div>

                            <!-- Loading State -->
                            <div id="datasets-loading" class="text-center py-5 d-none">
                                <div class="spinner-border text-primary" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                                <p class="text-muted mt-2">Loading datasets...</p>
                            </div>

                            <!-- Empty State -->
                            <div id="datasets-empty" class="text-center py-5 ${this.datasets.length > 0 ? 'd-none' : ''}">
                                <i class="bi bi-database text-muted" style="font-size: 4rem;"></i>
                                <h5 class="mt-3">No Datasets Yet</h5>
                                <p class="text-muted">Upload your first dataset to get started</p>
                                <button class="btn btn-primary" onclick="showUploadDatasetModal()">
                                    <i class="bi bi-upload me-1"></i> Upload Dataset
                                </button>
                            </div>

                            <!-- Datasets List -->
                            <div id="datasets-list" class="list-group list-group-flush ${this.datasets.length === 0 ? 'd-none' : ''}">
                                ${this.renderDatasetsList()}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderDatasetsList() {
        const filteredDatasets = this.datasets.filter(dataset => {
            if (!this.searchQuery) return true;
            return dataset.name.toLowerCase().includes(this.searchQuery.toLowerCase()) ||
                   (dataset.description && dataset.description.toLowerCase().includes(this.searchQuery.toLowerCase()));
        });

        if (filteredDatasets.length === 0) {
            return `
                <div class="text-center py-4">
                    <p class="text-muted">No datasets found matching "${this.searchQuery}"</p>
                </div>
            `;
        }

        return filteredDatasets.map(dataset => this.renderDatasetCard(dataset)).join('');
    }

    renderDatasetCard(dataset) {
        const statusMap = {
            'created': { class: 'bg-secondary', text: 'Created' },
            'uploading': { class: 'bg-info', text: 'Uploading' },
            'processing': { class: 'bg-warning', text: 'Processing' },
            'ready': { class: 'bg-success', text: 'Ready' },
            'annotated': { class: 'bg-primary', text: 'Annotated' },
            'error': { class: 'bg-danger', text: 'Error' }
        };

        const status = statusMap[dataset.status] || statusMap['created'];
        const lastModified = dataset.updated_at
            ? this.formatTimeAgo(new Date(dataset.updated_at))
            : this.formatTimeAgo(new Date(dataset.created_at));

        return `
            <div class="list-group-item border rounded mb-2 hover-shadow" data-dataset-id="${dataset.id}">
                <div class="row align-items-center">
                    <div class="col">
                        <div class="d-flex align-items-center gap-2 mb-1">
                            <h6 class="mb-0 fw-bold">${dataset.name}</h6>
                            <span class="badge ${status.class}">${status.text}</span>
                        </div>
                        ${dataset.description ? `<p class="text-muted small mb-2">${dataset.description}</p>` : ''}
                        <div class="d-flex gap-3 text-muted small">
                            <span><i class="bi bi-image me-1"></i>${dataset.total_images || 0} images</span>
                            <span><i class="bi bi-tag me-1"></i>${dataset.total_classes || 0} classes</span>
                            <span><i class="bi bi-clock me-1"></i>Modified ${lastModified}</span>
                        </div>
                    </div>
                    <div class="col-auto">
                        <button class="btn btn-outline-primary btn-sm me-2" onclick="navigateToAutoAnnotate(${dataset.id})">
                            <i class="bi bi-sparkles me-1"></i> Auto-Annotate
                        </button>
                        <button class="btn btn-primary btn-sm" onclick="navigateToDatasetDetail(${dataset.id})">
                            <i class="bi bi-eye me-1"></i> View Dataset
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    formatTimeAgo(date) {
        const seconds = Math.floor((new Date() - date) / 1000);

        const intervals = {
            year: 31536000,
            month: 2592000,
            week: 604800,
            day: 86400,
            hour: 3600,
            minute: 60
        };

        for (const [unit, secondsInUnit] of Object.entries(intervals)) {
            const interval = Math.floor(seconds / secondsInUnit);
            if (interval >= 1) {
                return `${interval} ${unit}${interval > 1 ? 's' : ''} ago`;
            }
        }

        return 'just now';
    }

    attachEventListeners() {
        // Search functionality
        const searchInput = document.getElementById('dataset-search');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.searchQuery = e.target.value;
                this.updateDatasetsList();
            });
        }
    }

    updateDatasetsList() {
        const listContainer = document.getElementById('datasets-list');
        if (listContainer) {
            listContainer.innerHTML = this.renderDatasetsList();
        }
    }
}

// Configure Dataset Modal
function showConfigureDatasetModal() {
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
                                <label class="form-label">Auto-Annotation Settings</label>
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="enable-auto-annotation" checked>
                                    <label class="form-check-label" for="enable-auto-annotation">
                                        Enable automatic annotation
                                    </label>
                                </div>
                            </div>

                            <div id="auto-annotation-settings" class="border rounded p-3 mb-3">
                                <div class="mb-3">
                                    <label for="confidence-threshold" class="form-label">
                                        Confidence Threshold: <span id="confidence-value">0.5</span>
                                    </label>
                                    <input type="range" class="form-range" id="confidence-threshold"
                                           min="0" max="1" step="0.05" value="0.5">
                                    <small class="text-muted">Only annotations above this confidence will be saved</small>
                                </div>

                                <div class="mb-0">
                                    <label for="annotation-format" class="form-label">Annotation Format</label>
                                    <select class="form-select" id="annotation-format">
                                        <option value="yolo" selected>YOLO</option>
                                        <option value="coco">COCO</option>
                                        <option value="pascal-voc">Pascal VOC</option>
                                    </select>
                                </div>
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

    // Remove existing modal
    const existingModal = document.getElementById('configureDatasetModal');
    if (existingModal) existingModal.remove();

    document.body.insertAdjacentHTML('beforeend', modalHTML);
    const modal = new bootstrap.Modal(document.getElementById('configureDatasetModal'));

    // Event listeners
    document.getElementById('confidence-threshold').addEventListener('input', (e) => {
        document.getElementById('confidence-value').textContent = e.target.value;
    });

    document.getElementById('enable-auto-annotation').addEventListener('change', (e) => {
        document.getElementById('auto-annotation-settings').style.display =
            e.target.checked ? 'block' : 'none';
    });

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
        const enableAutoAnnotation = document.getElementById('enable-auto-annotation').checked;
        const confidenceThreshold = parseFloat(document.getElementById('confidence-threshold').value);
        const annotationFormat = document.getElementById('annotation-format').value;

        // Get all class names
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

            const createdDataset = await apiService.createDataset(datasetData);

            modal.hide();
            showToast('Dataset configured successfully!', 'success');

            // Dynamically update the datasets list without page reload
            setTimeout(async () => {
                if (window.currentPageInstance && window.currentPageInstance.constructor.name === 'DatasetsPage') {
                    // Reload datasets from API
                    await window.currentPageInstance.loadDatasets();
                    await window.currentPageInstance.loadStats();

                    // Update the DOM
                    const listContainer = document.getElementById('datasets-list');
                    const emptyState = document.getElementById('datasets-empty');

                    if (listContainer && emptyState) {
                        if (window.currentPageInstance.datasets.length > 0) {
                            listContainer.classList.remove('d-none');
                            listContainer.innerHTML = window.currentPageInstance.renderDatasetsList();
                            emptyState.classList.add('d-none');
                        }
                    }

                    // Update stats
                    document.getElementById('stat-total-images').textContent = window.currentPageInstance.stats.totalImages.toLocaleString();
                    document.getElementById('stat-datasets').textContent = window.currentPageInstance.stats.totalDatasets;
                    document.getElementById('stat-classes').textContent = window.currentPageInstance.stats.totalClasses;
                    document.getElementById('stat-annotated').textContent = window.currentPageInstance.stats.autoAnnotatedPercent + '%';
                } else {
                    // Fallback: navigate to datasets page
                    window.location.hash = '#/datasets';
                }
            }, 500);
        } catch (error) {
            console.error('Error creating dataset:', error);
            showToast('Failed to create dataset: ' + error.message, 'error');
        }
    });

    modal.show();
}

// Upload Dataset Modal
function showUploadDatasetModal() {
    const modalHTML = `
        <div class="modal fade" id="uploadDatasetModal" tabindex="-1" aria-hidden="true">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="bi bi-upload me-2"></i>Upload Dataset
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="mb-3">
                            <label for="upload-dataset-select" class="form-label">Select Dataset</label>
                            <select class="form-select" id="upload-dataset-select">
                                <option value="">-- Create New Dataset --</option>
                            </select>
                        </div>

                        <div id="new-dataset-fields" class="mb-3">
                            <div class="mb-3">
                                <label for="new-dataset-name" class="form-label">Dataset Name *</label>
                                <input type="text" class="form-control" id="new-dataset-name" required>
                            </div>
                            <div class="mb-3">
                                <label for="new-dataset-description" class="form-label">Description</label>
                                <textarea class="form-control" id="new-dataset-description" rows="2"></textarea>
                            </div>
                        </div>

                        <div class="mb-3">
                            <label class="form-label">Upload Method</label>
                            <div class="btn-group w-100" role="group">
                                <input type="radio" class="btn-check" name="upload-method" id="upload-images" value="images" checked>
                                <label class="btn btn-outline-primary" for="upload-images">
                                    <i class="bi bi-image me-1"></i> Images
                                </label>

                                <input type="radio" class="btn-check" name="upload-method" id="upload-zip" value="zip">
                                <label class="btn btn-outline-primary" for="upload-zip">
                                    <i class="bi bi-file-zip me-1"></i> ZIP File
                                </label>
                            </div>
                        </div>

                        <div id="upload-images-section">
                            <div class="mb-3">
                                <label for="image-files" class="form-label">Select Images</label>
                                <input type="file" class="form-control" id="image-files"
                                       accept="image/*" multiple>
                                <small class="text-muted">Supports JPG, PNG. Max 100 files at once.</small>
                            </div>
                        </div>

                        <div id="upload-zip-section" class="d-none">
                            <div class="mb-3">
                                <label for="zip-file" class="form-label">Select ZIP File</label>
                                <input type="file" class="form-control" id="zip-file" accept=".zip">
                                <small class="text-muted">ZIP should contain images in the root or folders</small>
                            </div>
                        </div>

                        <div id="upload-preview" class="mb-3 d-none">
                            <label class="form-label">Selected Files</label>
                            <div id="files-preview" class="border rounded p-3" style="max-height: 200px; overflow-y: auto;">
                            </div>
                        </div>

                        <div id="upload-progress" class="d-none">
                            <label class="form-label">Upload Progress</label>
                            <div class="progress">
                                <div id="upload-progress-bar" class="progress-bar progress-bar-striped progress-bar-animated"
                                     role="progressbar" style="width: 0%"></div>
                            </div>
                            <p class="text-muted small mt-2" id="upload-status">Preparing upload...</p>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" id="start-upload-btn">
                            <i class="bi bi-upload me-1"></i> Start Upload
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    const existingModal = document.getElementById('uploadDatasetModal');
    if (existingModal) existingModal.remove();

    document.body.insertAdjacentHTML('beforeend', modalHTML);
    const modal = new bootstrap.Modal(document.getElementById('uploadDatasetModal'));

    // Load existing datasets
    loadDatasetsForUpload();

    // Event listeners
    document.querySelectorAll('[name="upload-method"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'images') {
                document.getElementById('upload-images-section').classList.remove('d-none');
                document.getElementById('upload-zip-section').classList.add('d-none');
            } else {
                document.getElementById('upload-images-section').classList.add('d-none');
                document.getElementById('upload-zip-section').classList.remove('d-none');
            }
        });
    });

    document.getElementById('upload-dataset-select').addEventListener('change', (e) => {
        const newDatasetFields = document.getElementById('new-dataset-fields');
        newDatasetFields.style.display = e.target.value === '' ? 'block' : 'none';
    });

    document.getElementById('image-files').addEventListener('change', (e) => {
        showFilePreview(e.target.files);
    });

    document.getElementById('zip-file').addEventListener('change', (e) => {
        showFilePreview(e.target.files);
    });

    document.getElementById('start-upload-btn').addEventListener('click', handleUpload);

    modal.show();
}

async function loadDatasetsForUpload() {
    try {
        const datasets = await apiService.getDatasets();
        const select = document.getElementById('upload-dataset-select');

        datasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.id;
            option.textContent = dataset.name;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading datasets:', error);
    }
}

function showFilePreview(files) {
    if (files.length === 0) return;

    const preview = document.getElementById('upload-preview');
    const filesPreview = document.getElementById('files-preview');

    preview.classList.remove('d-none');

    // Check for duplicate filenames
    const fileNames = Array.from(files).map(f => f.name);
    const duplicates = fileNames.filter((name, index) => fileNames.indexOf(name) !== index);

    let html = `<p class="mb-2"><strong>${files.length} file(s) selected</strong></p>`;

    if (duplicates.length > 0) {
        html += `<div class="alert alert-warning alert-sm py-2 mb-2">
            <i class="bi bi-exclamation-triangle me-1"></i>
            <strong>Warning:</strong> ${duplicates.length} duplicate filename(s) detected: ${[...new Set(duplicates)].join(', ')}
        </div>`;
    }

    html += `<ul class="list-unstyled mb-0">`;

    Array.from(files).slice(0, 10).forEach(file => {
        const isDuplicate = duplicates.includes(file.name);
        html += `<li class="small ${isDuplicate ? 'text-warning' : ''}">
            <i class="bi bi-${isDuplicate ? 'exclamation-triangle' : 'file-image'} me-1"></i>
            ${file.name} (${formatFileSize(file.size)})
        </li>`;
    });

    if (files.length > 10) {
        html += `<li class="small text-muted">... and ${files.length - 10} more files</li>`;
    }

    html += '</ul>';
    filesPreview.innerHTML = html;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

async function handleUpload() {
    const uploadMethod = document.querySelector('[name="upload-method"]:checked').value;
    const datasetId = document.getElementById('upload-dataset-select').value;
    const newDatasetName = document.getElementById('new-dataset-name').value;
    const newDatasetDescription = document.getElementById('new-dataset-description').value;

    let files;
    if (uploadMethod === 'images') {
        files = document.getElementById('image-files').files;
    } else {
        files = document.getElementById('zip-file').files;
    }

    if (files.length === 0) {
        showToast('Please select files to upload', 'error');
        return;
    }

    if (datasetId === '' && !newDatasetName) {
        showToast('Please enter a dataset name or select an existing dataset', 'error');
        return;
    }

    // Check for duplicate filenames
    const fileNames = Array.from(files).map(f => f.name);
    const duplicates = fileNames.filter((name, index) => fileNames.indexOf(name) !== index);

    if (duplicates.length > 0) {
        const uniqueDuplicates = [...new Set(duplicates)];
        const confirmMessage = `Warning: ${uniqueDuplicates.length} duplicate filename(s) detected:\n\n${uniqueDuplicates.join('\n')}\n\nDuplicate files will be renamed with unique IDs. Continue?`;

        if (!confirm(confirmMessage)) {
            return;
        }
    }

    // Show progress
    document.getElementById('upload-progress').classList.remove('d-none');
    document.getElementById('start-upload-btn').disabled = true;

    try {
        const formData = new FormData();

        // Add files
        Array.from(files).forEach(file => {
            formData.append('files', file);
        });

        // Add dataset info
        if (datasetId) {
            formData.append('dataset_id', datasetId);
        } else {
            formData.append('name', newDatasetName);
            if (newDatasetDescription) {
                formData.append('description', newDatasetDescription);
            }
        }

        // Simulate progress
        let progress = 0;
        const progressBar = document.getElementById('upload-progress-bar');
        const statusText = document.getElementById('upload-status');

        const progressInterval = setInterval(() => {
            progress += 10;
            if (progress >= 90) clearInterval(progressInterval);
            progressBar.style.width = progress + '%';
            statusText.textContent = `Uploading... ${progress}%`;
        }, 500);

        const result = await apiService.uploadDataset(formData);

        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        statusText.textContent = 'Upload complete!';

        showToast('Dataset uploaded successfully!', 'success');

        setTimeout(async () => {
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('uploadDatasetModal'));
            if (modal) modal.hide();

            // Reload datasets without full page reload
            if (window.currentPageInstance && window.currentPageInstance.constructor.name === 'DatasetsPage') {
                await window.currentPageInstance.loadDatasets();
                await window.currentPageInstance.loadStats();

                // Update the DOM
                const listContainer = document.getElementById('datasets-list');
                const emptyState = document.getElementById('datasets-empty');

                if (listContainer && emptyState) {
                    if (window.currentPageInstance.datasets.length > 0) {
                        listContainer.classList.remove('d-none');
                        listContainer.innerHTML = window.currentPageInstance.renderDatasetsList();
                        emptyState.classList.add('d-none');
                    }
                }

                // Update stats
                document.getElementById('stat-total-images').textContent = window.currentPageInstance.stats.totalImages.toLocaleString();
                document.getElementById('stat-datasets').textContent = window.currentPageInstance.stats.totalDatasets;
                document.getElementById('stat-classes').textContent = window.currentPageInstance.stats.totalClasses;
                document.getElementById('stat-annotated').textContent = window.currentPageInstance.stats.autoAnnotatedPercent + '%';
            } else {
                // Fallback: navigate to datasets page
                window.location.hash = '#/datasets';
            }
        }, 1500);

    } catch (error) {
        console.error('Upload error:', error);
        showToast('Upload failed: ' + error.message, 'error');
        document.getElementById('start-upload-btn').disabled = false;
    }
}

// Navigation functions
function navigateToAutoAnnotate(datasetId) {
    window.location.hash = `#/auto-annotate/${datasetId}`;
}

function navigateToDatasetDetail(datasetId) {
    window.location.hash = `#/dataset-detail/${datasetId}`;
}
