// Upload YOLO Labels Page Component

class UploadLabelsPage {
    constructor(datasetId) {
        this.datasetId = datasetId;
        this.dataset = null;
        this.labelFiles = [];
        this.dataYamlFile = null;
        this.overwriteExisting = false;
        this.isUploading = false;
        this.progress = 0;
    }

    async init() {
        console.log('[UploadLabelsPage] Initializing...');
        await this.loadDataset();
        this.updatePage();
    }

    async loadDataset() {
        try {
            console.log('[UploadLabelsPage] Loading dataset...');
            this.dataset = await apiService.getDataset(this.datasetId);
            console.log('[UploadLabelsPage] Dataset loaded:', this.dataset);
        } catch (error) {
            console.error('[UploadLabelsPage] Error loading dataset:', error);
            showToast('Failed to load dataset', 'error');
        }
    }

    updatePage() {
        const app = document.getElementById('app');
        if (app) {
            app.innerHTML = this.render();
            this.attachEventListeners();
        }
    }

    render() {
        return `
            <div class="container-fluid py-4">
                <div class="row">
                    <div class="col-12">
                        <div class="d-flex justify-content-between align-items-center mb-4">
                            <h2 class="mb-0">
                                <i class="bi bi-upload me-2"></i>Upload YOLO Labels
                            </h2>
                            <a href="#/dataset-detail/${this.datasetId}" class="btn btn-outline-secondary">
                                <i class="bi bi-arrow-left me-1"></i>Back to Dataset
                            </a>
                        </div>

                        ${this.dataset ? `
                            <div class="card border-0 shadow-sm mb-4">
                                <div class="card-header bg-white border-0">
                                    <h5 class="mb-0 fw-bold">
                                        <i class="bi bi-database me-2"></i>Dataset Information
                                    </h5>
                                </div>
                                <div class="card-body">
                                    <div class="row">
                                        <div class="col-md-6">
                                            <p class="mb-1"><strong>Dataset Name:</strong> ${this.dataset.name}</p>
                                            <p class="mb-1"><strong>Total Images:</strong> ${this.dataset.total_images || 0}</p>
                                        </div>
                                        <div class="col-md-6">
                                            <p class="mb-1"><strong>Annotated Images:</strong> ${this.dataset.annotated_images || 0}</p>
                                            <p class="mb-1"><strong>Total Annotations:</strong> ${this.dataset.total_annotations || 0}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ` : '<div class="text-center py-5"><div class="spinner-border" role="status"></div></div>'}

                        <!-- Upload Instructions -->
                        <div class="card border-0 shadow-sm mb-4">
                            <div class="card-header bg-white border-0">
                                <h5 class="mb-0 fw-bold">
                                    <i class="bi bi-info-circle me-2"></i>Upload Instructions
                                </h5>
                            </div>
                            <div class="card-body">
                                <ol>
                                    <li><strong>Label Files (.txt):</strong> Upload one or more YOLO format label files. Each file should match an image filename (e.g., <code>image1.jpg</code> → <code>image1.txt</code>)</li>
                                    <li><strong>data.yaml (Optional):</strong> Upload a data.yaml file with class definitions. If not provided, classes will be auto-created as <code>class_0</code>, <code>class_1</code>, etc.</li>
                                    <li><strong>YOLO Format:</strong> Each line in a label file should be: <code>class_id x_center y_center width height</code> (all normalized 0-1)</li>
                                    <li><strong>data.yaml Format:</strong>
                                        <pre class="bg-light p-2 mt-2"><code>nc: 3
names: ['class1', 'class2', 'class3']</code></pre>
                                    </li>
                                </ol>
                            </div>
                        </div>

                        <!-- Upload Form -->
                        <div class="card border-0 shadow-sm mb-4">
                            <div class="card-header bg-white border-0">
                                <h5 class="mb-0 fw-bold">
                                    <i class="bi bi-cloud-upload me-2"></i>Upload Files
                                </h5>
                            </div>
                            <div class="card-body">
                                <form id="upload-labels-form">
                                    <!-- Label Files -->
                                    <div class="mb-4">
                                        <label for="label-files" class="form-label fw-bold">
                                            Label Files (.txt) <span class="text-danger">*</span>
                                        </label>
                                        <input 
                                            type="file" 
                                            class="form-control" 
                                            id="label-files" 
                                            accept=".txt"
                                            multiple
                                            required
                                            ${this.isUploading ? 'disabled' : ''}
                                        >
                                        <small class="form-text text-muted">
                                            Select one or more YOLO format label files. Each file should correspond to an image in the dataset.
                                        </small>
                                        <div id="label-files-list" class="mt-2"></div>
                                    </div>

                                    <!-- data.yaml File -->
                                    <div class="mb-4">
                                        <label for="data-yaml" class="form-label fw-bold">
                                            data.yaml (Optional)
                                        </label>
                                        <input 
                                            type="file" 
                                            class="form-control" 
                                            id="data-yaml" 
                                            accept=".yaml,.yml"
                                            ${this.isUploading ? 'disabled' : ''}
                                        >
                                        <small class="form-text text-muted">
                                            Optional: Upload data.yaml file with class definitions. If not provided, classes will be auto-created.
                                        </small>
                                        <div id="data-yaml-name" class="mt-2"></div>
                                    </div>

                                    <!-- Overwrite Existing -->
                                    <div class="mb-4">
                                        <div class="form-check">
                                            <input 
                                                class="form-check-input" 
                                                type="checkbox" 
                                                id="overwrite-existing"
                                                ${this.overwriteExisting ? 'checked' : ''}
                                                ${this.isUploading ? 'disabled' : ''}
                                            >
                                            <label class="form-check-label" for="overwrite-existing">
                                                <strong>Overwrite existing annotations</strong>
                                                <br><small class="text-muted">If checked, existing annotations for matching images will be deleted before adding new ones.</small>
                                            </label>
                                        </div>
                                    </div>

                                    <!-- Upload Button -->
                                    <div class="d-grid gap-2">
                                        <button 
                                            type="submit" 
                                            class="btn btn-primary btn-lg"
                                            id="upload-btn"
                                            ${this.isUploading ? 'disabled' : ''}
                                        >
                                            ${this.isUploading ? `
                                                <span class="spinner-border spinner-border-sm me-2" role="status"></span>
                                                Uploading...
                                            ` : `
                                                <i class="bi bi-cloud-upload me-2"></i>Upload Labels
                                            `}
                                        </button>
                                    </div>
                                </form>

                                <!-- Progress Bar -->
                                ${this.isUploading ? `
                                    <div class="mt-4">
                                        <div class="progress" style="height: 25px;">
                                            <div 
                                                class="progress-bar progress-bar-striped progress-bar-animated" 
                                                role="progressbar" 
                                                style="width: ${this.progress}%"
                                                aria-valuenow="${this.progress}" 
                                                aria-valuemin="0" 
                                                aria-valuemax="100"
                                            >
                                                ${this.progress}%
                                            </div>
                                        </div>
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    attachEventListeners() {
        // Label files input
        const labelFilesInput = document.getElementById('label-files');
        if (labelFilesInput) {
            labelFilesInput.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                this.labelFiles = files;
                this.updateLabelFilesList();
            });
        }

        // data.yaml input
        const dataYamlInput = document.getElementById('data-yaml');
        if (dataYamlInput) {
            dataYamlInput.addEventListener('change', (e) => {
                this.dataYamlFile = e.target.files[0] || null;
                this.updateDataYamlName();
            });
        }

        // Overwrite checkbox
        const overwriteCheckbox = document.getElementById('overwrite-existing');
        if (overwriteCheckbox) {
            overwriteCheckbox.addEventListener('change', (e) => {
                this.overwriteExisting = e.target.checked;
            });
        }

        // Form submit
        const form = document.getElementById('upload-labels-form');
        if (form) {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleUpload();
            });
        }
    }

    updateLabelFilesList() {
        const listDiv = document.getElementById('label-files-list');
        if (listDiv) {
            if (this.labelFiles.length === 0) {
                listDiv.innerHTML = '';
            } else {
                listDiv.innerHTML = `
                    <div class="alert alert-info mb-0">
                        <strong>${this.labelFiles.length}</strong> file(s) selected:
                        <ul class="mb-0 mt-2">
                            ${this.labelFiles.slice(0, 10).map(f => `<li>${f.name}</li>`).join('')}
                            ${this.labelFiles.length > 10 ? `<li><em>... and ${this.labelFiles.length - 10} more</em></li>` : ''}
                        </ul>
                    </div>
                `;
            }
        }
    }

    updateDataYamlName() {
        const nameDiv = document.getElementById('data-yaml-name');
        if (nameDiv) {
            if (this.dataYamlFile) {
                nameDiv.innerHTML = `
                    <div class="alert alert-success mb-0">
                        <i class="bi bi-check-circle me-1"></i>Selected: <strong>${this.dataYamlFile.name}</strong>
                    </div>
                `;
            } else {
                nameDiv.innerHTML = '';
            }
        }
    }

    async handleUpload() {
        if (this.labelFiles.length === 0) {
            showToast('Please select at least one label file', 'error');
            return;
        }

        this.isUploading = true;
        this.progress = 0;
        this.updatePage();

        try {
            showToast('Starting upload...', 'info');

            // Create FormData
            const formData = new FormData();
            
            // Add label files
            for (const file of this.labelFiles) {
                formData.append('label_files', file);
            }

            // Add data.yaml if provided
            if (this.dataYamlFile) {
                formData.append('data_yaml', this.dataYamlFile);
            }

            // Add overwrite_existing flag
            formData.append('overwrite_existing', this.overwriteExisting.toString());

            // Simulate progress
            const progressInterval = setInterval(() => {
                if (this.progress < 90) {
                    this.progress += 10;
                    this.updatePage();
                }
            }, 500);

            // Upload
            const result = await apiService.uploadYoloLabels(this.datasetId, formData);

            clearInterval(progressInterval);
            this.progress = 100;
            this.updatePage();

            // Show success message
            showToast(
                `Successfully uploaded ${result.successful_files} files and created ${result.total_annotations} annotations!`,
                'success'
            );

            // Redirect after delay
            setTimeout(() => {
                window.location.hash = `#/dataset-detail/${this.datasetId}`;
            }, 2000);

        } catch (error) {
            console.error('[UploadLabelsPage] Upload error:', error);
            showToast(`Upload failed: ${error.message || 'Unknown error'}`, 'error');
            this.isUploading = false;
            this.progress = 0;
            this.updatePage();
        }
    }
}

