// Dataset Helper Functions for Upload Modal

function showUploadDatasetModal() {
    console.log('[DatasetsPage] Showing upload dataset modal...');
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
                            <label for="image-files" class="form-label">Select Images</label>
                            <input type="file" class="form-control" id="image-files"
                                   accept="image/*" multiple>
                            <small class="text-muted">Supports JPG, PNG. Max 100 files at once.</small>
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

    loadDatasetsForUpload();

    document.getElementById('upload-dataset-select').addEventListener('change', (e) => {
        const newDatasetFields = document.getElementById('new-dataset-fields');
        newDatasetFields.style.display = e.target.value === '' ? 'block' : 'none';
    });

    document.getElementById('image-files').addEventListener('change', (e) => {
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
        console.error('[DatasetsPage] Error loading datasets for upload:', error);
    }
}

function showFilePreview(files) {
    if (files.length === 0) return;

    const preview = document.getElementById('upload-preview');
    const filesPreview = document.getElementById('files-preview');

    preview.classList.remove('d-none');

    let html = `<p class="mb-2"><strong>${files.length} file(s) selected</strong></p><ul class="list-unstyled mb-0">`;

    Array.from(files).slice(0, 10).forEach(file => {
        const size = formatFileSize(file.size);
        html += `<li class="small">
            <i class="bi bi-file-image me-1"></i>
            ${file.name} (${size})
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
    const datasetId = document.getElementById('upload-dataset-select').value;
    const newDatasetName = document.getElementById('new-dataset-name').value;
    const newDatasetDescription = document.getElementById('new-dataset-description').value;
    const files = document.getElementById('image-files').files;

    if (files.length === 0) {
        showToast('Please select files to upload', 'error');
        return;
    }

    if (datasetId === '' && !newDatasetName) {
        showToast('Please enter a dataset name or select an existing dataset', 'error');
        return;
    }

    console.log('[DatasetsPage] Starting presigned URL upload...', {
        datasetId,
        newDatasetName,
        filesCount: files.length
    });

    document.getElementById('upload-progress').classList.remove('d-none');
    document.getElementById('start-upload-btn').disabled = true;

    const progressBar = document.getElementById('upload-progress-bar');
    const statusText = document.getElementById('upload-status');

    try {
        // Progress callback for tracking upload progress
        const onProgress = (current, total) => {
            const percent = Math.round((current / total) * 100);
            progressBar.style.width = percent + '%';
            statusText.textContent = `Uploading... ${current}/${total} files (${percent}%)`;
        };

        // Use presigned URL upload method
        const result = await apiService.uploadWithPresignedUrl(
            files,
            datasetId || null,
            newDatasetName || null,
            newDatasetDescription || null,
            onProgress
        );

        console.log('[DatasetsPage] Upload result:', result);

        progressBar.style.width = '100%';
        statusText.textContent = 'Upload complete!';

        const successCount = result.successful_count || 0;
        const failedCount = result.failed_count || 0;

        if (failedCount > 0) {
            showToast(`Upload completed with warnings: ${successCount} succeeded, ${failedCount} failed`, 'warning');
        } else {
            showToast(`Dataset uploaded successfully! ${successCount} files uploaded.`, 'success');
        }

        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('uploadDatasetModal'));
        if (modal) modal.hide();

        // Reload datasets without page refresh and select the uploaded dataset
        if (window.currentPageInstance && typeof window.currentPageInstance.reloadDatasets === 'function') {
            await window.currentPageInstance.reloadDatasets(result.dataset_id);
        }

    } catch (error) {
        console.error('[DatasetsPage] Upload error:', error);
        showToast('Upload failed: ' + error.message, 'error');
        document.getElementById('start-upload-btn').disabled = false;
        progressBar.style.width = '0%';
        statusText.textContent = 'Upload failed';
    }
}

function navigateToAutoAnnotate(datasetId) {
    window.location.hash = `#/auto-annotate/${datasetId}`;
}

function navigateToDatasetDetail(datasetId) {
    window.location.hash = `#/dataset-detail/${datasetId}`;
}

console.log('[DatasetsHelpers] Upload modal helpers loaded');
