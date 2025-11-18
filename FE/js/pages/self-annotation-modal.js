// Self-Annotation Modal - Manual bbox annotation tool
// Allows users to draw bounding boxes and assign class labels

let selfAnnotationState = {
    datasetId: null,
    datasetName: null,
    images: [],
    currentImageIndex: 0,
    currentImage: null,
    annotations: [],
    imageAnnotations: new Map(), // imageId -> [annotations] - frontend-only storage
    drawing: false,
    startX: 0,
    startY: 0,
    currentBox: null,
    canvas: null,
    ctx: null,
    img: null,
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    labelClasses: new Map(), // className -> { id, color }
    nextClassId: 1
};

// Show self-annotation modal
async function showSelfAnnotationModal(datasetId, datasetName) {
    console.log(`[SelfAnnotation] Opening modal for dataset ${datasetId}: ${datasetName}`);

    selfAnnotationState.datasetId = datasetId;
    selfAnnotationState.datasetName = datasetName;
    selfAnnotationState.annotations = [];

    // Create modal if doesn't exist
    if (!document.getElementById('selfAnnotationModal')) {
        createSelfAnnotationModal();
    }

    // Load dataset images
    await loadDatasetImages(datasetId);

    if (selfAnnotationState.images.length === 0) {
        showToast('No images found in this dataset', 'warning');
        return;
    }

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('selfAnnotationModal'));
    modal.show();

    // Load first image after modal is shown
    setTimeout(async () => {
        if (selfAnnotationState.images.length > 0) {
            console.log('[SelfAnnotation] Loading first image...');
            await loadImageToCanvas(0);
        }
    }, 300);
}

// Show self-annotation modal for a single image
async function showSingleImageSelfAnnotation(datasetId, datasetName, imageId) {
    console.log(`[SelfAnnotation] Opening modal for single image ${imageId} in dataset ${datasetId}`);

    selfAnnotationState.datasetId = datasetId;
    selfAnnotationState.datasetName = datasetName;
    selfAnnotationState.annotations = [];

    // Create modal if doesn't exist
    if (!document.getElementById('selfAnnotationModal')) {
        createSelfAnnotationModal();
    }

    // Load only the specified image
    try {
        const images = await apiService.getDatasetImages(datasetId);
        const targetImage = images.find(img => img.id === imageId);

        if (!targetImage) {
            showToast('Image not found', 'error');
            return;
        }

        // Set only this image
        selfAnnotationState.images = [targetImage];
        selfAnnotationState.currentImageIndex = 0;

        console.log('[SelfAnnotation] Loaded single image:', targetImage);

    } catch (error) {
        console.error('[SelfAnnotation] Failed to load image:', error);
        showToast('Failed to load image', 'error');
        return;
    }

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('selfAnnotationModal'));
    modal.show();

    // Load the image after modal is shown
    setTimeout(async () => {
        console.log('[SelfAnnotation] Loading single image...');
        await loadImageToCanvas(0);
    }, 300);
}

// Create modal HTML
function createSelfAnnotationModal() {
    const modalHTML = `
        <div class="modal fade" id="selfAnnotationModal" tabindex="-1" data-bs-backdrop="static" data-bs-keyboard="false">
            <div class="modal-dialog modal-fullscreen">
                <div class="modal-content">
                    <div class="modal-header bg-warning text-dark">
                        <h5 class="modal-title">
                            <i class="bi bi-pencil-square me-2"></i>Self-Annotation
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row">
                            <!-- Left: Canvas -->
                            <div class="col-lg-9">
                                <div class="card">
                                    <div class="card-header">
                                        <div class="d-flex justify-content-between align-items-center">
                                            <span id="image-info">Image 1 / 1</span>
                                            <div class="btn-group btn-group-sm">
                                                <button class="btn btn-outline-secondary" onclick="previousImage()" id="prev-btn">
                                                    <i class="bi bi-arrow-left"></i> Previous
                                                </button>
                                                <button class="btn btn-outline-secondary" onclick="nextImage()" id="next-btn">
                                                    Next <i class="bi bi-arrow-right"></i>
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="card-body p-0" style="background: #f0f0f0;">
                                        <div style="position: relative; overflow: auto; max-height: 800px;" id="canvas-container">
                                            <canvas id="annotation-canvas" style="cursor: crosshair; display: block; margin: 0 auto; max-width: 100%;"></canvas>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Right: Annotations List -->
                            <div class="col-lg-3">
                                <div class="card h-100">
                                    <div class="card-header">
                                        <h6 class="mb-0">
                                            <i class="bi bi-list-ul me-2"></i>Annotations
                                            (<span id="annotations-count">0</span>)
                                        </h6>
                                    </div>
                                    <div class="card-body" style="max-height: 500px; overflow-y: auto;">
                                        <div id="annotations-list">
                                            <div class="text-muted text-center py-4">
                                                <i class="bi bi-inbox display-4"></i>
                                                <p class="mt-2">No annotations yet.<br>Draw boxes on the image.</p>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="card-footer">
                                        <button class="btn btn-sm btn-outline-danger w-100" onclick="clearAllAnnotations()">
                                            <i class="bi bi-trash me-1"></i> Clear All
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-success" onclick="saveAllAnnotations()">
                            <i class="bi bi-check-circle me-1"></i> Save All Annotations
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('beforeend', modalHTML);

    // Add class input modal
    const classInputModalHTML = `
        <div class="modal fade" id="classInputModal" tabindex="-1" data-bs-backdrop="true" data-bs-keyboard="true">
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Enter Class Name</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <input type="text"
                               id="class-name-input"
                               class="form-control form-control-lg"
                               placeholder="e.g., person, car, dog"
                               autocomplete="off"
                               autofocus>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" id="confirm-class-btn">Add Annotation</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('beforeend', classInputModalHTML);

    // Setup canvas event listeners
    setupCanvasListeners();
}

// Load dataset images
async function loadDatasetImages(datasetId) {
    try {
        console.log(`[SelfAnnotation] Loading images for dataset ${datasetId}...`);
        const images = await apiService.getDatasetImages(datasetId);

        if (images && Array.isArray(images)) {
            selfAnnotationState.images = images;
            console.log(`[SelfAnnotation] Loaded ${selfAnnotationState.images.length} images`);
        } else {
            console.warn('[SelfAnnotation] No images found or invalid response');
            selfAnnotationState.images = [];
        }
    } catch (error) {
        console.error('[SelfAnnotation] Failed to load images:', error);
        showToast('Failed to load dataset images', 'error');
        selfAnnotationState.images = [];
    }
}

// Load image to canvas
async function loadImageToCanvas(index) {
    if (index < 0 || index >= selfAnnotationState.images.length) {
        console.warn('[SelfAnnotation] Invalid image index:', index);
        return;
    }

    selfAnnotationState.currentImageIndex = index;
    const imageData = selfAnnotationState.images[index];
    selfAnnotationState.currentImage = imageData;

    console.log(`[SelfAnnotation] Loading image ${index + 1}/${selfAnnotationState.images.length}`);
    console.log('[SelfAnnotation] Image data:', imageData);

    // Update image info
    const imageInfoElement = document.getElementById('image-info');
    if (imageInfoElement) {
        imageInfoElement.textContent = `Image ${index + 1} / ${selfAnnotationState.images.length}`;
    }

    // Enable/disable navigation buttons
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    if (prevBtn) prevBtn.disabled = (index === 0);
    if (nextBtn) nextBtn.disabled = (index === selfAnnotationState.images.length - 1);

    // Load image
    const canvas = document.getElementById('annotation-canvas');
    if (!canvas) {
        console.error('[SelfAnnotation] Canvas not found');
        return;
    }

    selfAnnotationState.canvas = canvas;
    selfAnnotationState.ctx = canvas.getContext('2d');

    // Load image from presigned URL
    try {
        console.log(`[SelfAnnotation] Fetching presigned URL for asset ${imageData.id}...`);
        const response = await apiService.get(`/datasets/assets/${imageData.id}/presigned-download`);
        console.log('[SelfAnnotation] Presigned URL response:', response);

        if (!response || !response.download_url) {
            console.error('[SelfAnnotation] No download URL in response:', response);
            showToast('No image URL available', 'error');
            return;
        }

        const imageUrl = response.download_url;
        console.log('[SelfAnnotation] Image URL:', imageUrl);

        // Create image element
        const img = new Image();
        // Remove crossOrigin to avoid CORS issues with S3
        // img.crossOrigin = 'anonymous';

        img.onload = function() {
            console.log(`[SelfAnnotation] Image loaded successfully: ${img.width}x${img.height}`);
            selfAnnotationState.img = img;

            // Calculate scale to fit canvas in container while maintaining aspect ratio
            const container = document.getElementById('canvas-container');
            const maxWidth = container ? container.clientWidth : 1200;
            const maxHeight = 800;

            let displayWidth = img.width;
            let displayHeight = img.height;
            let scale = 1;

            // Scale down if image is too large
            if (img.width > maxWidth || img.height > maxHeight) {
                const scaleX = maxWidth / img.width;
                const scaleY = maxHeight / img.height;
                scale = Math.min(scaleX, scaleY);
                displayWidth = img.width * scale;
                displayHeight = img.height * scale;
            }

            // Set canvas to display size
            canvas.width = displayWidth;
            canvas.height = displayHeight;

            // Store scale and original dimensions for coordinate conversion
            selfAnnotationState.scale = scale;
            selfAnnotationState.originalWidth = img.width;
            selfAnnotationState.originalHeight = img.height;
            selfAnnotationState.displayWidth = displayWidth;
            selfAnnotationState.displayHeight = displayHeight;

            // Draw image scaled to fit
            selfAnnotationState.ctx.clearRect(0, 0, canvas.width, canvas.height);
            selfAnnotationState.ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

            console.log(`[SelfAnnotation] Canvas size: ${displayWidth}x${displayHeight}, scale: ${scale}`);

            // Load existing annotations for this image
            loadImageAnnotations(imageData.id);
        };

        img.onerror = function(e) {
            console.error('[SelfAnnotation] Image load error:', e);
            console.error('[SelfAnnotation] Failed image URL:', imageUrl);
            showToast('Failed to load image from S3', 'error');
        };

        // Set image source
        img.src = imageUrl;

    } catch (error) {
        console.error('[SelfAnnotation] Failed to get presigned URL:', error);
        console.error('[SelfAnnotation] Error details:', error.message, error.stack);
        showToast('Failed to load image URL: ' + error.message, 'error');
    }
}

// Load existing annotations for current image
async function loadImageAnnotations(imageId) {
    try {
        console.log(`[SelfAnnotation] Loading annotations for image ${imageId}...`);

        // First check frontend storage
        if (selfAnnotationState.imageAnnotations.has(imageId)) {
            selfAnnotationState.annotations = selfAnnotationState.imageAnnotations.get(imageId);
            console.log(`[SelfAnnotation] Loaded ${selfAnnotationState.annotations.length} annotations from frontend storage`);
            redrawCanvas();
            updateAnnotationsList();
            return;
        }

        // If not in frontend storage, try to load from backend
        const annotations = await apiService.getImageAnnotations(imageId);

        // Use original dimensions and scale for coordinate conversion
        const originalWidth = selfAnnotationState.originalWidth || selfAnnotationState.img.width;
        const originalHeight = selfAnnotationState.originalHeight || selfAnnotationState.img.height;
        const scale = selfAnnotationState.scale || 1;

        selfAnnotationState.annotations = annotations.map(ann => {
            // Check if geometry and bbox exist
            if (!ann.geometry || !ann.geometry.bbox) {
                console.warn('[SelfAnnotation] Annotation missing geometry:', ann);
                return null;
            }

            // Convert normalized coordinates to original image coordinates
            const origX = ann.geometry.bbox.x_center * originalWidth - (ann.geometry.bbox.width * originalWidth / 2);
            const origY = ann.geometry.bbox.y_center * originalHeight - (ann.geometry.bbox.height * originalHeight / 2);
            const origWidth = ann.geometry.bbox.width * originalWidth;
            const origHeight = ann.geometry.bbox.height * originalHeight;

            // Convert to display coordinates
            return {
                id: ann.id,
                className: ann.label_class ? ann.label_class.display_name : 'Unknown',
                x: origX * scale,
                y: origY * scale,
                width: origWidth * scale,
                height: origHeight * scale,
                confidence: ann.confidence || 1.0,
                source: ann.source || 'human',
                saved: true
            };
        }).filter(ann => ann !== null);

        console.log(`[SelfAnnotation] Loaded ${selfAnnotationState.annotations.length} annotations from backend`);
        redrawCanvas();
        updateAnnotationsList();
    } catch (error) {
        console.error('[SelfAnnotation] Failed to load annotations:', error);
        selfAnnotationState.annotations = [];
        updateAnnotationsList();
    }
}

// Setup canvas mouse event listeners
function setupCanvasListeners() {
    // We'll add listeners after canvas is created
    setTimeout(() => {
        const canvas = document.getElementById('annotation-canvas');
        if (!canvas) return;

        canvas.addEventListener('mousedown', handleMouseDown);
        canvas.addEventListener('mousemove', handleMouseMove);
        canvas.addEventListener('mouseup', handleMouseUp);
        // Removed canvas click event - deletion only via trash button in annotations list
    }, 100);
}

function handleMouseDown(e) {
    const rect = selfAnnotationState.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Start new annotation
    selfAnnotationState.startX = mouseX;
    selfAnnotationState.startY = mouseY;
    selfAnnotationState.currentX = mouseX;
    selfAnnotationState.currentY = mouseY;
    selfAnnotationState.drawing = true;

    console.log(`[SelfAnnotation] Start drawing at (${mouseX}, ${mouseY})`);
}

function handleMouseMove(e) {
    if (!selfAnnotationState.drawing) return;

    const rect = selfAnnotationState.canvas.getBoundingClientRect();
    selfAnnotationState.currentX = e.clientX - rect.left;
    selfAnnotationState.currentY = e.clientY - rect.top;

    // Redraw image and existing annotations
    redrawCanvas();

    // Draw current box
    const ctx = selfAnnotationState.ctx;
    const x = Math.min(selfAnnotationState.startX, selfAnnotationState.currentX);
    const y = Math.min(selfAnnotationState.startY, selfAnnotationState.currentY);
    const width = Math.abs(selfAnnotationState.currentX - selfAnnotationState.startX);
    const height = Math.abs(selfAnnotationState.currentY - selfAnnotationState.startY);

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

function handleMouseUp(e) {
    if (!selfAnnotationState.drawing) return;

    const rect = selfAnnotationState.canvas.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;

    selfAnnotationState.drawing = false;

    // Calculate bbox
    const x = Math.min(selfAnnotationState.startX, endX);
    const y = Math.min(selfAnnotationState.startY, endY);
    const width = Math.abs(endX - selfAnnotationState.startX);
    const height = Math.abs(endY - selfAnnotationState.startY);

    // Ignore very small boxes
    if (width < 10 || height < 10) {
        console.log('[SelfAnnotation] Box too small, ignoring');
        redrawCanvas();
        return;
    }

    console.log(`[SelfAnnotation] Created box: (${x}, ${y}, ${width}, ${height})`);

    // Show class input dialog
    showClassInputDialog(x, y, width, height);
}

// Removed: Canvas click deletion feature
// Annotations can only be deleted via trash button in the annotations list

function showClassInputDialog(x, y, width, height) {
    console.log('[ClassInput] Opening Bootstrap modal for class input');

    const modalElement = document.getElementById('classInputModal');
    const inputElement = document.getElementById('class-name-input');
    const confirmBtn = document.getElementById('confirm-class-btn');

    if (!modalElement || !inputElement || !confirmBtn) {
        console.error('[ClassInput] Modal elements not found');
        return;
    }

    // Clear previous input
    inputElement.value = '';

    // Store box coordinates for later use
    window._pendingAnnotation = { x, y, width, height };

    // Get or create Bootstrap modal instance
    let modal = bootstrap.Modal.getInstance(modalElement);
    if (!modal) {
        modal = new bootstrap.Modal(modalElement, {
            backdrop: true,
            keyboard: true,
            focus: true
        });
    }

    // Remove and recreate confirm button to clear old listeners
    const newConfirmBtn = confirmBtn.cloneNode(true);
    confirmBtn.parentNode.replaceChild(newConfirmBtn, confirmBtn);

    // Remove and recreate input to clear old listeners
    const newInputElement = inputElement.cloneNode(true);
    inputElement.parentNode.replaceChild(newInputElement, inputElement);
    newInputElement.value = '';

    // Add event listener for confirm button (once)
    newConfirmBtn.addEventListener('click', () => {
        const className = newInputElement.value.trim();
        if (className) {
            confirmClassInput(window._pendingAnnotation.x, window._pendingAnnotation.y,
                            window._pendingAnnotation.width, window._pendingAnnotation.height);
        }
        modal.hide();
    }, { once: true });

    // Handle Enter key in input (once)
    newInputElement.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            const className = newInputElement.value.trim();
            if (className) {
                confirmClassInput(window._pendingAnnotation.x, window._pendingAnnotation.y,
                                window._pendingAnnotation.width, window._pendingAnnotation.height);
            }
            modal.hide();
        }
    });

    // Focus input when modal is shown (once)
    modalElement.addEventListener('shown.bs.modal', () => {
        console.log('[ClassInput] Modal shown, focusing input');
        setTimeout(() => {
            newInputElement.focus();
            newInputElement.select();
        }, 100);
    }, { once: true });

    // Clean up on modal hide (once)
    modalElement.addEventListener('hidden.bs.modal', () => {
        console.log('[ClassInput] Modal hidden');
        redrawCanvas();
        window._pendingAnnotation = null;
    }, { once: true });

    // Show the modal
    modal.show();
}

function confirmClassInput(x, y, width, height) {
    const input = document.getElementById('class-name-input');
    const className = input ? input.value.trim() : '';

    console.log('[ClassInput] Confirming with class name:', className);

    if (!className) {
        console.log('[SelfAnnotation] No class name entered, annotation cancelled');
        return;
    }

    // Add annotation with scale and original dimensions for later conversion
    const annotation = {
        id: null,
        className: className,
        x,
        y,
        width,
        height,
        scale: selfAnnotationState.scale,
        originalWidth: selfAnnotationState.originalWidth,
        originalHeight: selfAnnotationState.originalHeight,
        confidence: 1.0,
        source: 'human',
        saved: false
    };

    selfAnnotationState.annotations.push(annotation);

    // Update display
    redrawCanvas();
    updateAnnotationsList();

    console.log(`[SelfAnnotation] Added annotation:`, annotation);
    showToast(`Added annotation: ${className}`, 'success');
}

function deleteAnnotationByIndex(index) {
    if (index < 0 || index >= selfAnnotationState.annotations.length) return;

    const annotation = selfAnnotationState.annotations[index];
    console.log(`[SelfAnnotation] Deleting annotation ${index}:`, annotation);

    // If saved annotation, delete from backend
    if (annotation.saved && annotation.id) {
        apiService.deleteAnnotation(annotation.id)
            .then(() => {
                console.log(`[SelfAnnotation] Deleted annotation ${annotation.id} from backend`);
            })
            .catch(error => {
                console.error(`[SelfAnnotation] Failed to delete annotation:`, error);
                showToast('Failed to delete annotation from server', 'error');
            });
    }

    selfAnnotationState.annotations.splice(index, 1);
    redrawCanvas();
    updateAnnotationsList();
    showToast('Annotation deleted', 'info');
}

// Redraw canvas with image and all annotations
function redrawCanvas() {
    if (!selfAnnotationState.ctx || !selfAnnotationState.img) return;

    const ctx = selfAnnotationState.ctx;
    const img = selfAnnotationState.img;
    const displayWidth = selfAnnotationState.displayWidth || img.width;
    const displayHeight = selfAnnotationState.displayHeight || img.height;

    // Clear and draw image (scaled to fit)
    ctx.clearRect(0, 0, selfAnnotationState.canvas.width, selfAnnotationState.canvas.height);
    ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

    // Draw all annotations
    selfAnnotationState.annotations.forEach((ann, index) => {
        // Draw bbox with hover effect
        ctx.strokeStyle = ann.saved ? '#4caf50' : '#ff9800';
        ctx.lineWidth = 3;
        ctx.strokeRect(ann.x, ann.y, ann.width, ann.height);

        // Add subtle fill to make boxes more visible
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

// Update annotations list in sidebar
function updateAnnotationsList() {
    const listContainer = document.getElementById('annotations-list');
    const countElement = document.getElementById('annotations-count');

    countElement.textContent = selfAnnotationState.annotations.length;

    if (selfAnnotationState.annotations.length === 0) {
        listContainer.innerHTML = `
            <div class="text-muted text-center py-4">
                <i class="bi bi-inbox display-4"></i>
                <p class="mt-2">No annotations yet.<br>Draw boxes on the image.</p>
            </div>
        `;
        return;
    }

    let html = '<div class="list-group list-group-flush">';

    selfAnnotationState.annotations.forEach((ann, index) => {
        html += `
            <div class="list-group-item">
                <div class="d-flex justify-content-between align-items-start">
                    <div class="flex-grow-1">
                        <div class="d-flex align-items-center mb-1">
                            <span class="badge bg-primary me-2">${index + 1}</span>
                            <input type="text" class="form-control form-control-sm"
                                value="${ann.className}"
                                onchange="updateAnnotationClass(${index}, this.value)"
                                placeholder="Class name">
                        </div>
                        <div class="small text-muted">
                            Box: (${Math.round(ann.x)}, ${Math.round(ann.y)})
                            ${Math.round(ann.width)}×${Math.round(ann.height)}px
                        </div>
                    </div>
                    <button class="btn btn-sm btn-outline-danger" onclick="deleteAnnotation(${index})">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
        `;
    });

    html += '</div>';
    listContainer.innerHTML = html;
}

// Update annotation class name
function updateAnnotationClass(index, newClassName) {
    if (index < 0 || index >= selfAnnotationState.annotations.length) return;

    const annotation = selfAnnotationState.annotations[index];
    annotation.className = newClassName.trim();
    annotation.saved = false; // Mark as unsaved

    console.log(`[SelfAnnotation] Updated annotation ${index} class to "${newClassName}"`);
    redrawCanvas();
    updateAnnotationsList();
}

// Delete annotation
async function deleteAnnotation(index) {
    if (index < 0 || index >= selfAnnotationState.annotations.length) return;

    if (!confirm('Delete this annotation?')) return;

    const annotation = selfAnnotationState.annotations[index];
    const currentImage = selfAnnotationState.currentImage;
    console.log(`[SelfAnnotation] Deleting annotation ${index}:`, annotation);

    // If saved annotation, delete from backend
    if (annotation.saved && annotation.id) {
        try {
            await apiService.deleteAnnotation(annotation.id);
            console.log(`[SelfAnnotation] Deleted annotation ${annotation.id} from backend`);
        } catch (error) {
            console.error(`[SelfAnnotation] Failed to delete annotation:`, error);
            showToast('Failed to delete annotation from server', 'error');
            return; // Don't proceed if backend delete failed
        }
    }

    // Remove from state
    selfAnnotationState.annotations.splice(index, 1);

    // Update label file in S3
    if (currentImage) {
        try {
            const remainingAnnotations = selfAnnotationState.annotations.filter(ann =>
                ann.saved && ann.imageId === currentImage.id
            );

            if (remainingAnnotations.length === 0) {
                // No more annotations - delete label file from S3
                await apiService.deleteLabelFromS3(selfAnnotationState.datasetId, currentImage.filename);
                console.log(`[SelfAnnotation] Deleted label file for ${currentImage.filename}`);
            } else {
                // Re-upload updated label file with remaining annotations
                await uploadLabelsToS3();
                console.log(`[SelfAnnotation] Updated label file for ${currentImage.filename}`);
            }
        } catch (error) {
            console.error(`[SelfAnnotation] Failed to update label file in S3:`, error);
            showToast('Deleted annotation but failed to update labels in S3', 'warning');
        }
    }

    redrawCanvas();
    updateAnnotationsList();
}

// Clear all annotations
function clearAllAnnotations() {
    if (!confirm('Clear all annotations for this image?')) return;

    console.log(`[SelfAnnotation] Clearing all annotations`);
    selfAnnotationState.annotations = [];
    redrawCanvas();
    updateAnnotationsList();
}

// Save all annotations to backend
async function saveAllAnnotations() {
    // First save current image's annotations to frontend storage
    saveCurrentAnnotationsToFrontend();

    console.log('[SelfAnnotation] Saving all annotations to backend...');

    // Collect all annotations from all images
    let totalAnnotations = 0;
    for (const [imageId, annotations] of selfAnnotationState.imageAnnotations.entries()) {
        totalAnnotations += annotations.filter(ann => !ann.saved).length;
    }

    if (totalAnnotations === 0) {
        showToast('No annotations to save', 'warning');
        return;
    }

    try {
        let successCount = 0;
        let failCount = 0;

        // First, get or create label classes for all unique class names
        const uniqueClassNames = new Set();
        for (const [imageId, annotations] of selfAnnotationState.imageAnnotations.entries()) {
            annotations.forEach(ann => uniqueClassNames.add(ann.className));
        }

        // Create label classes map (className -> label_class_id)
        const labelClassMap = await createOrGetLabelClasses(Array.from(uniqueClassNames));

        // Process each image's annotations
        for (const [imageId, annotations] of selfAnnotationState.imageAnnotations.entries()) {
            const unsavedAnnotations = annotations.filter(ann => !ann.saved);

            if (unsavedAnnotations.length === 0) continue;

            // Find the image to get its dimensions
            const imageData = selfAnnotationState.images.find(img => img.id === imageId);
            if (!imageData) {
                console.error(`[SelfAnnotation] Image ${imageId} not found`);
                continue;
            }

            for (const ann of unsavedAnnotations) {
                try {
                    // Get label_class_id for this annotation
                    const labelClassId = labelClassMap.get(ann.className);
                    if (!labelClassId) {
                        console.error(`[SelfAnnotation] No label class ID for ${ann.className}`);
                        failCount++;
                        continue;
                    }

                    // Need to get the image's original dimensions
                    // For simplicity, we'll use the stored scale from when the image was loaded
                    // This is a simplification - ideally we'd store original dimensions per image
                    const scale = ann.scale || selfAnnotationState.scale || 1;
                    const originalWidth = ann.originalWidth || selfAnnotationState.originalWidth || imageData.width || 1000;
                    const originalHeight = ann.originalHeight || selfAnnotationState.originalHeight || imageData.height || 1000;

                    // Convert display coordinates back to original image coordinates
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
                        asset_id: imageId,
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
                    ann.saved = true;
                    successCount++;

                } catch (error) {
                    console.error(`[SelfAnnotation] Failed to save annotation:`, error);
                    failCount++;
                }
            }
        }

        console.log(`[SelfAnnotation] Save complete: ${successCount} success, ${failCount} failed`);

        if (successCount > 0) {
            // Upload label files to S3 for all images that had annotations saved
            const uploadedImages = new Set();
            try {
                for (const [imageId, annotations] of selfAnnotationState.imageAnnotations.entries()) {
                    const savedAnnotations = annotations.filter(ann => ann.saved);
                    if (savedAnnotations.length > 0) {
                        const imageData = selfAnnotationState.images.find(img => img.id === imageId);
                        if (imageData) {
                            await uploadLabelsForImage(imageData, savedAnnotations);
                            uploadedImages.add(imageId);
                        }
                    }
                }
                showToast(`Saved ${successCount} annotation(s) and uploaded ${uploadedImages.size} label file(s) to S3`, 'success');
            } catch (labelError) {
                console.error('[SelfAnnotation] Failed to upload labels to S3:', labelError);
                showToast(`Saved ${successCount} annotation(s) but failed to upload some labels to S3`, 'warning');
            }

            // After all annotations are saved and label files are uploaded, regenerate data.yaml
            try {
                const datasetId = selfAnnotationState.datasetId;
                await apiService.post(`/datasets/${datasetId}/upload-data-yaml`);
                console.log(`[SelfAnnotation] data.yaml regenerated successfully for dataset ${datasetId}`);
            } catch (yamlError) {
                console.error('[SelfAnnotation] Failed to regenerate data.yaml:', yamlError);
            }

            // Update current image display
            redrawCanvas();
            updateAnnotationsList();

            // Update dataset information in the parent page
            // Check both dataset-detail page and datasets list page
            const detailPage = window.currentDatasetDetailPage;
            const datasetsPage = window.currentDatasetsPage;
            
            console.log('[SelfAnnotation] Checking for page instances...');
            console.log('[SelfAnnotation] Dataset detail page:', !!detailPage);
            console.log('[SelfAnnotation] Datasets list page:', !!datasetsPage);
            
            if (detailPage && detailPage.updateDatasetInfo) {
                console.log('[SelfAnnotation] Calling updateDatasetInfo on detail page...');
                await detailPage.updateDatasetInfo();
                console.log('[SelfAnnotation] Detail page update completed');
            } else if (datasetsPage && datasetsPage.updateDataInfo) {
                console.log('[SelfAnnotation] Calling updateDataInfo on datasets page...');
                await datasetsPage.updateDataInfo();
                console.log('[SelfAnnotation] Datasets page update completed');
            } else {
                console.warn('[SelfAnnotation] Cannot update dataset info - no page instance found');
            }
        }

        if (failCount > 0) {
            showToast(`Failed to save ${failCount} annotation(s)`, 'error');
        }

    } catch (error) {
        console.error('[SelfAnnotation] Save error:', error);
        showToast('Failed to save annotations', 'error');
    }
}

// Upload labels to S3 for a specific image with given annotations
async function uploadLabelsForImage(imageData, annotations) {
    try {
        if (!imageData || !annotations || annotations.length === 0) {
            console.warn('[SelfAnnotation] Invalid image or annotations for label upload');
            return;
        }

        console.log(`[SelfAnnotation] Uploading labels for ${imageData.filename} (${annotations.length} annotations)`);

        // Get label classes mapping
        const labelClasses = await apiService.get(`/datasets/${selfAnnotationState.datasetId}/label-classes`);
        const labelClassesMap = new Map();

        if (labelClasses && Array.isArray(labelClasses)) {
            labelClasses.forEach(cls => {
                labelClassesMap.set(cls.display_name, { id: cls.id, name: cls.display_name, yolo_index: cls.yolo_index });
            });
        }

        // Convert display coordinates to normalized coordinates
        // Use stored dimensions from when annotations were created
        const normalizedAnnotations = annotations.map(ann => {
            // Get original dimensions (stored when annotation was created)
            const originalWidth = ann.originalWidth || imageData.width || 1000;
            const originalHeight = ann.originalHeight || imageData.height || 1000;
            const scale = ann.scale || 1;

            // Convert from display coordinates back to original image coordinates
            const origX = ann.x / scale;
            const origY = ann.y / scale;
            const origWidth = ann.width / scale;
            const origHeight = ann.height / scale;

            // Normalize to 0-1 range
            const normalizedWidth = origWidth / originalWidth;
            const normalizedHeight = origHeight / originalHeight;
            const normalizedX = origX / originalWidth;
            const normalizedY = origY / originalHeight;
            return {
                className: ann.className,
                x_center: normalizedX + (normalizedWidth / 2),
                y_center: normalizedY + (normalizedHeight / 2),
                width: normalizedWidth,
                height: normalizedHeight
            };
        });

        console.log(`[SelfAnnotation] Normalized annotations:`, normalizedAnnotations);

        // Convert annotations to YOLO format
        const labelContent = convertToYOLO(normalizedAnnotations, labelClassesMap);

        if (!labelContent) {
            console.warn('[SelfAnnotation] No valid label content generated');
            return;
        }

        console.log(`[SelfAnnotation] Label content generated (${labelContent.split('\n').length} lines):`, labelContent);

        // Upload to S3
        await apiService.uploadLabel(
            selfAnnotationState.datasetId,
            imageData.filename,
            labelContent
        );

        console.log(`[SelfAnnotation] Successfully uploaded labels for ${imageData.filename}`);

    } catch (error) {
        console.error('[SelfAnnotation] Failed to upload labels to S3:', error);
        throw error;
    }
}

// Upload labels to S3 for current image (called after individual annotation edits)
async function uploadLabelsToS3() {
    const currentImage = selfAnnotationState.currentImage;
    if (!currentImage) {
        console.warn('[SelfAnnotation] No current image for label upload');
        return;
    }

    const annotations = selfAnnotationState.annotations.filter(ann => ann.saved);
    if (annotations.length === 0) {
        console.log('[SelfAnnotation] No saved annotations to upload for current image');
        return;
    }

    await uploadLabelsForImage(currentImage, annotations);
}

// Create or get label classes for the given class names
async function createOrGetLabelClasses(classNames) {
    const labelClassMap = new Map();

    try {
        // Get existing label classes for this dataset
        const datasetId = selfAnnotationState.datasetId;
        const existingClasses = await apiService.get(`/datasets/${datasetId}/label-classes`);

        // Map existing classes
        if (existingClasses && Array.isArray(existingClasses)) {
            existingClasses.forEach(cls => {
                labelClassMap.set(cls.display_name, cls.id);
            });
        }

        // Create missing classes
        for (const className of classNames) {
            if (!labelClassMap.has(className)) {
                try {
                    const newClass = await apiService.post(`/datasets/${datasetId}/label-classes`, {
                        display_name: className,
                        color: getRandomColor()
                    });
                    labelClassMap.set(className, newClass.id);
                    console.log(`[SelfAnnotation] Created label class: ${className} (id: ${newClass.id})`);
                } catch (error) {
                    console.error(`[SelfAnnotation] Failed to create label class ${className}:`, error);
                }
            }
        }

    } catch (error) {
        console.error('[SelfAnnotation] Error managing label classes:', error);
    }

    return labelClassMap;
}

// Generate random color for label class
function getRandomColor() {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
}

// Navigation functions
async function previousImage() {
    if (selfAnnotationState.currentImageIndex > 0) {
        // Save current annotations to frontend storage
        saveCurrentAnnotationsToFrontend();
        await loadImageToCanvas(selfAnnotationState.currentImageIndex - 1);
    }
}

async function nextImage() {
    if (selfAnnotationState.currentImageIndex < selfAnnotationState.images.length - 1) {
        // Save current annotations to frontend storage
        saveCurrentAnnotationsToFrontend();
        await loadImageToCanvas(selfAnnotationState.currentImageIndex + 1);
    }
}

// Save current image's annotations to frontend storage (not backend)
function saveCurrentAnnotationsToFrontend() {
    if (selfAnnotationState.currentImage) {
        const imageId = selfAnnotationState.currentImage.id;
        // Store a copy of annotations for this image, ensuring imageId is set
        const annotationsWithImageId = selfAnnotationState.annotations.map(ann => ({
            ...ann,
            imageId: imageId
        }));
        selfAnnotationState.imageAnnotations.set(imageId, annotationsWithImageId);
        console.log(`[SelfAnnotation] Saved ${annotationsWithImageId.length} annotations for image ${imageId} to frontend`);
    }
}

// Auto-save annotations for the current image using yolo_index instead of class_id
async function autoSaveCurrentAnnotations() {
    // Filter annotations that are not saved yet
    const unsavedAnnotations = selfAnnotationState.annotations.filter(ann => !ann.saved);
    if (unsavedAnnotations.length === 0) {
        console.log('[SelfAnnotation] No unsaved annotations to auto-save');
        return;
    }

    console.log(`[SelfAnnotation] Auto-saving ${unsavedAnnotations.length} annotations...`);

    try {
        // Fetch all label classes to map display_name → { id, yolo_index }
        const datasetId = selfAnnotationState.datasetId;
        const labelClasses = await apiService.get(`/datasets/${datasetId}/label-classes`);
        const labelMap = new Map();
        labelClasses.forEach(cls => {
            labelMap.set(cls.display_name, {
                id: cls.id,
                yolo_index: cls.yolo_index
            });
        });

        // Iterate over unsaved annotations and upload them
        for (const ann of unsavedAnnotations) {
            try {
                // Use original image dimensions for normalization
                const originalWidth = selfAnnotationState.originalWidth || selfAnnotationState.img.width;
                const originalHeight = selfAnnotationState.originalHeight || selfAnnotationState.img.height;
                const scale = selfAnnotationState.scale || 1;

                // Convert displayed coordinates to original coordinates
                const origX = ann.x / scale;
                const origY = ann.y / scale;
                const origWidth = ann.width / scale;
                const origHeight = ann.height / scale;

                // Normalize bbox coordinates (0~1)
                const normalizedGeometry = {
                    bbox: {
                        x_center: (origX + origWidth / 2) / originalWidth,
                        y_center: (origY + origHeight / 2) / originalHeight,
                        width: origWidth / originalWidth,
                        height: origHeight / originalHeight
                    }
                };

                // Retrieve yolo_index and label_class_id for this annotation
                const classInfo = labelMap.get(ann.className);
                if (!classInfo || classInfo.yolo_index === undefined || classInfo.yolo_index === null) {
                    console.warn(`[AutoSave] Missing yolo_index for class ${ann.className}`);
                    continue;
                }

                // Prepare annotation payload
                const annotationData = {
                    asset_id: selfAnnotationState.currentImage.id,
                    label_class_id: classInfo.id,      // keep id for backend reference
                    yolo_index: classInfo.yolo_index,  // use yolo_index for YOLO training
                    geometry_type: 'bbox',
                    geometry: normalizedGeometry,
                    is_normalized: true,
                    source: 'human',
                    confidence: 1.0,
                    annotator_name: 'user'
                };

                // Send annotation to backend
                const response = await apiService.createAnnotation(annotationData);

                // Mark annotation as saved
                ann.id = response.id;
                ann.saved = true;
            } catch (error) {
                console.error('[SelfAnnotation] Failed to auto-save annotation:', error);
            }
        }

        console.log('[SelfAnnotation] Auto-save complete');
        
        // Update dataset information if any annotations were saved
        if (unsavedAnnotations.length > 0) {
            // Check both dataset-detail page and datasets list page
            const detailPage = window.currentDatasetDetailPage;
            const datasetsPage = window.currentDatasetsPage;
            
            console.log('[AutoSave] Checking for page instances...');
            console.log('[AutoSave] Dataset detail page:', !!detailPage);
            console.log('[AutoSave] Datasets list page:', !!datasetsPage);
            
            if (detailPage && detailPage.updateDatasetInfo) {
                console.log('[AutoSave] Calling updateDatasetInfo on detail page...');
                await detailPage.updateDatasetInfo();
                console.log('[AutoSave] Detail page update completed');
            } else if (datasetsPage && datasetsPage.updateDataInfo) {
                console.log('[AutoSave] Calling updateDataInfo on datasets page...');
                await datasetsPage.updateDataInfo();
                console.log('[AutoSave] Datasets page update completed');
            } else {
                console.warn('[AutoSave] Cannot update dataset info - no page instance found');
            }
        }
    } catch (error) {
        console.error('[SelfAnnotation] Auto-save error:', error);
    }
}

