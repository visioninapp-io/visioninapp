// LLM Modal Component - AI Model Training Modal
// Converted from React/TypeScript to vanilla JavaScript

let llmModalState = {
    currentStep: 1, // 1: query input, 2: training in progress, 3: completion
    query: "",
    selectedDatasetId: null,
    datasets: [],
    timers: []
};

// Show LLM Modal
async function showLLMModal() {
    // Create modal if doesn't exist
    if (!document.getElementById('llmModal')) {
        createLLMModal();
    }

    // Reset state
    llmModalState = {
        currentStep: 1,
        query: "",
        selectedDatasetId: null,
        datasets: [],
        timers: []
    };

    // Show modal first
    const modal = new bootstrap.Modal(document.getElementById('llmModal'));
    modal.show();

    // Render step first (빈 select로)
    renderStep(llmModalState.currentStep);

    // Then load datasets and update select
    await loadDatasetsForLLMModal();
    // 데이터셋 로드 후 다시 업데이트
    updateDatasetSelect();
}

// Create modal HTML
function createLLMModal() {
    const modalHTML = `
        <div class="modal fade" id="llmModal" tabindex="-1" data-bs-backdrop="static" data-bs-keyboard="false">
            <div class="modal-dialog modal-dialog-centered modal-lg modal-fullscreen-md-down">
                <div class="modal-content">
                    <!-- Progress Bar -->
                    <div class="progress" style="height: 4px;">
                        <div class="progress-bar bg-primary" role="progressbar" id="llm-progress-bar" style="width: 33%"></div>
                    </div>

                    <!-- Header -->
                    <div class="modal-header">
                        <div class="d-flex align-items-center gap-2">
                            <span class="badge bg-primary" id="llm-step-badge">Step 1/3</span>
                            <span class="text-muted" id="llm-step-label">Query Input</span>
                        </div>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" onclick="closeLLMModal()"></button>
                    </div>

                    <!-- Body -->
                    <div class="modal-body" id="llm-modal-body" style="min-height: 480px;">
                        <!-- Content will be rendered here -->
                    </div>

                    <!-- Footer -->
                    <div class="modal-footer bg-light border-top">
                        <div class="d-flex gap-1" id="llm-step-indicators">
                            <div class="rounded-circle bg-primary" style="width: 8px; height: 8px;"></div>
                            <div class="rounded-circle bg-secondary" style="width: 8px; height: 8px;"></div>
                            <div class="rounded-circle bg-secondary" style="width: 8px; height: 8px;"></div>
                        </div>
                        <small class="text-muted" id="llm-footer-text">Enter query and select dataset</small>
                    </div>
                </div>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('beforeend', modalHTML);
}

// Render step content
function renderStep(step) {
    const body = document.getElementById('llm-modal-body');
    if (!body) return;

    // Clear timers
    llmModalState.timers.forEach(timer => clearTimeout(timer));
    llmModalState.timers = [];

    // Update progress bar
    const progressBar = document.getElementById('llm-progress-bar');
    if (progressBar) {
        progressBar.style.width = `${(step / 3) * 100}%`;
    }

    // Update step badge
    const stepBadge = document.getElementById('llm-step-badge');
    if (stepBadge) {
        stepBadge.textContent = `Step ${step}/3`;
    }

    // Update step label
    const stepLabel = document.getElementById('llm-step-label');
    if (stepLabel) {
        const labels = {
            1: "Query Input",
            2: "Training in Progress",
            3: "Complete"
        };
        stepLabel.textContent = labels[step];
    }

    // Update step indicators
    updateStepIndicators(step);

    // Update footer text
    const footerText = document.getElementById('llm-footer-text');
    if (footerText) {
        const texts = {
            1: "Enter query and select dataset",
            2: "Model training is in progress",
            3: "Training completed"
        };
        footerText.textContent = texts[step];
    }

    // Render step content
    switch(step) {
        case 1:
            renderQueryInputStep(body);
            break;
        case 2:
            renderTrainingProgressStep(body);
            startTrainingProgress();
            break;
        case 3:
            renderCompletionStep(body);
            break;
    }
}

// Update step indicators
function updateStepIndicators(currentStep) {
    const indicators = document.getElementById('llm-step-indicators');
    if (!indicators) return;

    indicators.innerHTML = '';
    for (let i = 1; i <= 3; i++) {
        const indicator = document.createElement('div');
        indicator.className = `rounded-circle ${i <= currentStep ? 'bg-primary' : 'bg-secondary'}`;
        indicator.style.width = '8px';
        indicator.style.height = '8px';
        indicators.appendChild(indicator);
    }
}

// Step 1: Query Input
function renderQueryInputStep(container) {
    container.innerHTML = `
        <div class="space-y-4">
            <div>
                <h3 class="fw-bold mb-2">AI Model Training</h3>
                <p class="text-muted">Enter a training query and select a dataset.</p>
            </div>

            <div class="mb-4">
                <label class="form-label fw-semibold">User Query</label>
                <textarea 
                       class="form-control form-control-lg" 
                       id="llm-query-input"
                       rows="6"
                       placeholder="ex) Use the YOLOv12 model and set the number of epochs to 100. Apply data augmentation during training. Use learning rate of 0.001 and batch size of 32."
                       style="resize: vertical; min-height: 120px;">${llmModalState.query}</textarea>
                <small class="text-muted">Enter detailed training instructions or model configuration</small>
            </div>

            <!-- Dataset Selection Field -->
            <div class="mb-4">
                <label class="form-label fw-semibold">Select Dataset</label>
                <select class="form-select form-select-lg" 
                        id="llm-dataset-select"
                        onchange="onDatasetChange(this.value)">
                    <option value="">-- Select Dataset --</option>
                </select>
                <small class="text-muted">Select a dataset for training</small>
            </div>

            <button class="btn btn-primary w-100 py-3" 
                    onclick="submitQuery()"
                    id="llm-submit-btn">
                Start Training →
            </button>
        </div>
    `;

    // Save input value in real-time
    const input = document.getElementById('llm-query-input');
    if (input) {
        // Ctrl+Enter to submit
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                submitQuery();
            }
        });
        
        // Save input value in real-time
        input.addEventListener('input', (e) => {
            llmModalState.query = e.target.value;
        });
    }

    // 렌더링 후 데이터셋 옵션 업데이트 (데이터셋이 이미 로드된 경우에만)
    // 데이터셋이 아직 로드되지 않았으면 showLLMModal에서 나중에 업데이트됨
    if (llmModalState.datasets && llmModalState.datasets.length > 0) {
        updateDatasetSelect();
    }
}

// 데이터셋 로드 함수
async function loadDatasetsForLLMModal() {
    try {
        llmModalState.datasets = await apiService.getDatasets();
        console.log('[LLM Modal] Loaded datasets:', llmModalState.datasets.length);
        
        // 데이터셋이 있으면 첫 번째 선택
        if (llmModalState.datasets.length > 0 && !llmModalState.selectedDatasetId) {
            llmModalState.selectedDatasetId = llmModalState.datasets[0].id;
            console.log('[LLM Modal] Auto-selected first dataset:', llmModalState.selectedDatasetId);
        }
    } catch (error) {
        console.error('[LLM Modal] Error loading datasets:', error);
        llmModalState.datasets = [];
    }
}


// 데이터셋 셀렉트 박스 업데이트 (다른 페이지들과 동일한 패턴)
function updateDatasetSelect() {
    const select = document.getElementById('llm-dataset-select');
    if (!select) {
        console.log('[LLM Modal] Select element not found');
        return;
    }
    
    // 기존 옵션 제거 (첫 번째 "-- Select Dataset --" 제외)
    while (select.children.length > 1) {
        select.removeChild(select.lastChild);
    }
    
    // 데이터셋이 없으면 리턴
    if (!llmModalState.datasets || llmModalState.datasets.length === 0) {
        console.log('[LLM Modal] No datasets available');
        return;
    }
    
    // 데이터셋 옵션 추가
    llmModalState.datasets.forEach(dataset => {
        const option = document.createElement('option');
        option.value = String(dataset.id);  // 문자열로 변환
        option.textContent = `${dataset.name} (${dataset.total_assets || dataset.total_images || 0} images)`;
        // 타입을 명확히 비교
        if (String(dataset.id) === String(llmModalState.selectedDatasetId)) {
            option.selected = true;
        }
        select.appendChild(option);
    });
    
    console.log('[LLM Modal] Updated dataset select with', llmModalState.datasets.length, 'datasets');
}

// 데이터셋 선택 변경 핸들러
function onDatasetChange(datasetId) {
    llmModalState.selectedDatasetId = datasetId ? parseInt(datasetId) : null;
    console.log('[LLM Modal] Selected dataset:', llmModalState.selectedDatasetId);
}

function submitQuery() {
    const input = document.getElementById('llm-query-input');
    if (!input || !input.value.trim()) {
        showToast('Please enter a query', 'warning');
        return;
    }

    // Dataset selection is required
    if (!llmModalState.selectedDatasetId) {
        showToast('Please select a dataset', 'warning');
        return;
    }

    llmModalState.query = input.value.trim();
    llmModalState.currentStep = 2;
    renderStep(2);
}

// Step 2: Training Progress
function renderTrainingProgressStep(container) {
    container.innerHTML = `
        <div class="text-center">
            <h3 class="fw-bold mb-4">Model Training in Progress</h3>
            <p class="text-muted mb-5">The model is processing training data. Please wait...</p>

            <!-- Simple Loading Animation -->
            <div class="d-flex justify-content-center mb-5">
                <div class="d-flex flex-column align-items-center">
                    <div class="spinner-border text-primary mb-3" role="status" style="width: 4rem; height: 4rem;">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <div class="d-flex align-items-center gap-2">
                        <span class="fs-5 fw-semibold text-primary">Training in Progress</span>
                    </div>
                </div>
            </div>

            <!-- Status Messages -->
            <div class="row g-3 mb-4">
                <div class="col-6">
                    <div class="p-3 border rounded" id="status-1">
                        <div class="d-flex align-items-center gap-2">
                            <div class="spinner-border spinner-border-sm text-primary" role="status"></div>
                            <span class="small">Training</span>
                        </div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="p-3 border rounded" id="status-2">
                        <div class="d-flex align-items-center gap-2">
                            <div class="rounded-circle bg-secondary" style="width: 8px; height: 8px;"></div>
                            <span class="small">Data Loading</span>
                        </div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="p-3 border rounded" id="status-3">
                        <div class="d-flex align-items-center gap-2">
                            <div class="rounded-circle bg-secondary" style="width: 8px; height: 8px;"></div>
                            <span class="small">Model Compilation</span>
                        </div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="p-3 border rounded" id="status-4">
                        <div class="d-flex align-items-center gap-2">
                            <div class="rounded-circle bg-secondary" style="width: 8px; height: 8px;"></div>
                            <span class="small">Final Validation</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function startTrainingProgress() {
    // Simulate training progress with status updates
    let currentStatus = 0;
    const statuses = [
        { id: 'status-1', label: 'Training', delay: 2000 },
        { id: 'status-2', label: 'Data Loading', delay: 3000 },
        { id: 'status-3', label: 'Model Compilation', delay: 4000 },
        { id: 'status-4', label: 'Final Validation', delay: 3000 }
    ];

    const updateStatus = (index) => {
        if (llmModalState.currentStep !== 2) return;
        
        if (index < statuses.length) {
            const status = statuses[index];
            const element = document.getElementById(status.id);
            if (element) {
                element.className = 'p-3 border border-success rounded bg-success bg-opacity-10';
                const dot = element.querySelector('.rounded-circle, .spinner-border');
                if (dot) {
                    dot.className = 'rounded-circle bg-success';
                    dot.style.width = '8px';
                    dot.style.height = '8px';
                }
                const text = element.querySelector('span');
                if (text) {
                    text.className = 'small text-success';
                    // Update spinner to checkmark if it was a spinner
                    const spinner = element.querySelector('.spinner-border');
                    if (spinner) {
                        spinner.remove();
                        const checkmark = document.createElement('div');
                        checkmark.className = 'rounded-circle bg-success';
                        checkmark.style.width = '8px';
                        checkmark.style.height = '8px';
                        element.querySelector('.d-flex').insertBefore(checkmark, text);
                    }
                }
            }

            const timer = setTimeout(() => {
                updateStatus(index + 1);
            }, status.delay);
            llmModalState.timers.push(timer);
        } else {
            // All statuses complete, move to completion step
            const timer = setTimeout(() => {
                if (llmModalState.currentStep === 2) {
                    llmModalState.currentStep = 3;
                    renderStep(3);
                }
            }, 1000);
            llmModalState.timers.push(timer);
        }
    };

    updateStatus(0);
}

// Step 3: Completion
function renderCompletionStep(container) {
    container.innerHTML = `
        <div class="text-center">
            <div class="mb-4">
                <span class="display-1 text-success">✓</span>
            </div>
            <h3 class="fw-bold mb-2">Training Complete!</h3>
            <p class="text-muted mb-5">Model training has been successfully completed.</p>

            <!-- Action Button -->
            <div class="d-grid gap-2">
                <button class="btn btn-primary btn-lg" onclick="resetLLMModal()">
                    🔄 New Model Training
                </button>
            </div>
        </div>
    `;
}

function resetLLMModal() {
    // Clear all timers
    llmModalState.timers.forEach(timer => clearTimeout(timer));
    llmModalState.timers = [];
    
    llmModalState.currentStep = 1;
    llmModalState.query = "";
    llmModalState.selectedDatasetId = null;
    renderStep(1);
    // Update dataset select after reset
    updateDatasetSelect();
}

function closeLLMModal() {
    // Clear all timers
    llmModalState.timers.forEach(timer => clearTimeout(timer));
    llmModalState.timers = [];
    resetLLMModal();
}

// Make functions globally available
window.showLLMModal = showLLMModal;
window.closeLLMModal = closeLLMModal;
window.submitQuery = submitQuery;
window.resetLLMModal = resetLLMModal;
window.onDatasetChange = onDatasetChange;

