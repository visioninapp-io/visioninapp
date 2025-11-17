// LLM Modal Component - AI Model Training Modal
// Converted from React/TypeScript to vanilla JavaScript

let llmModalState = {
    currentStep: 1, // 1: query input, 2: training in progress, 3: completion
    query: "",
    selectedDatasetId: null,
    datasets: [],
    timers: [],
    trainingJob: null,  // 학습 작업 정보 저장
    jobId: null,        // RabbitMQ 구독을 위한 job_id
    rabbitmqSubscriptions: [],  // 구독 관리
    needsConversion: false,  // 모델 변환 필요 여부
    conversionType: null,  // 'onnx' 또는 'tensorrt'
    hyperparameters: null  // 하이퍼파라미터 정보 (train.hpo 메시지에서 받음)
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
        timers: [],
        trainingJob: null,
        jobId: null,
        rabbitmqSubscriptions: [],
        needsConversion: false,
        conversionType: null,
        hyperparameters: null
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
        const response = await apiService.getDatasets();
        llmModalState.datasets = Array.isArray(response) ? response : (response.datasets || response.data || []);
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

async function submitQuery() {
    console.log('[LLM Modal] submitQuery() called');
    console.log('[LLM Modal] Current state:', {
        query: llmModalState.query,
        selectedDatasetId: llmModalState.selectedDatasetId,
        datasetsCount: llmModalState.datasets.length
    });
    
    const input = document.getElementById('llm-query-input');
    if (!input || !input.value.trim()) {
        console.warn('[LLM Modal] Query input is empty');
        showToast('Please enter a query', 'warning');
        return;
    }

    // Dataset selection is required
    if (!llmModalState.selectedDatasetId) {
        console.warn('[LLM Modal] No dataset selected');
        showToast('Please select a dataset', 'warning');
        return;
    }
    
    console.log('[LLM Modal] Validation passed, proceeding with API call...');

    llmModalState.query = input.value.trim();
    
    // Submit button 비활성화
    const submitBtn = document.getElementById('llm-submit-btn');
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Starting...';
    }

    try {
        // 데이터셋 정보 가져오기
        const dataset = llmModalState.datasets.find(d => d.id === llmModalState.selectedDatasetId);
        if (!dataset) {
            showToast('Dataset not found', 'error');
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.innerHTML = 'Start Training →';
            }
            return;
        }

        // S3 prefix 생성 (데이터셋에 s3_prefix가 있으면 사용, 없으면 기본값)
        const s3Prefix = dataset.s3_prefix || dataset.s3Prefix || `datasets/${dataset.name}/`;
        
        console.log('[LLM Modal] Starting training with:', {
            query: llmModalState.query,
            dataset: dataset.name,
            s3Prefix: s3Prefix
        });

        // apiService 확인
        if (!apiService) {
            console.error('[LLM Modal] apiService is not defined!');
            showToast('API Service is not available', 'error');
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.innerHTML = 'Start Training →';
            }
            return;
        }

        if (typeof apiService.createLLMTraining !== 'function') {
            console.error('[LLM Modal] apiService.createLLMTraining is not a function!');
            showToast('LLM training API is not available', 'error');
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.innerHTML = 'Start Training →';
            }
            return;
        }

        console.log('[LLM Modal] Calling apiService.createLLMTraining...');
        
        // LLM 학습 요청
        const response = await apiService.createLLMTraining({
            user_query: llmModalState.query,
            dataset_name: dataset.name,
            dataset_s3_prefix: s3Prefix,
            run_name: `llm_${dataset.name}_${Date.now()}`
        });
        
        console.log('[LLM Modal] API call completed, response:', response);

        console.log('[LLM Modal] Training started successfully:', response);
        showToast('Training started successfully!', 'success');
        
        // 학습 작업 정보 저장
        llmModalState.trainingJob = response;
        
        // job_id 추출 (응답 구조에 따라 조정)
        // external_job_id는 hyperparameters에 저장될 수 있음
        llmModalState.jobId = response.external_job_id 
            || (response.hyperparameters && response.hyperparameters.external_job_id)
            || response.job_id 
            || response.id;
        console.log('[LLM Modal] Job ID:', llmModalState.jobId);
        console.log('[LLM Modal] Full response:', response);
        console.log('[LLM Modal] Hyperparameters:', response.hyperparameters);
        
        // Step 2로 이동
        llmModalState.currentStep = 2;
        renderStep(2);
        
    } catch (error) {
        console.error('[LLM Modal] Training error:', error);
        console.error('[LLM Modal] Error details:', {
            message: error.message,
            detail: error.detail,
            stack: error.stack,
            fullError: error
        });
        
        const errorMessage = error.message || error.detail || error.toString();
        showToast(`Training failed: ${errorMessage}`, 'error');
        
        // Submit button 다시 활성화
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = 'Start Training →';
        }
        
        // Step 2로 이동하지 않도록 return
        return;
    }
}

// Step 2: Training Progress
function renderTrainingProgressStep(container) {
    container.innerHTML = `
        <div class="text-center">
            <h3 class="fw-bold mb-4">Model Training in Progress</h3>
            <p class="text-muted mb-4">The model is processing training data. Please wait...</p>

            <!-- Progress Bar with Percentage - 더 예쁘게 배치 -->
            <div class="mb-5">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-muted small fw-medium">Progress</span>
                    <div class="d-flex align-items-center gap-2">
                        <button class="btn btn-sm" 
                                id="hyperparameter-btn"
                                disabled
                                onclick="showHyperparameterModalFromLLM()"
                                title="View Hyperparameters"
                                style="opacity: 0.5; cursor: not-allowed;">
                            <i class="bi bi-sliders"></i> Hyperparameters
                        </button>
                        <span class="fw-bold text-primary" id="training-progress-text" style="font-size: 1.1rem;">0%</span>
                    </div>
                </div>
                <div class="progress mb-3" style="height: 32px; border-radius: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div class="progress-bar progress-bar-striped progress-bar-animated bg-primary" 
                         role="progressbar" 
                         id="training-progress-bar"
                         style="width: 0%; border-radius: 16px;"
                         aria-valuenow="0" 
                         aria-valuemin="0" 
                         aria-valuemax="100">
                    </div>
                </div>
                <div class="d-flex justify-content-center">
                    <span class="badge bg-light text-dark px-3 py-2 fw-normal border" id="training-progress-message" style="font-size: 0.9rem;">
                        Analyzing your prompt...
                    </span>
                </div>
            </div>

            <!-- Status Boxes - 균형 있는 배치 -->
            <div class="row g-3 mb-4">
                <div class="col-12 col-md-4">
                    <div class="p-3 border rounded h-100 d-flex align-items-center" id="status-analyze" style="min-height: 60px;">
                        <div class="d-flex align-items-center gap-2 w-100">
                            <div class="rounded-circle bg-secondary" style="width: 10px; height: 10px; flex-shrink: 0;"></div>
                            <span class="small fw-medium flex-grow-1" style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Analyze Prompt</span>
                        </div>
                    </div>
                </div>
                <div class="col-12 col-md-4">
                    <div class="p-3 border rounded h-100 d-flex align-items-center" id="status-download" style="min-height: 60px;">
                        <div class="d-flex align-items-center gap-2 w-100">
                            <div class="rounded-circle bg-secondary" style="width: 10px; height: 10px; flex-shrink: 0;"></div>
                            <span class="small fw-medium flex-grow-1" style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Downloading Dataset</span>
                        </div>
                    </div>
                </div>
                <div class="col-12 col-md-4">
                    <div class="p-3 border rounded h-100 d-flex align-items-center" id="status-prepare" style="min-height: 60px;">
                        <div class="d-flex align-items-center gap-2 w-100">
                            <div class="rounded-circle bg-secondary" style="width: 10px; height: 10px; flex-shrink: 0;"></div>
                            <span class="small fw-medium flex-grow-1" style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Preparing Data</span>
                        </div>
                    </div>
                </div>
                <div class="col-12 col-md-4">
                    <div class="p-3 border rounded h-100 d-flex align-items-center" id="status-train" style="min-height: 60px;">
                        <div class="d-flex align-items-center gap-2 w-100">
                            <div class="rounded-circle bg-secondary" style="width: 10px; height: 10px; flex-shrink: 0;"></div>
                            <span class="small fw-medium flex-grow-1" style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Training Model</span>
                        </div>
                    </div>
                </div>
                <div class="col-12 col-md-4">
                    <div class="p-3 border rounded h-100 d-flex align-items-center" id="status-upload" style="min-height: 60px;">
                        <div class="d-flex align-items-center gap-2 w-100">
                            <div class="rounded-circle bg-secondary" style="width: 10px; height: 10px; flex-shrink: 0;"></div>
                            <span class="small fw-medium flex-grow-1" style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Uploading Model</span>
                        </div>
                    </div>
                </div>
                <div class="col-12 col-md-4" id="status-export-container" style="display: none;">
                    <div class="p-3 border rounded h-100 d-flex align-items-center" id="status-export" style="min-height: 60px;">
                        <div class="d-flex align-items-center gap-2 w-100">
                            <div class="rounded-circle bg-secondary" style="width: 10px; height: 10px; flex-shrink: 0;"></div>
                            <span class="small fw-medium flex-grow-1" style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Converting Model</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // 상태 추적 초기화
    currentActiveStage = 'analyze.prompt'; // 초기 단계: Analyze Prompt
    completedStages.clear();
    
    // 초기 상태: Analyze Prompt를 활성화 상태로 설정
    updateStatusBox('status-analyze', 'active');
    updateStatusBoxText('status-analyze', 'Analyzing your prompt...');
}

// RabbitMQ 진행 상황 시작
async function startRabbitMQProgress() {
    if (!llmModalState.jobId) {
        console.error('[LLM Modal] Job ID is missing, cannot subscribe to RabbitMQ');
        console.warn('[LLM Modal] Progress updates will not be available without job_id');
        return;
    }

    try {
        // RabbitMQ 연결 확인
        if (!window.rabbitmqService) {
            console.error('[LLM Modal] RabbitMQ service not available');
            console.warn('[LLM Modal] Progress updates will not be available without RabbitMQ');
            return;
        }

        const jobId = llmModalState.jobId;
        console.log(`[LLM Modal] Subscribing to progress for job: ${jobId}`);
        
        if (!jobId) {
            console.error('[LLM Modal] Job ID is null or undefined, cannot subscribe');
            return;
        }

        // RabbitMQ 연결 확인 및 자동 연결
        if (!rabbitmqService.connected) {
            console.log('[LLM Modal] RabbitMQ not connected, attempting to connect...');
            try {
                await rabbitmqService.connect();
                console.log('[LLM Modal] RabbitMQ connected successfully');
            } catch (error) {
                console.error('[LLM Modal] Failed to connect to RabbitMQ:', error);
                return;
            }
        }

        // GPU 서버가 job.progress.{stage} 형식으로 메시지를 보내므로
        // 모든 progress 메시지를 구독하고 body의 job_id로 필터링
        // STOMP over WebSocket에서는 # (0개 이상) 와일드카드 지원
        const progressRoutingKey = `job.progress.#`;
        
        // 개별 routing key로 구독 (와일드카드가 안 될 수 있음)
        const individualKeys = [
            'job.progress.analyze.prompt',      // Analyze Prompt 단계 (새로 추가)
            'job.progress.train.download_dataset',
            'job.progress.train.prepare_split',
            'job.progress.train.start',
            'job.progress.upload',
            'job.progress.done',                // 완료 이벤트 (100%)
            'train.llm.log',                    // 학습 진행률 업데이트 (epoch별 퍼센트)
            'convert.exchanges',                // 변환 정보 수신
            'job.progress.onnx.done',           // ONNX 변환 완료
            'job.progress.trt.done',            // TensorRT 변환 완료
            'train.hpo'                         // 하이퍼파라미터 메시지
        ];
        
        // 에러 이벤트 구독 (job.{job_id}.error 또는 job.#.error)
        const errorKeys = [
            'job.#.error',  // 모든 job의 에러 이벤트 구독
        ];
        
        console.log(`[LLM Modal] Attempting to subscribe to progress messages`);
        console.log(`[LLM Modal] Expected job_id: "${jobId}" (type: ${typeof jobId})`);
        console.log(`[LLM Modal] ⚠️ Note: GPU server may use a different job_id. Accepting all progress messages for now.`);
        
        // 모든 progress 메시지를 받아서 처리 (job_id 필터링 제거)
        // 이유: LLM builder가 job_id를 재생성하므로 백엔드의 external_job_id와 불일치할 수 있음
        // 대신: 가장 최근에 받은 메시지의 진행 상황을 표시
        
        // 메시지 수신 시간 추적 (같은 stage의 이전 메시지는 무시)
        let lastMessageTime = {};
        let receivedJobIds = new Set();
        
        // 모든 개별 routing key에 구독
        individualKeys.forEach(routingKey => {
            try {
                // convert.exchanges는 jobs.event exchange를 사용 (LLM convert_dispatcher)
                // train.hpo는 jobs.cmd exchange를 사용
                let exchangeName = 'jobs.events';
                if (routingKey === 'convert.exchanges') {
                    exchangeName = 'jobs.event';
                } else if (routingKey === 'train.hpo') {
                    exchangeName = 'jobs.cmd';
                }
                
                const subscriptionId = rabbitmqService.subscribe(
                    routingKey,
                    (message) => { 
                        console.log(`[LLM Modal] 📨 Progress message received for ${routingKey}:`, message);
                        
                        // train.hpo 메시지 처리
                        if (routingKey === 'train.hpo') {
                            handleHyperparameterMessage(message);
                            return; // train.hpo는 여기서 처리 완료
                        }
                        
                        const messageJobId = String(message.job_id || message.jobId || 'unknown');
                        receivedJobIds.add(messageJobId);
                        
                        // job_id 로깅 (디버깅용)
                        if (messageJobId !== jobId) {
                            console.log(`[LLM Modal] ⚠️ Job ID differs: received="${messageJobId}", expected="${jobId}"`);
                            console.log(`[LLM Modal] 💡 Processing anyway (LLM builder may have regenerated job_id)`);
                        } else {
                            console.log(`[LLM Modal] ✅ Job ID matches: "${messageJobId}"`);
                        }
                        
                        // 중복 메시지 방지: 같은 stage의 메시지가 너무 빠르게 연속으로 오는 경우 무시
                        // train.llm.log 메시지는 epoch별로 오므로 모든 메시지를 처리 (중복 방지 제외)
                        if (routingKey !== 'train.llm.log') {
                            const stage = message.stage || routingKey;
                            const now = Date.now();
                            if (lastMessageTime[stage] && (now - lastMessageTime[stage] < 100)) {
                                console.log(`[LLM Modal] ⏭️ Skipping duplicate message for stage: ${stage}`);
                                return;
                            }
                            lastMessageTime[stage] = now;
                        } else {
                            // train.llm.log 메시지는 epoch 정보로 중복 방지
                            const epoch = message.epoch || 0;
                            const percentage = message.percentage || 0;
                            const logKey = `train.llm.log.${epoch}.${percentage}`;
                            const now = Date.now();
                            // 같은 epoch와 percentage의 메시지는 50ms 내에 무시 (너무 빠른 업데이트 방지)
                            if (lastMessageTime[logKey] && (now - lastMessageTime[logKey] < 50)) {
                                console.log(`[LLM Modal] ⏭️ Skipping duplicate train.llm.log message: epoch=${epoch}, percentage=${percentage}`);
                                return;
                            }
                            lastMessageTime[logKey] = now;
                        }
                        
                        // 모든 메시지 처리 (job_id 필터링 없음)
                        // routingKey 정보도 전달하여 완료 이벤트를 더 확실하게 감지
                        console.log(`[LLM Modal] ✅ Processing progress message:`, message);
                        handleProgressMessage(message, routingKey);
                        
                        // 실제 job_id가 다른 경우, llmModalState.jobId 업데이트
                        if (messageJobId !== 'unknown' && messageJobId !== jobId) {
                            console.log(`[LLM Modal] 🔄 Updating job_id from "${jobId}" to "${messageJobId}"`);
                            llmModalState.jobId = messageJobId;
                        }
                    },
                    'exchange',
                    exchangeName  // 동적으로 exchange 선택 (convert.exchanges는 jobs.event 사용)
                );
                llmModalState.rabbitmqSubscriptions.push({ routingKey, subscriptionId });
                console.log(`[LLM Modal] ✅ Subscribed to: ${routingKey}`);
            } catch (err) {
                console.error(`[LLM Modal] ❌ Failed to subscribe to ${routingKey}:`, err);
            }
        });
        
        // 에러 이벤트 구독 (job_id 필터링 없이 모든 에러 수신)
        // 이유: LLM builder가 job_id를 재생성하므로, 현재 job_id와 일치하지 않을 수 있음
        errorKeys.forEach(errorKey => {
            try {
                const errorSubscriptionId = rabbitmqService.subscribe(
                    errorKey,
                    (message) => {
                        console.error(`[LLM Modal] ❌ Error event received:`, message);
                        
                        const errorMessage = message.message || 'Unknown error occurred';
                        const errorStage = message.stage || 'unknown';
                        const messageJobId = String(message.job_id || message.jobId || 'unknown');
                        
                        console.log(`[LLM Modal] Error job_id: "${messageJobId}", current job_id: "${jobId}"`);
                        
                        // job_id가 일치하거나 unknown인 경우만 처리
                        // 다른 job의 에러는 무시 (하지만 LLM builder가 job_id를 재생성하므로 완화)
                        if (messageJobId !== 'unknown' && messageJobId !== jobId) {
                            // 받은 job_id가 현재 job_id와 다르면, 현재 job_id를 업데이트하고 처리
                            // (LLM builder가 job_id를 재생성했을 가능성)
                            console.log(`[LLM Modal] ⚠️ Error job_id differs: "${messageJobId}" vs "${jobId}"`);
                            console.log(`[LLM Modal] 💡 Processing anyway (LLM builder may have regenerated job_id)`);
                            llmModalState.jobId = messageJobId;
                        }
                        
                        // 에러 처리 (모든 에러 처리 - job_id가 다른 경우도 처리)
                        console.log(`[LLM Modal] 🚨 Processing error: stage="${errorStage}", message="${errorMessage.substring(0, 50)}..."`);
                        handleErrorMessage(errorMessage, errorStage);
                    },
                    'exchange',
                    'jobs.events'
                );
                llmModalState.rabbitmqSubscriptions.push({ routingKey: errorKey, subscriptionId: errorSubscriptionId });
                console.log(`[LLM Modal] ✅ Subscribed to error events: ${errorKey}`);
            } catch (err) {
                console.error(`[LLM Modal] ❌ Failed to subscribe to error events ${errorKey}:`, err);
            }
        });
        
        // 와일드카드도 시도 (작동하면 더 효율적) - 주석 처리 (개별 구독 사용)
        /*
        try {
            const subscriptionId = rabbitmqService.subscribe(
                progressRoutingKey,
                (message) => { 
                    console.log(`[LLM Modal] 📨 Progress message received (wildcard) for ${progressRoutingKey}:`, message);
                    
                    const messageJobId = String(message.job_id || message.jobId || 'unknown');
                    receivedJobIds.add(messageJobId);
                    
                    // job_id 로깅 (디버깅용)
                    if (messageJobId !== jobId) {
                        console.log(`[LLM Modal] ⚠️ Job ID differs: received="${messageJobId}", expected="${jobId}"`);
                        console.log(`[LLM Modal] 💡 Processing anyway (LLM builder may have regenerated job_id)`);
                    } else {
                        console.log(`[LLM Modal] ✅ Job ID matches: "${messageJobId}"`);
                    }
                    
                    // 중복 메시지 방지
                    const stage = message.stage || 'unknown';
                    const now = Date.now();
                    if (lastMessageTime[stage] && (now - lastMessageTime[stage] < 100)) {
                        console.log(`[LLM Modal] ⏭️ Skipping duplicate message for stage: ${stage}`);
                        return;
                    }
                    lastMessageTime[stage] = now;
                    
                    // 모든 메시지 처리
                    console.log(`[LLM Modal] ✅ Processing progress message:`, message);
                    handleProgressMessage(message);
                    
                    // 실제 job_id가 다른 경우, llmModalState.jobId 업데이트
                    if (messageJobId !== 'unknown' && messageJobId !== jobId) {
                        console.log(`[LLM Modal] 🔄 Updating job_id from "${jobId}" to "${messageJobId}"`);
                        llmModalState.jobId = messageJobId;
                    }
                },
                'exchange',
                'jobs.events'
            );
            llmModalState.rabbitmqSubscriptions.push({ routingKey: progressRoutingKey, subscriptionId });
            console.log(`[LLM Modal] ✅ Subscribed to wildcard: ${progressRoutingKey}`);
        } catch (error) {
            console.warn(`[LLM Modal] Wildcard subscription failed (using individual subscriptions):`, error);
        }
        */

    } catch (error) {
        console.error('[LLM Modal] Error setting up RabbitMQ subscriptions:', error);
        console.warn('[LLM Modal] Progress updates will not be available without RabbitMQ connection');
    }
}

// 진행 중인 단계 추적
let currentActiveStage = null;
let completedStages = new Set();

// RabbitMQ 메시지 처리
function handleProgressMessage(message, routingKey = '') {
    console.log(`[LLM Modal] 🎯 handleProgressMessage called with:`, message, `routingKey: ${routingKey}`);
    console.log(`[LLM Modal] Current step: ${llmModalState.currentStep}, required: 2`);
    
    if (llmModalState.currentStep !== 2) {
        console.warn(`[LLM Modal] ⚠️ Ignoring message - current step is ${llmModalState.currentStep}, not 2`);
        return;
    }
    
    // convert.exchanges 메시지 처리 (변환 정보 수신)
    if (routingKey === 'convert.exchanges') {
        console.log(`[LLM Modal] 🔄 Conversion info received:`, message);
        const onnx = String(message.onnx || 'false').toLowerCase() === 'true';
        const tensorrt = String(message.tensorrt || 'false').toLowerCase() === 'true';
        
        if (onnx || tensorrt) {
            llmModalState.needsConversion = true;
            llmModalState.conversionType = onnx ? 'onnx' : 'tensorrt';
            console.log(`[LLM Modal] ✅ Model conversion required: ${llmModalState.conversionType}`);
            
            // Export 박스 표시
            const exportContainer = document.getElementById('status-export-container');
            if (exportContainer) {
                exportContainer.style.display = 'block';
            }
        } else {
            // 둘 다 false인 경우 = 일반 모델 (변환 불필요)
            llmModalState.needsConversion = false;
            llmModalState.conversionType = null;
            console.log(`[LLM Modal] ✅ Regular model (no conversion needed)`);
        }
        return; // convert.exchanges 메시지는 여기서 처리 완료
    }
        
    // train.llm.log 메시지 처리 (epoch별 학습 진행률 업데이트)
    // 메시지 구조: { job_id, epoch, total_epochs, percentage }
    if (routingKey === 'train.llm.log') {
        const epoch = message.epoch || 0;
        const totalEpochs = message.total_epochs || 0;
        const percentage = message.percentage || 0;
        
        console.log(`[LLM Modal] 📊 Training progress update: epoch ${epoch}/${totalEpochs}, percentage: ${percentage}%`);
        
        // stage를 train.start로 설정
        const stage = 'train.start';
        
        // 프로그레스 바 업데이트
        const progressBar = document.getElementById('training-progress-bar');
        const progressText = document.getElementById('training-progress-text');
        const progressMessage = document.getElementById('training-progress-message');
        
        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
            progressBar.setAttribute('aria-valuenow', percentage);
            // 학습 중이면 애니메이션 유지
            if (percentage < 100) {
                progressBar.classList.add('progress-bar-striped', 'progress-bar-animated');
            } else {
                progressBar.classList.remove('progress-bar-striped', 'progress-bar-animated');
            }
        }
        if (progressText) {
            progressText.textContent = `${Math.round(percentage)}%`;
        }
        if (progressMessage) {
            // badge 스타일 유지
            if (!progressMessage.classList.contains('bg-success')) {
                progressMessage.className = 'badge bg-light text-dark px-3 py-2 fw-normal border';
            }
            if (totalEpochs > 0) {
                progressMessage.textContent = `Training: epoch ${epoch}/${totalEpochs} (${Math.round(percentage)}%)`;
            } else {
                progressMessage.textContent = `Training: epoch ${epoch} (${Math.round(percentage)}%)`;
            }
        }
        
        // train.start 단계를 active로 유지
        updateStatusBox('status-train', 'active');
        currentActiveStage = 'train.start';
        
        // 이전 단계들을 complete로 설정 (아직 완료되지 않았다면)
        const stageOrder = {
            'analyze.prompt': { id: 'status-analyze', order: 0 },
            'train.download_dataset': { id: 'status-download', order: 1 },
            'train.prepare_split': { id: 'status-prepare', order: 2 },
            'train.start': { id: 'status-train', order: 3 },
            'upload': { id: 'status-upload', order: 4 },
            'export': { id: 'status-export', order: 5 }
        };
        
        // train.start 이전 단계들을 complete로 설정
        Object.keys(stageOrder).forEach(key => {
            if (stageOrder[key].order < stageOrder['train.start'].order) {
                if (!completedStages.has(key)) {
                    updateStatusBox(stageOrder[key].id, 'complete');
                    completedStages.add(key);
                }
            }
        });
        
        // 100% 완료 시 완료 처리로 전환
        if (percentage >= 100) {
            console.log('[LLM Modal] Training completed via train.llm.log (100%)');
            updateStatusBox('status-train', 'complete');
            // upload 단계로 전환 준비
            currentActiveStage = 'upload';
        }
        
        return; // train.llm.log 메시지는 여기서 처리 완료
    }
    
    // 메시지에서 stage 추출 (message.stage 또는 routing key에서 추출)
    // RabbitMQ 메시지 구조: { job_id, event, stage, percent, message }
    let stage = message.stage || '';
    let percent = message.percent || 0;
    let messageText = message.message || '';

    console.log(`[LLM Modal] Raw Stage: "${stage}", Percent: ${percent}, Message: "${messageText}", RoutingKey: "${routingKey}"`);
    
    // ONNX/TensorRT 변환 완료 메시지 처리
    if (routingKey === 'job.progress.onnx.done' || routingKey === 'job.progress.trt.done') {
        console.log(`[LLM Modal] 🎉 Model conversion completed: ${routingKey}`);
        
        // Export 상태 박스를 complete로 설정
        updateStatusBox('status-export', 'complete');
        
        // 프로그레스 바 100%
        const progressBar = document.getElementById('training-progress-bar');
        const progressText = document.getElementById('training-progress-text');
        const progressMessage = document.getElementById('training-progress-message');
        
        if (progressBar) {
            progressBar.style.width = '100%';
            progressBar.setAttribute('aria-valuenow', 100);
            progressBar.classList.remove('progress-bar-striped', 'progress-bar-animated');
        }
        if (progressText) {
            progressText.textContent = '100%';
        }
        if (progressMessage) {
            progressMessage.className = 'badge bg-success text-white px-3 py-2 fw-normal border';
            const conversionType = routingKey.includes('onnx') ? 'ONNX' : 'TensorRT';
            progressMessage.textContent = `${conversionType} conversion completed successfully!`;
        }
        
        // Step 3로 이동
        const timer = setTimeout(() => {
            if (llmModalState.currentStep === 2) {
                cleanupRabbitMQSubscriptions();
                currentActiveStage = null;
                completedStages.clear();
                llmModalState.currentStep = 3;
                renderStep(3);
            }
        }, 1000);
        llmModalState.timers.push(timer);
        
        return; // onnx/trt done 이벤트는 여기서 처리 완료
    }
    
    // routingKey가 'job.progress.done'이면 완료로 처리 (더 확실한 감지)
    if (routingKey === 'job.progress.done' || routingKey.endsWith('.done')) {
        stage = 'done';
        console.log(`[LLM Modal] ✅ Detected completion via routingKey: ${routingKey}`);
    }

    // stage 값 정규화 (다양한 형식 지원)
    // 예: 'analyze.prompt', 'train.download_dataset', 'download_dataset', 'train.download' 등
    if (!stage && messageText) {
        // message에서 stage 추출 시도
        const lowerMessage = messageText.toLowerCase();
        if (lowerMessage.includes('analyze') || lowerMessage.includes('prompt') || lowerMessage.includes('parsing')) {
            stage = 'analyze.prompt';
        } else if (lowerMessage.includes('download')) {
            stage = 'train.download_dataset';
        } else if (lowerMessage.includes('prepare') || lowerMessage.includes('split')) {
            stage = 'train.prepare_split';
        } else if (lowerMessage.includes('train') && !lowerMessage.includes('download')) {
            stage = 'train.start';
        } else if (lowerMessage.includes('upload')) {
            stage = 'upload';
        } else if (lowerMessage.includes('done') || lowerMessage.includes('finish')) {
            stage = 'done';
        }
    }
    
    // done 이벤트 처리 (100% 완료)
    // 조건: stage가 'done'이거나, percent가 100 이상이거나, routingKey가 'job.progress.done'인 경우
    // 단, 변환이 필요한 경우는 여기서 완료 처리하지 않음 (onnx.done/trt.done에서 처리)
    if (stage === 'done' || percent >= 100 || routingKey === 'job.progress.done') {
        console.log('[LLM Modal] Training completed (100%), checking conversion requirements');
        console.log(`[LLM Modal] Completion detected: stage="${stage}", percent=${percent}, routingKey="${routingKey}"`);
        console.log(`[LLM Modal] Needs conversion: ${llmModalState.needsConversion}, type: ${llmModalState.conversionType}`);
        
        // 변환이 필요한 경우, Upload까지만 complete로 설정하고 Export는 active로 설정
        if (llmModalState.needsConversion) {
            console.log('[LLM Modal] 🔄 Conversion required, waiting for onnx/trt.done message');
            
            const stageOrder = {
                'analyze.prompt': { id: 'status-analyze', order: 0, label: 'Analyze Prompt' },
                'train.download_dataset': { id: 'status-download', order: 1, label: 'Downloading Dataset' },
                'train.prepare_split': { id: 'status-prepare', order: 2, label: 'Preparing Data' },
                'train.start': { id: 'status-train', order: 3, label: 'Training Model' },
                'upload': { id: 'status-upload', order: 4, label: 'Uploading Model' }
            };
            
            Object.keys(stageOrder).forEach(key => {
                updateStatusBox(stageOrder[key].id, 'complete');
            });
            
            // Export 상태를 active로 설정
            updateStatusBox('status-export', 'active');
            
            // 프로그레스 바는 95% 정도로 설정 (변환 대기 중)
            const progressBar = document.getElementById('training-progress-bar');
            const progressText = document.getElementById('training-progress-text');
            const progressMessage = document.getElementById('training-progress-message');
            
            if (progressBar) {
                progressBar.style.width = '95%';
                progressBar.setAttribute('aria-valuenow', 95);
                progressBar.classList.add('progress-bar-striped', 'progress-bar-animated');
            }
            if (progressText) {
                progressText.textContent = '95%';
            }
            if (progressMessage) {
                progressMessage.className = 'badge bg-light text-dark px-3 py-2 fw-normal border';
                const conversionType = llmModalState.conversionType === 'onnx' ? 'ONNX' : 'TensorRT';
                progressMessage.textContent = `Converting model to ${conversionType}...`;
            }
            
            return; // 변환 대기 중, onnx/trt.done에서 완료 처리
        }
        
        // 변환이 필요 없는 경우, 모든 상태 박스를 complete로 설정
        const stageOrder = {
            'analyze.prompt': { id: 'status-analyze', order: 0, label: 'Analyze Prompt' },
            'train.download_dataset': { id: 'status-download', order: 1, label: 'Downloading Dataset' },
            'train.prepare_split': { id: 'status-prepare', order: 2, label: 'Preparing Data' },
            'train.start': { id: 'status-train', order: 3, label: 'Training Model' },
            'upload': { id: 'status-upload', order: 4, label: 'Uploading Model' }
        };
        
        Object.keys(stageOrder).forEach(key => {
            updateStatusBox(stageOrder[key].id, 'complete');
        });
        
        // 프로그레스 바를 100%로 설정
        const progressBar = document.getElementById('training-progress-bar');
        const progressText = document.getElementById('training-progress-text');
        const progressMessage = document.getElementById('training-progress-message');
        
        if (progressBar) {
            progressBar.style.width = '100%';
            progressBar.setAttribute('aria-valuenow', 100);
            progressBar.classList.remove('progress-bar-striped', 'progress-bar-animated');
        }
        if (progressText) {
            progressText.textContent = '100%';
        }
        if (progressMessage) {
            // 완료 시 badge 스타일을 success로 변경
            progressMessage.className = 'badge bg-success text-white px-3 py-2 fw-normal border';
            progressMessage.textContent = messageText || 'Training completed successfully!';
        }
        
        // Step 3로 이동
        const timer = setTimeout(() => {
            if (llmModalState.currentStep === 2) {
                cleanupRabbitMQSubscriptions();
                currentActiveStage = null;
                completedStages.clear();
                llmModalState.currentStep = 3;
                renderStep(3);
            }
        }, 1000);
        llmModalState.timers.push(timer);
        
        return; // done 이벤트는 여기서 처리 완료
    }

    // 단계 순서 정의 (Analyze Prompt 추가) - done 처리 이후에만 사용
    const stageOrder = {
        'analyze.prompt': { id: 'status-analyze', order: 0, label: 'Analyze Prompt' },
        'train.download_dataset': { id: 'status-download', order: 1, label: 'Downloading Dataset' },
        'train.prepare_split': { id: 'status-prepare', order: 2, label: 'Preparing Data' },
        'train.start': { id: 'status-train', order: 3, label: 'Training Model' },
        'upload': { id: 'status-upload', order: 4, label: 'Uploading Model' },
        'export': { id: 'status-export', order: 5, label: 'Converting Model' }
    };

    // 프로그레스 바 업데이트
    const progressBar = document.getElementById('training-progress-bar');
    const progressText = document.getElementById('training-progress-text');
    const progressMessage = document.getElementById('training-progress-message');

    if (progressBar) {
        progressBar.style.width = `${percent}%`;
        progressBar.setAttribute('aria-valuenow', percent);
    }
    if (progressText) {
        progressText.textContent = `${Math.round(percent)}%`;
    }
    if (progressMessage && messageText) {
        // badge 스타일 유지 (완료 상태가 아니면 기본 스타일)
        if (!progressMessage.classList.contains('bg-success')) {
            progressMessage.className = 'badge bg-light text-dark px-3 py-2 fw-normal border';
        }
        progressMessage.textContent = messageText;
    }

    // 현재 단계에 해당하는 상태 박스 찾기
    const currentStageInfo = stageOrder[stage];
    
    if (currentStageInfo) {
        const statusId = currentStageInfo.id;
        console.log(`[LLM Modal] Found stage: ${stage} -> ${statusId}, order: ${currentStageInfo.order}`);
        
        // 새로운 단계로 전환되는 경우
        if (currentActiveStage && currentActiveStage !== stage) {
            const previousStageInfo = stageOrder[currentActiveStage];
            
            // 이전 단계가 현재 단계보다 순서가 앞서면 완료 처리
            if (previousStageInfo && previousStageInfo.order < currentStageInfo.order) {
                console.log(`[LLM Modal] ✅ Completing previous stage: ${currentActiveStage} -> ${previousStageInfo.id}`);
                updateStatusBox(previousStageInfo.id, 'complete');
                completedStages.add(currentActiveStage);
            }
            
            // 이전 단계와 현재 단계 사이의 모든 단계를 완료 처리
            Object.keys(stageOrder).forEach(key => {
                const stageInfo = stageOrder[key];
                if (stageInfo.order < currentStageInfo.order && 
                    stageInfo.order > (previousStageInfo?.order || -1) &&
                    !completedStages.has(key)) {
                    console.log(`[LLM Modal] ✅ Completing intermediate stage: ${key} -> ${stageInfo.id}`);
                    updateStatusBox(stageInfo.id, 'complete');
                    completedStages.add(key);
                }
            });
        }
        
        // 현재 단계를 active로 설정
        currentActiveStage = stage;
        updateStatusBox(statusId, 'active');
        if (messageText) {
            updateStatusBoxText(statusId, messageText);
        }
        
        // 프로그레스 메시지도 업데이트 (analyze.prompt 단계 특별 처리)
        const progressMessage = document.getElementById('training-progress-message');
        if (progressMessage) {
            if (stage === 'analyze.prompt') {
                // badge 스타일 유지
                if (!progressMessage.classList.contains('bg-success')) {
                    progressMessage.className = 'badge bg-light text-dark px-3 py-2 fw-normal border';
                }
                progressMessage.textContent = messageText || 'Analyzing your prompt...';
            } else if (messageText) {
                // badge 스타일 유지
                if (!progressMessage.classList.contains('bg-success')) {
                    progressMessage.className = 'badge bg-light text-dark px-3 py-2 fw-normal border';
                }
                progressMessage.textContent = messageText;
            }
        }
        
        console.log(`[LLM Modal] 🎯 Activated stage: ${stage} (${statusId})`);
    } else {
        // stage를 찾을 수 없으면 percent 기반으로 추정
        console.warn(`[LLM Modal] Unknown stage: "${stage}", using percent-based estimation`);
        
        // Percent 기반 단계 추정 (Analyze Prompt 제외)
        let estimatedStage = null;
        if (percent < 10) {
            estimatedStage = 'analyze.prompt';
        } else if (percent < 25) {
            estimatedStage = 'train.download_dataset';
        } else if (percent < 40) {
            estimatedStage = 'train.prepare_split';
        } else if (percent < 90) {
            estimatedStage = 'train.start';
        } else {
            estimatedStage = 'upload';
        }
        
        const estimatedStageInfo = stageOrder[estimatedStage];
        if (estimatedStageInfo) {
            // 이전 단계들을 complete로 설정
            Object.keys(stageOrder).forEach(key => {
                const stageInfo = stageOrder[key];
                if (stageInfo.order < estimatedStageInfo.order) {
                    if (!completedStages.has(key)) {
                        updateStatusBox(stageInfo.id, 'complete');
                        completedStages.add(key);
                    }
                }
            });
            
            // 현재 단계를 active로 설정
            if (currentActiveStage !== estimatedStage) {
                if (currentActiveStage) {
                    const prevStageInfo = stageOrder[currentActiveStage];
                    if (prevStageInfo) {
                        updateStatusBox(prevStageInfo.id, 'complete');
                        completedStages.add(currentActiveStage);
                    }
                }
                currentActiveStage = estimatedStage;
                updateStatusBox(estimatedStageInfo.id, 'active');
                if (messageText) {
                    updateStatusBoxText(estimatedStageInfo.id, messageText);
                }
            }
            }
        }
}

// 에러 메시지 처리 (프론트엔드에서만 처리)
function handleErrorMessage(errorMessage, errorStage) {
    console.error(`[LLM Modal] 🚨 Error occurred in stage: ${errorStage}`);
    console.error(`[LLM Modal] Error message: ${errorMessage}`);
    
    if (llmModalState.currentStep !== 2) {
        console.warn(`[LLM Modal] ⚠️ Ignoring error - current step is ${llmModalState.currentStep}, not 2`);
        return;
    }
    
    // 프로그레스 메시지 업데이트
    const progressMessage = document.getElementById('training-progress-message');
    if (progressMessage) {
        const shortError = errorMessage.length > 150 
            ? errorMessage.substring(0, 150) + '...' 
            : errorMessage;
        progressMessage.textContent = `Error: ${shortError}`;
        progressMessage.className = 'text-danger fw-semibold mt-2 d-block';
    }
    
    // 프로그레스 바를 빨간색으로 변경 (애니메이션 제거)
    const progressBar = document.getElementById('training-progress-bar');
    if (progressBar) {
        progressBar.classList.remove('progress-bar-striped', 'progress-bar-animated');
        progressBar.classList.add('bg-danger');
    }
    
    // 에러가 발생한 단계를 빨간색으로 표시
    const stageOrder = {
        'analyze.prompt': { id: 'status-analyze', order: 0 },
        'train.download_dataset': { id: 'status-download', order: 1 },
        'train.prepare_split': { id: 'status-prepare', order: 2 },
        'train.start': { id: 'status-train', order: 3 },
        'train': { id: 'status-train', order: 3 }, // train 단계도 train.start와 동일하게 처리
        'upload': { id: 'status-upload', order: 4 }
    };
    
    const errorStageInfo = stageOrder[errorStage] || stageOrder['train.start'];
    if (errorStageInfo) {
        const statusBox = document.getElementById(errorStageInfo.id);
        if (statusBox) {
            // 에러 상태로 변경 (빨간색)
            statusBox.className = 'p-3 border border-danger rounded bg-danger bg-opacity-10';
            const dot = statusBox.querySelector('.rounded-circle, .spinner-border');
            if (dot) {
                dot.className = 'rounded-circle bg-danger';
                dot.style.width = '8px';
                dot.style.height = '8px';
                dot.classList.remove('spinner-border', 'spinner-border-sm', 'text-primary');
            }
            const text = statusBox.querySelector('span');
            if (text) {
                // 에러 메시지가 길면 줄임
                const shortError = errorMessage.length > 40 
                    ? errorMessage.substring(0, 40) + '...' 
                    : errorMessage;
                text.textContent = `Error: ${shortError}`;
                text.className = 'small text-danger fw-semibold';
            }
        }
    }
    
    // Toast 메시지 표시 (CUDA 에러인 경우 친절한 메시지)
    let userFriendlyMessage = errorMessage;
    if (errorMessage.includes('CUDA') || errorMessage.includes('cuda')) {
        userFriendlyMessage = 'GPU 서버에 CUDA 디바이스가 없습니다. 관리자에게 문의하세요.';
    }
    
    // Toast 메시지 표시 (showToast 함수가 전역으로 있음)
    if (typeof showToast === 'function') {
        showToast(`Training failed: ${userFriendlyMessage}`, 'error');
    } else {
        // Toast가 없으면 alert 표시
        console.error(`[LLM Modal] showToast function not available. Error: ${userFriendlyMessage}`);
        alert(`Training Error: ${userFriendlyMessage}`);
    }
}

// 상태 박스 업데이트
function updateStatusBox(statusId, state) {
    const element = document.getElementById(statusId);
    if (!element) {
        console.warn(`[LLM Modal] Status box not found: ${statusId}`);
        return;
    }

    // 기존 스타일 초기화
    element.classList.remove('border-primary', 'border-success', 'border-danger', 
                            'bg-primary', 'bg-success', 'bg-danger', 
                            'bg-opacity-10');
    
    // 점 요소 찾기 (spinner-border 먼저, 없으면 rounded-circle)
    let dot = element.querySelector('.spinner-border');
    if (!dot) {
        dot = element.querySelector('.rounded-circle');
    }
    
    const text = element.querySelector('span');

    // h-100 클래스 유지 (높이 균등 배치)
    const hasH100 = element.classList.contains('h-100');
    const h100Class = hasH100 ? ' h-100' : '';
    const hasFlex = element.classList.contains('d-flex');
    const flexClass = hasFlex ? ' d-flex align-items-center' : '';
    const minHeight = element.style.minHeight || '60px';
    
    if (state === 'active') {
        // 활성 상태: 파란색 테두리와 배경
        element.className = 'p-3 border border-primary rounded bg-primary bg-opacity-10' + h100Class + flexClass;
        element.style.minHeight = minHeight;
        
        if (dot) {
            // 회전 애니메이션 없이 파란색 점만 표시
            // spinner-border 클래스 완전히 제거
            dot.className = 'rounded-circle bg-primary';
            dot.style.width = '10px';
            dot.style.height = '10px';
            dot.style.minWidth = '10px';
            dot.style.minHeight = '10px';
            dot.style.flexShrink = '0';
            // 모든 애니메이션 및 회전 효과 제거
            dot.style.animation = 'none';
            dot.style.transform = 'none';
            dot.style.rotate = 'none';
            // Bootstrap spinner 관련 속성 제거
            dot.removeAttribute('role');
            const hiddenSpan = dot.querySelector('.visually-hidden');
            if (hiddenSpan) {
                hiddenSpan.remove();
            }
        }
        
        if (text) {
            text.className = 'small text-primary fw-semibold flex-grow-1';
            text.style.whiteSpace = 'nowrap';
            text.style.overflow = 'hidden';
            text.style.textOverflow = 'ellipsis';
        }
    } else if (state === 'complete') {
        // 완료 상태: 초록색 테두리와 배경
        element.className = 'p-3 border border-success rounded bg-success bg-opacity-10' + h100Class + flexClass;
        element.style.minHeight = minHeight;
        
        if (dot) {
            // 초록색 점으로 변경
            dot.className = 'rounded-circle bg-success';
            dot.style.width = '10px';
            dot.style.height = '10px';
            dot.style.minWidth = '10px';
            dot.style.minHeight = '10px';
            dot.style.flexShrink = '0';
            // 모든 애니메이션 및 회전 효과 제거
            dot.style.animation = 'none';
            dot.style.transform = 'none';
            dot.style.rotate = 'none';
            // Bootstrap spinner 관련 속성 제거
            dot.removeAttribute('role');
            const hiddenSpan = dot.querySelector('.visually-hidden');
            if (hiddenSpan) {
                hiddenSpan.remove();
            }
        }
        
        if (text) {
            text.className = 'small text-success fw-semibold flex-grow-1';
            text.style.whiteSpace = 'nowrap';
            text.style.overflow = 'hidden';
            text.style.textOverflow = 'ellipsis';
        }
    } else {
        // 기본 상태: 회색 테두리
        element.className = 'p-3 border border-secondary rounded' + h100Class + flexClass;
        element.style.minHeight = minHeight;
        
        if (dot) {
            dot.className = 'rounded-circle bg-secondary';
            dot.style.width = '10px';
            dot.style.height = '10px';
            dot.style.minWidth = '10px';
            dot.style.minHeight = '10px';
            dot.style.flexShrink = '0';
            // 모든 애니메이션 및 회전 효과 제거
            dot.style.animation = 'none';
            dot.style.transform = 'none';
            dot.style.rotate = 'none';
            // Bootstrap spinner 관련 속성 제거
            dot.removeAttribute('role');
            const hiddenSpan = dot.querySelector('.visually-hidden');
            if (hiddenSpan) {
                hiddenSpan.remove();
            }
        }
        
        if (text) {
            text.className = 'small text-secondary fw-medium flex-grow-1';
            text.style.whiteSpace = 'nowrap';
            text.style.overflow = 'hidden';
            text.style.textOverflow = 'ellipsis';
        }
    }
}

// 상태 박스 텍스트 업데이트
function updateStatusBoxText(statusId, message) {
    const element = document.getElementById(statusId);
    if (!element) return;
    
    const text = element.querySelector('span');
    if (text && message) {
        // 메시지가 있으면 텍스트 업데이트
        text.textContent = message;
        // 텍스트 오버플로우 처리 유지
        text.style.whiteSpace = 'nowrap';
        text.style.overflow = 'hidden';
        text.style.textOverflow = 'ellipsis';
    }
}

// RabbitMQ 구독 정리
function cleanupRabbitMQSubscriptions() {
    if (llmModalState.rabbitmqSubscriptions && llmModalState.rabbitmqSubscriptions.length > 0) {
        console.log(`[LLM Modal] Cleaning up ${llmModalState.rabbitmqSubscriptions.length} RabbitMQ subscriptions`);
        llmModalState.rabbitmqSubscriptions.forEach(({ routingKey, subscriptionId }) => {
            try {
                if (window.rabbitmqService && rabbitmqService.connected) {
                    rabbitmqService.unsubscribe(routingKey);
                    console.log(`[LLM Modal] ✅ Unsubscribed from: ${routingKey}`);
                } else {
                    console.warn(`[LLM Modal] ⚠️ RabbitMQ service not available or not connected, skipping unsubscribe for: ${routingKey}`);
                }
            } catch (error) {
                console.error(`[LLM Modal] ❌ Error unsubscribing from ${routingKey}:`, error);
                // 에러가 발생해도 계속 진행 (이미 구독이 해제되었을 수 있음)
            }
        });
        llmModalState.rabbitmqSubscriptions = [];
        console.log('[LLM Modal] ✅ All RabbitMQ subscriptions cleaned up');
    } else {
        console.log('[LLM Modal] No RabbitMQ subscriptions to clean up');
    }
}

// 진행 상황 시작 (RabbitMQ만 사용, fallback 비활성화)
function startTrainingProgress() {
    console.log('[LLM Modal] Starting RabbitMQ progress monitoring (fallback disabled)');
    
    // RabbitMQ 구독 시작
    startRabbitMQProgress();
    
    // Fallback 비활성화 - RabbitMQ 메시지만 사용
    // 메시지를 받지 못하면 진행 상황이 업데이트되지 않음 (정상 동작)
    // 사용자가 실제 진행 상황을 볼 수 있도록 fallback 제거
}

// Fallback: 시뮬레이션된 진행 상황
function startTrainingProgressFallback() {
    console.log('[LLM Modal] Using fallback progress simulation');
    
    let currentPercent = 0;
    const updateProgress = () => {
        if (llmModalState.currentStep !== 2) return;
        
        currentPercent += Math.random() * 10;
        if (currentPercent > 100) currentPercent = 100;

        const progressBar = document.getElementById('training-progress-bar');
        const progressText = document.getElementById('training-progress-text');
        const progressMessage = document.getElementById('training-progress-message');

        if (progressBar) {
            progressBar.style.width = `${currentPercent}%`;
        }
        if (progressText) {
            progressText.textContent = `${Math.round(currentPercent)}%`;
        }
        if (progressMessage) {
            if (currentPercent < 25) progressMessage.textContent = 'Downloading dataset...';
            else if (currentPercent < 50) progressMessage.textContent = 'Preparing data...';
            else if (currentPercent < 90) progressMessage.textContent = 'Training model...';
            else progressMessage.textContent = 'Uploading model...';
        }

        if (currentPercent >= 100) {
            const timer = setTimeout(() => {
                if (llmModalState.currentStep === 2) {
                    llmModalState.currentStep = 3;
                    renderStep(3);
                }
            }, 1000);
            llmModalState.timers.push(timer);
        } else {
            const timer = setTimeout(updateProgress, 1000 + Math.random() * 2000);
            llmModalState.timers.push(timer);
        }
    };

    updateProgress();
}

// Step 3: Completion
function renderCompletionStep(container) {
    const job = llmModalState.trainingJob;
    const jobInfo = job ? `
        <div class="card border-0 bg-light mb-4">
            <div class="card-body">
                <h6 class="card-title fw-semibold mb-3">Training Job Info</h6>
                <div class="text-start">
                    <div class="mb-2">
                        <small class="text-muted">Job Name:</small>
                        <div class="fw-semibold">${job.name || 'N/A'}</div>
                    </div>
                    <div class="mb-2">
                        <small class="text-muted">Status:</small>
                        <div>
                            <span class="badge bg-success">
                                COMPLETED
                            </span>
                        </div>
                    </div>
                    ${job.architecture ? `
                    <div class="mb-2">
                        <small class="text-muted">Architecture:</small>
                        <div class="fw-semibold">${job.architecture}</div>
                    </div>
                    ` : ''}
                    ${job.hyperparameters?.epochs ? `
                    <div class="mb-2">
                        <small class="text-muted">Epochs:</small>
                        <div class="fw-semibold">${job.hyperparameters.epochs}</div>
                    </div>
                    ` : ''}
                </div>
            </div>
        </div>
    ` : '';

    container.innerHTML = `
        <div class="text-center">
            <div class="mb-4">
                <div class="d-inline-flex align-items-center justify-content-center rounded-circle bg-success bg-opacity-10" 
                     style="width: 80px; height: 80px; margin-bottom: 1rem;">
                    <span class="display-4 text-success">✓</span>
                </div>
            </div>
            <h3 class="fw-bold mb-3">Training Complete!</h3>
            <p class="text-muted mb-4">Model training has been successfully completed.</p>
            
            ${jobInfo}

            <!-- Action Button -->
            <div class="d-grid gap-2 col-md-6 mx-auto mt-4">
                <button class="btn btn-primary btn-lg" onclick="resetLLMModal()">
                    🔄 Start New Training
                </button>
                <button class="btn btn-outline-secondary" onclick="closeLLMModal()">
                    Close
                </button>
            </div>
        </div>
    `;
}

function resetLLMModal() {
    // Clear all timers
    llmModalState.timers.forEach(timer => clearTimeout(timer));
    llmModalState.timers = [];
    
    // Cleanup RabbitMQ subscriptions
    cleanupRabbitMQSubscriptions();
    
    // Reset stage tracking
    currentActiveStage = null;
    completedStages.clear();
    
    llmModalState.currentStep = 1;
    llmModalState.query = "";
    llmModalState.selectedDatasetId = null;
    llmModalState.trainingJob = null;
    llmModalState.jobId = null;
    llmModalState.hyperparameters = null;
    renderStep(1);
    // Update dataset select after reset
    updateDatasetSelect();
}

function closeLLMModal() {
    // Clear all timers
    llmModalState.timers.forEach(timer => clearTimeout(timer));
    llmModalState.timers = [];
    
    // Cleanup RabbitMQ subscriptions
    cleanupRabbitMQSubscriptions();
    
    resetLLMModal();
}

// 하이퍼파라미터 메시지 처리
function handleHyperparameterMessage(message) {
    console.log('[LLM Modal] 📨 Received hyperparameter message:', message);
    
    try {
        const { job_id, hyperparams } = message;
        
        if (!job_id || !hyperparams) {
            console.warn('[LLM Modal] Invalid hyperparameter message format:', message);
            return;
        }
        
        // 하이퍼파라미터 저장
        llmModalState.hyperparameters = hyperparams;
        console.log('[LLM Modal] ✅ Stored hyperparameters');
        
        // 버튼 활성화
        const hyperparameterBtn = document.getElementById('hyperparameter-btn');
        if (hyperparameterBtn) {
            hyperparameterBtn.disabled = false;
            hyperparameterBtn.classList.remove('btn-secondary');
            hyperparameterBtn.classList.add('btn-primary');
            hyperparameterBtn.style.opacity = '1';
            hyperparameterBtn.style.cursor = 'pointer';
            console.log('[LLM Modal] ✅ Hyperparameter button activated');
        } else {
            console.warn('[LLM Modal] Hyperparameter button not found');
        }
        
    } catch (error) {
        console.error('[LLM Modal] Error handling hyperparameter message:', error);
    }
}

// 하이퍼파라미터 모달 표시 (LLM 모달에서 호출)
function showHyperparameterModalFromLLM() {
    if (!llmModalState.hyperparameters) {
        showToast('Hyperparameters not available yet', 'warning');
        return;
    }
    
    // TrainingPage의 showHyperparameterModal 함수 사용
    if (window.trainingPage && typeof window.trainingPage.showHyperparameterModal === 'function') {
        // job_id를 사용하여 모달 표시 (llmModalState.jobId 사용)
        const jobId = llmModalState.jobId || 'llm-training';
        window.trainingPage.hyperparameters = {};
        window.trainingPage.hyperparameters[jobId] = llmModalState.hyperparameters;
        window.trainingPage.showHyperparameterModal(jobId);
    } else {
        // TrainingPage가 없으면 직접 모달 생성
        showHyperparameterModalDirect(llmModalState.hyperparameters);
    }
}

// 직접 하이퍼파라미터 모달 표시
function showHyperparameterModalDirect(hyperparams) {
    const modalHTML = `
        <div class="modal fade" id="llmHyperparameterModal" tabindex="-1">
            <div class="modal-dialog modal-lg modal-dialog-scrollable">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="bi bi-sliders me-2"></i>Hyperparameters
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row g-3">
                            ${renderHyperparameterFields(hyperparams)}
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal if any
    const existingModal = document.getElementById('llmHyperparameterModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Add modal to body
    document.body.insertAdjacentHTML('beforeend', modalHTML);

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('llmHyperparameterModal'));
    modal.show();
}

// 하이퍼파라미터 필드 렌더링
function renderHyperparameterFields(hyperparams) {
    const fields = [];
    
    // Group hyperparameters by category
    const categories = {
        'Model': ['model_name'],
        'Training': ['epochs', 'batch', 'imgsz', 'workers', 'patience'],
        'Optimizer': ['optimizer', 'lr0', 'lrf', 'weight_decay', 'momentum'],
        'Learning Rate Schedule': ['warmup_epochs', 'warmup_bias_lr'],
        'Augmentation': ['augment', 'mosaic', 'mixup'],
        'Other': ['amp']
    };

    // Helper to format value
    const formatValue = (value) => {
        if (typeof value === 'boolean') {
            return value ? '<span class="badge bg-success">Yes</span>' : '<span class="badge bg-secondary">No</span>';
        }
        if (typeof value === 'number') {
            return value.toLocaleString();
        }
        return String(value);
    };

    // Helper to format label
    const formatLabel = (key) => {
        const labels = {
            'model_name': 'Model Name',
            'epochs': 'Epochs',
            'batch': 'Batch Size',
            'imgsz': 'Image Size',
            'workers': 'Workers',
            'optimizer': 'Optimizer',
            'lr0': 'Initial Learning Rate',
            'lrf': 'Final Learning Rate',
            'weight_decay': 'Weight Decay',
            'momentum': 'Momentum',
            'warmup_epochs': 'Warmup Epochs',
            'warmup_bias_lr': 'Warmup Bias LR',
            'augment': 'Augmentation',
            'mosaic': 'Mosaic',
            'mixup': 'Mixup',
            'amp': 'Mixed Precision (AMP)',
            'patience': 'Early Stopping Patience'
        };
        return labels[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    };

    // Render each category
    Object.keys(categories).forEach(category => {
        const keys = categories[category];
        const hasAny = keys.some(key => hyperparams.hasOwnProperty(key));
        
        if (hasAny) {
            fields.push(`
                <div class="col-12">
                    <h6 class="text-primary border-bottom pb-2 mb-3">${category}</h6>
                </div>
            `);
            
            keys.forEach(key => {
                if (hyperparams.hasOwnProperty(key)) {
                    fields.push(`
                        <div class="col-md-6">
                            <div class="card border-0 bg-light">
                                <div class="card-body p-3">
                                    <p class="text-muted small mb-1">${formatLabel(key)}</p>
                                    <p class="fw-bold mb-0">${formatValue(hyperparams[key])}</p>
                                </div>
                            </div>
                        </div>
                    `);
                }
            });
        }
    });

    // Add any remaining fields not in categories
    const categorizedKeys = Object.values(categories).flat();
    Object.keys(hyperparams).forEach(key => {
        if (!categorizedKeys.includes(key) && key !== 'job_id') {
            fields.push(`
                <div class="col-md-6">
                    <div class="card border-0 bg-light">
                        <div class="card-body p-3">
                            <p class="text-muted small mb-1">${formatLabel(key)}</p>
                            <p class="fw-bold mb-0">${formatValue(hyperparams[key])}</p>
                        </div>
                    </div>
                </div>
            `);
        }
    });

    return fields.join('');
}

// Make functions globally available
window.showLLMModal = showLLMModal;
window.closeLLMModal = closeLLMModal;
window.submitQuery = submitQuery;
window.resetLLMModal = resetLLMModal;
window.onDatasetChange = onDatasetChange;
window.showHyperparameterModalFromLLM = showHyperparameterModalFromLLM;

