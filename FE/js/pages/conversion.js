// Conversion Page Component

class ConversionPage {
    constructor() {
        this.models = [];
        this.selectedModelId = null;
        this.selectedS3Uri = null;
        this.selectedS3Key = null;
    }

    async init() {
        await this.loadModels();
        // 전역에서 접근 가능하도록 설정
        window.currentPage = this;
    }

    async loadModels() {
        try {
            // Get trained models from S3 (backend now includes artifact info from database)
            const trainedModels = await window.apiService.getTrainedModels();
            
            // Transform to match expected format with s3_uri
            // Backend now provides: has_artifact, artifact_id, model_version_id
            this.models = trainedModels.map(m => ({
                id: m.id || m.model_id || m.model_name,
                model_id: m.model_id,
                name: m.model_name || '',
                s3_key: m.s3_key || m.relative_path,
                s3_uri: m.s3_bucket ? `s3://${m.s3_bucket}/${m.s3_key || m.relative_path}` : null,
                file_size_mb: m.file_size_mb,
                hasArtifact: m.has_artifact || false,  // From backend database check
                artifact_id: m.artifact_id || null,
                model_version_id: m.model_version_id || null
            })).filter(m => m.s3_uri); // Only include models with valid S3 URI
            
            this.renderModelsSection();
        } catch (error) {
            console.error('Failed to load models:', error);
            this.models = [];
            this.renderModelsSection();
        }
    }

    async selectModel(modelId, s3Uri, hasArtifact, s3Key) {
        if (!s3Uri) {
            this.showConversionErrorModal('This model does not have a valid S3 URI. Please ensure the model has been uploaded to S3.');
            return;
        }
        
        if (!hasArtifact) {
            const proceed = await this.showWarningModal(
                'Warning: This model may not have an artifact record in the database.',
                'Conversion requires the model to have a ModelArtifact record.\n\nIf conversion fails, the model may need to be re-uploaded or the artifact record created.\n\nDo you want to proceed anyway?'
            );
            if (!proceed) {
                return;
            }
        }
        
        this.selectedModelId = modelId;
        this.selectedS3Uri = s3Uri;
        this.selectedS3Key = s3Key;
        
        // UI 업데이트 (선택된 모델 하이라이트)
        document.querySelectorAll('.model-card').forEach(card => {
            card.classList.remove('border-primary', 'border-2');
        });
        const selectedCard = document.querySelector(`[data-model-id="${modelId}"]`);
        if (selectedCard) {
            selectedCard.classList.add('border-primary', 'border-2');
        }
    }

    async startConversion() {
        if (!this.selectedS3Uri) {
            this.showConversionErrorModal('Please select a model first');
            return;
        }

        try {
            const formatSelect = document.getElementById('target-format');
            const precisionSelect = document.getElementById('precision-select');
            
            if (!formatSelect || !precisionSelect) {
                this.showConversionErrorModal('Please wait for the page to fully load');
                return;
            }
            
            const format = formatSelect.value;
            const precision = precisionSelect.value;

            // 트레이닝된 모델의 원본 경로를 기준으로 conversion 경로 구성
            // 원본: models/{dataset_name}/train/{version}/best.pt
            // 결과: models/{dataset_name}/train/{version}/{format}/{version}/model.{ext}
            if (!this.selectedS3Key) {
                this.showConversionErrorModal('Model S3 path information is missing. Please select the model again.');
                return;
            }

            // S3 key에서 파일명을 제거하고 디렉토리 경로만 가져오기
            const s3KeyParts = this.selectedS3Key.split('/');
            s3KeyParts.pop(); // 마지막 파일명 제거
            const baseDir = s3KeyParts.join('/');

            // 다음 버전 번호 가져오기
            const versionResponse = await window.apiService.getNextConversionVersion(baseDir, format);
            const version = versionResponse.version || 'v1';
            const prefix = `${baseDir}/${format}/${version}`;

            let payload;
            let result;
            
            if (format === 'onnx') {
                payload = {
                    model: {
                        s3_uri: this.selectedS3Uri
                    },
                    output: {
                        prefix: prefix,
                        model_name: 'model.onnx'
                    },
                    ops: {
                        dynamic: true,
                        simplify: true,
                        opset: 13,
                        imgsz: 640,
                        precision: precision  // fp32, fp16, int8
                    }
                };
                console.log('[Conversion] ONNX payload:', JSON.stringify(payload, null, 2));
                result = await window.apiService.convertToOnnx(payload);
                console.log('ONNX conversion started:', result);
            } else if (format === 'tensorrt') {
                payload = {
                    model: {
                        s3_uri: this.selectedS3Uri
                    },
                    output: {
                        prefix: prefix,
                        model_name: 'model.engine'
                    },
                    trt: {
                        precision: precision,
                        imgsz: 640,
                        dynamic: true
                    }
                };
                console.log('[Conversion] TensorRT payload:', JSON.stringify(payload, null, 2));
                result = await window.apiService.convertToTensorRT(payload);
                console.log('TensorRT conversion started:', result);
            } else {
                this.showConversionErrorModal('Unsupported format. Please select ONNX or TensorRT.');
                return;
            }

            this.showConversionSuccessModal(format.toUpperCase(), result.job_id || 'N/A');
        } catch (error) {
            console.error('Conversion failed:', error);
            this.showConversionErrorModal(error.message || 'Unknown error');
        }
    }

    renderModelsSection() {
        const modelsContainer = document.getElementById('models-list');
        if (!modelsContainer) return;

        if (this.models.length === 0) {
            modelsContainer.innerHTML = `
                <div class="text-center py-5">
                    <i class="bi bi-inbox text-muted" style="font-size: 3rem;"></i>
                    <p class="text-muted mt-3">No trained models available</p>
                    <a href="#/training" class="btn btn-primary mt-2">Train a Model</a>
                </div>
            `;
            return;
        }

        modelsContainer.innerHTML = this.models.map(model => `
            <div class="card mb-3 hover-shadow cursor-pointer model-card ${!model.hasArtifact ? 'border-warning' : ''}" 
                 data-model-id="${model.id}"
                 onclick="window.currentPage.selectModel('${model.id}', '${model.s3_uri || ''}', ${model.hasArtifact}, '${model.s3_key || ''}')">
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <h6 class="fw-bold mb-2">${model.name}</h6>
                            <div class="d-flex gap-3 text-muted small">
                                <span><i class="bi bi-cpu me-1"></i>PyTorch</span>
                                <span><i class="bi bi-hdd me-1"></i>${model.file_size_mb || 'N/A'}MB</span>
                                ${model.hasArtifact 
                                    ? '<span class="text-success"><i class="bi bi-check-circle me-1"></i>Ready for Conversion</span>' 
                                    : '<span class="text-warning"><i class="bi bi-exclamation-triangle me-1"></i>No Artifact Record</span>'}
                            </div>
                            ${model.s3_key ? `<small class="text-muted d-block mt-1">${model.s3_key}</small>` : ''}
                            ${!model.hasArtifact ? '<small class="text-warning d-block mt-1"><i class="bi bi-info-circle me-1"></i>Conversion may fail - artifact record missing</small>' : ''}
                        </div>
                        <span class="badge ${model.hasArtifact ? 'bg-success' : 'bg-warning'}">${model.model_id ? `ID: ${model.model_id}` : 'S3 Only'}</span>
                    </div>
                </div>
            </div>
        `).join('');
    }

    render() {
        return `
            <div class="min-vh-100 bg-light">
                <div class="container py-4">
                    <!-- Page Header -->
                    <div class="mb-4">
                        <h1 class="display-5 fw-bold mb-2">Model Conversion</h1>
                        <p class="text-muted">Export and optimize models for deployment</p>
                    </div>

                    <div class="row g-4 mb-4">
                        <!-- Select Model Card -->
                        <div class="col-lg-6">
                            <div class="card border-0 shadow-sm h-100">
                                <div class="card-header bg-white">
                                    <h5 class="mb-1 fw-bold">Select Model</h5>
                                    <p class="text-muted mb-0 small">Choose a trained model to convert</p>
                                </div>
                                <div class="card-body p-0">
                                    <div id="models-list" style="max-height: 60vh; overflow-y: auto; overflow-x: hidden; padding: 1rem; scroll-behavior: smooth;">
                                        <div class="text-center py-5">
                                            <div class="spinner-border text-primary" role="status">
                                                <span class="visually-hidden">Loading...</span>
                                            </div>
                                            <p class="text-muted mt-3">Loading models...</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Conversion Settings Card -->
                        <div class="col-lg-6">
                            <div class="card border-0 shadow-sm h-100">
                                <div class="card-header bg-white">
                                    <h5 class="mb-1 fw-bold">Conversion Settings</h5>
                                    <p class="text-muted mb-0 small">Configure export parameters</p>
                                </div>
                                <div class="card-body">
                                    <div class="mb-3">
                                        <label class="form-label fw-medium">Target Format</label>
                                        <select class="form-select" id="target-format">
                                            <option value="onnx" selected>ONNX</option>
                                            <option value="tensorrt">TensorRT</option>
                                        </select>
                                    </div>

                                    <div class="mb-4">
                                        <label class="form-label fw-medium">Precision</label>
                                        <select class="form-select" id="precision-select">
                                            <option value="fp32">FP32 (Full Precision)</option>
                                            <option value="fp16" selected>FP16 (Half Precision)</option>
                                            <option value="int8">INT8 (Quantized)</option>
                                        </select>
                                    </div>

                                    <button class="btn btn-primary w-100" onclick="window.currentPage.startConversion()">
                                        <i class="bi bi-lightning-charge-fill me-2"></i>Start Conversion
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Supported Formats Card -->
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-white">
                            <h5 class="mb-1 fw-bold">Supported Export Formats</h5>
                            <p class="text-muted mb-0 small">Choose the best format for your deployment target</p>
                        </div>
                        <div class="card-body">
                            <div class="row g-4">
                                <div class="col-md-6">
                                    <div class="card h-100 hover-shadow">
                                        <div class="card-body">
                                            <div class="d-flex align-items-center gap-2 mb-2">
                                                <span class="fs-2">🔄</span>
                                                <div>
                                                    <h6 class="fw-bold mb-0">ONNX</h6>
                                                    <span class="badge bg-primary small">Cross-platform</span>
                                                </div>
                                            </div>
                                            <p class="text-muted small mb-3">
                                                Universal model format for deployment across different frameworks and platforms
                                            </p>
                                            <ul class="small text-muted mb-0">
                                                <li>Works with TensorFlow, PyTorch, etc.</li>
                                                <li>Good compatibility and portability</li>
                                                <li>Standard inference performance</li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card h-100 hover-shadow">
                                        <div class="card-body">
                                            <div class="d-flex align-items-center gap-2 mb-2">
                                                <span class="fs-2">⚡</span>
                                                <div>
                                                    <h6 class="fw-bold mb-0">TensorRT</h6>
                                                    <span class="badge bg-success small">NVIDIA GPU</span>
                                                </div>
                                            </div>
                                            <p class="text-muted small mb-3">
                                                Optimized for NVIDIA GPUs with up to 5x faster inference performance
                                            </p>
                                            <ul class="small text-muted mb-0">
                                                <li>Maximum performance on NVIDIA GPUs</li>
                                                <li>FP32, FP16, INT8 precision support</li>
                                                <li>Ideal for production deployment</li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Conversion Success Modal -->
            <div class="modal fade" id="conversionSuccessModal" tabindex="-1" aria-labelledby="conversionSuccessModalLabel" aria-hidden="true">
                <div class="modal-dialog modal-dialog-centered">
                    <div class="modal-content">
                        <div class="modal-header bg-success text-white">
                            <h5 class="modal-title" id="conversionSuccessModalLabel">
                                <i class="bi bi-check-circle-fill me-2"></i>Conversion Started
                            </h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <p class="mb-0" id="conversionSuccessMessage"></p>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-success" data-bs-dismiss="modal">
                                <i class="bi bi-check me-1"></i>확인
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Conversion Error Modal -->
            <div class="modal fade" id="conversionErrorModal" tabindex="-1" aria-labelledby="conversionErrorModalLabel" aria-hidden="true">
                <div class="modal-dialog modal-dialog-centered">
                    <div class="modal-content">
                        <div class="modal-header bg-danger text-white">
                            <h5 class="modal-title" id="conversionErrorModalLabel">
                                <i class="bi bi-exclamation-triangle-fill me-2"></i>Conversion Failed
                            </h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <p id="conversionErrorMessage" class="mb-0"></p>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-danger" data-bs-dismiss="modal">
                                <i class="bi bi-x me-1"></i>확인
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    showConversionSuccessModal(format, jobId) {
        const modal = document.getElementById('conversionSuccessModal');
        const messageEl = document.getElementById('conversionSuccessMessage');
        
        if (messageEl) {
            messageEl.textContent = `${format} conversion started!`;
        }
        
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
    }

    showConversionErrorModal(errorMessage) {
        const modal = document.getElementById('conversionErrorModal');
        const messageEl = document.getElementById('conversionErrorMessage');
        
        if (messageEl) {
            messageEl.textContent = errorMessage.startsWith('Conversion failed:') ? errorMessage : errorMessage;
        }
        
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
    }

    showWarningModal(title, message) {
        return new Promise((resolve) => {
            const modalId = 'warningModal';
            let modalEl = document.getElementById(modalId);
            
            // Create modal if it doesn't exist
            if (!modalEl) {
                modalEl = document.createElement('div');
                modalEl.id = modalId;
                modalEl.className = 'modal fade';
                modalEl.setAttribute('tabindex', '-1');
                modalEl.innerHTML = `
                    <div class="modal-dialog modal-dialog-centered">
                        <div class="modal-content">
                            <div class="modal-header bg-warning">
                                <h5 class="modal-title">
                                    <i class="bi bi-exclamation-triangle-fill me-2"></i>${title}
                                </h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                            </div>
                            <div class="modal-body">
                                <p style="white-space: pre-line;">${message}</p>
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                                <button type="button" class="btn btn-warning" id="confirmWarningBtn">
                                    <i class="bi bi-check me-1"></i>Proceed
                                </button>
                            </div>
                        </div>
                    </div>
                `;
                document.body.appendChild(modalEl);
            }

            // Update content
            const titleEl = modalEl.querySelector('.modal-title');
            const messageEl = modalEl.querySelector('.modal-body p');
            if (titleEl) titleEl.innerHTML = `<i class="bi bi-exclamation-triangle-fill me-2"></i>${title}`;
            if (messageEl) messageEl.textContent = message;

            // Set up event listeners
            const confirmBtn = modalEl.querySelector('#confirmWarningBtn');
            const bsModal = new bootstrap.Modal(modalEl);
            
            const handleConfirm = () => {
                bsModal.hide();
                resolve(true);
            };
            
            const handleCancel = () => {
                bsModal.hide();
                resolve(false);
            };

            // Remove old listeners and add new ones
            confirmBtn.replaceWith(confirmBtn.cloneNode(true));
            const newConfirmBtn = modalEl.querySelector('#confirmWarningBtn');
            newConfirmBtn.addEventListener('click', handleConfirm);
            
            modalEl.addEventListener('hidden.bs.modal', () => {
                resolve(false);
            }, { once: true });

            bsModal.show();
        });
    }
}
