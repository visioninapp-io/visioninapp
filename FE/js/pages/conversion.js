// Conversion Page Component

class ConversionPage {
    constructor() {
        this.models = [];
        this.selectedModelId = null;
        this.selectedS3Uri = null;
    }

    async init() {
        await this.loadModels();
        // 전역에서 접근 가능하도록 설정
        window.currentPage = this;
    }

    async loadModels() {
        try {
            const response = await window.apiService.getModels();
            this.models = response;
            
            // 각 모델의 artifact 정보 가져오기
            for (let model of this.models) {
                try {
                    const artifacts = await window.apiService.getModelArtifacts(model.id);
                    if (artifacts && artifacts.length > 0) {
                        // PT 파일 찾기
                        const ptArtifact = artifacts.find(a => a.format === 'pt' || a.storage_uri?.endsWith('.pt'));
                        if (ptArtifact) {
                            model.s3_uri = `s3://visioninapp-bucket/${ptArtifact.storage_uri}`;
                        }
                    }
                } catch (error) {
                    console.warn(`Failed to load artifacts for model ${model.id}:`, error);
                }
            }
            
            this.renderModelsSection();
        } catch (error) {
            console.error('Failed to load models:', error);
            this.models = [];
            this.renderModelsSection();
        }
    }

    selectModel(modelId, s3Uri) {
        if (!s3Uri) {
            alert('This model does not have a valid artifact. Please ensure the model has been trained.');
            return;
        }
        
        this.selectedModelId = modelId;
        this.selectedS3Uri = s3Uri;
        
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
            alert('Please select a model first');
            return;
        }

        try {
            const formatSelect = document.getElementById('target-format');
            const precisionSelect = document.getElementById('precision-select');
            
            if (!formatSelect || !precisionSelect) {
                alert('Please wait for the page to fully load');
                return;
            }
            
            const format = formatSelect.value;
            const precision = precisionSelect.value;
            const timestamp = Date.now();

            const payload = {
                model: {
                    s3_uri: this.selectedS3Uri
                },
                output: {
                    prefix: `exports/${format}/${timestamp}`
                }
            };

            let result;
            if (format === 'onnx') {
                payload.output.model_name = 'model.onnx';
                payload.ops = {
                    dynamic: true,
                    simplify: true,
                    opset: 13,
                    imgsz: 640,
                    precision: precision  // fp32, fp16, int8
                };
                result = await window.apiService.convertToOnnx(payload);
                console.log('ONNX conversion started:', result);
            } else if (format === 'tensorrt') {
                payload.output.model_name = 'model.engine';
                payload.trt = {
                    precision: precision,
                    imgsz: 640,
                    dynamic: true
                };
                result = await window.apiService.convertToTensorRT(payload);
                console.log('TensorRT conversion started:', result);
            } else {
                alert('Unsupported format. Please select ONNX or TensorRT.');
                return;
            }

            alert(`${format.toUpperCase()} conversion started!\nJob ID: ${result.job_id || 'N/A'}`);
        } catch (error) {
            console.error('Conversion failed:', error);
            alert('Conversion failed: ' + (error.message || 'Unknown error'));
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
            <div class="card mb-3 hover-shadow cursor-pointer model-card" 
                 data-model-id="${model.id}"
                 onclick="window.currentPage.selectModel(${model.id}, '${model.s3_uri || ''}')">
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <h6 class="fw-bold mb-2">${model.name}</h6>
                            <div class="d-flex gap-3 text-muted small">
                                <span><i class="bi bi-cpu me-1"></i>PyTorch</span>
                                ${model.s3_uri ? '<span class="text-success"><i class="bi bi-check-circle me-1"></i>Ready</span>' : '<span class="text-warning"><i class="bi bi-exclamation-triangle me-1"></i>No artifact</span>'}
                            </div>
                        </div>
                        <span class="badge bg-secondary">ID: ${model.id}</span>
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
                                <div class="card-body" id="models-list">
                                    <div class="text-center py-5">
                                        <div class="spinner-border text-primary" role="status">
                                            <span class="visually-hidden">Loading...</span>
                                        </div>
                                        <p class="text-muted mt-3">Loading models...</p>
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
        `;
    }
}
