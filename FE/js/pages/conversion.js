// Conversion Page Component

class ConversionPage {
    constructor() {
        this.models = [];
    }

    async init() {
        await this.loadModels();
    }

    async loadModels() {
        try {
            const response = await window.apiService.get('/models');
            this.models = response;
            this.renderModelsSection();
        } catch (error) {
            console.error('Failed to load models:', error);
            this.models = [];
            this.renderModelsSection();
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
            <div class="card mb-3 hover-shadow cursor-pointer">
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <h6 class="fw-bold mb-2">${model.name}</h6>
                            <div class="d-flex gap-3 text-muted small">
                                <span><i class="bi bi-cpu me-1"></i>${model.framework || 'PyTorch'}</span>
                                <span><i class="bi bi-file-earmark me-1"></i>${model.size || 'N/A'}</span>
                                ${model.accuracy ? `<span class="text-success"><i class="bi bi-check-circle me-1"></i>Accuracy: ${model.accuracy}%</span>` : ''}
                            </div>
                        </div>
                        <span class="badge badge-success">${model.status}</span>
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
                                        <select class="form-select">
                                            <option selected>ONNX</option>
                                            <option>TensorRT</option>
                                            <option>OpenVINO</option>
                                            <option>CoreML</option>
                                        </select>
                                    </div>

                                    <div class="mb-3">
                                        <label class="form-label fw-medium">Optimization Level</label>
                                        <select class="form-select">
                                            <option>Speed (Max Performance)</option>
                                            <option selected>Balanced</option>
                                            <option>Size (Minimal Footprint)</option>
                                        </select>
                                    </div>

                                    <div class="mb-4">
                                        <label class="form-label fw-medium">Precision</label>
                                        <select class="form-select">
                                            <option>FP32 (Full Precision)</option>
                                            <option selected>FP16 (Half Precision)</option>
                                            <option>INT8 (Quantized)</option>
                                        </select>
                                    </div>

                                    <button class="btn btn-primary w-100">
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
                                <div class="col-md-6 col-lg-3">
                                    <div class="card h-100 hover-shadow">
                                        <div class="card-body">
                                            <div class="d-flex align-items-center gap-2 mb-2">
                                                <span class="fs-2">🔄</span>
                                                <div>
                                                    <h6 class="fw-bold mb-0">ONNX</h6>
                                                    <span class="badge bg-primary small">Recommended</span>
                                                </div>
                                            </div>
                                            <p class="text-muted small mb-3">Cross-platform model format for deployment</p>
                                            <button class="btn btn-outline-primary btn-sm w-100">
                                                <i class="bi bi-download me-1"></i> Export
                                            </button>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6 col-lg-3">
                                    <div class="card h-100 hover-shadow">
                                        <div class="card-body">
                                            <div class="d-flex align-items-center gap-2 mb-2">
                                                <span class="fs-2">⚡</span>
                                                <div>
                                                    <h6 class="fw-bold mb-0">TensorRT</h6>
                                                </div>
                                            </div>
                                            <p class="text-muted small mb-3">Optimized for NVIDIA GPUs, up to 5x faster inference</p>
                                            <button class="btn btn-outline-primary btn-sm w-100">
                                                <i class="bi bi-download me-1"></i> Export
                                            </button>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6 col-lg-3">
                                    <div class="card h-100 hover-shadow">
                                        <div class="card-body">
                                            <div class="d-flex align-items-center gap-2 mb-2">
                                                <span class="fs-2">🚀</span>
                                                <div>
                                                    <h6 class="fw-bold mb-0">OpenVINO</h6>
                                                </div>
                                            </div>
                                            <p class="text-muted small mb-3">Intel hardware acceleration support</p>
                                            <button class="btn btn-outline-primary btn-sm w-100">
                                                <i class="bi bi-download me-1"></i> Export
                                            </button>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6 col-lg-3">
                                    <div class="card h-100 hover-shadow">
                                        <div class="card-body">
                                            <div class="d-flex align-items-center gap-2 mb-2">
                                                <span class="fs-2">🍎</span>
                                                <div>
                                                    <h6 class="fw-bold mb-0">CoreML</h6>
                                                </div>
                                            </div>
                                            <p class="text-muted small mb-3">Apple devices optimization</p>
                                            <button class="btn btn-outline-primary btn-sm w-100">
                                                <i class="bi bi-download me-1"></i> Export
                                            </button>
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
