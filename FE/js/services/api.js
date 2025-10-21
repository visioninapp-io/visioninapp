// API Service Layer - Handles all backend communications
// Enhanced with robust error handling and debugging

const API_BASE_URL = 'http://localhost:8000/api/v1';

class APIService {
    constructor() {
        this.baseURL = API_BASE_URL;
        this.authToken = null;
        this.devMode = true; // Development mode - works without Firebase auth
        console.log('[API Service] Initialized with base URL:', this.baseURL);
        console.log('[API Service] Dev mode:', this.devMode);
    }

    // Set authentication token
    setAuthToken(token) {
        this.authToken = token;
        if (token) {
            localStorage.setItem('authToken', token);
            console.log('[API Service] Auth token set');
        }
    }

    // Get authentication token
    getAuthToken() {
        if (!this.authToken) {
            this.authToken = localStorage.getItem('authToken');
        }
        return this.authToken;
    }

    // Clear authentication
    clearAuth() {
        this.authToken = null;
        localStorage.removeItem('authToken');
        console.log('[API Service] Auth cleared');
    }

    // Make HTTP request with enhanced error handling
    async request(endpoint, options = {}) {
        const headers = {
            ...options.headers,
        };

        // Only set Content-Type for non-FormData requests
        if (!(options.body instanceof FormData)) {
            headers['Content-Type'] = 'application/json';
        }

        // Add auth token if available (optional in dev mode)
        const token = this.getAuthToken();
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        const config = {
            ...options,
            headers,
        };

        const url = `${this.baseURL}${endpoint}`;
        console.log(`[API Request] ${options.method || 'GET'} ${url}`);

        if (token) {
            console.log('[API Request] Auth token present:', token.substring(0, 20) + '...');
        } else {
            console.log('[API Request] No auth token (dev mode)');
        }

        try {
            const response = await fetch(url, config);

            console.log(`[API Response] Status: ${response.status} ${response.statusText}`);

            // Handle different response statuses
            if (!response.ok) {
                let errorDetail;
                let errorBody;

                try {
                    errorBody = await response.json();
                    errorDetail = errorBody.detail || `HTTP Error: ${response.status}`;
                    console.error('[API Error Response]', errorBody);
                } catch {
                    errorDetail = `HTTP Error: ${response.status} - ${response.statusText}`;
                }

                // Show toast for errors
                if (typeof showToast === 'function') {
                    showToast(errorDetail, 'error');
                }

                throw new Error(errorDetail);
            }

            // Check status code first for No Content responses
            if (response.status === 204 || response.status === 205) {
                // No content - return null immediately without trying to parse
                console.log('[API Response] No content (204/205)');
                return null;
            }

            // Check if response has content
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                const data = await response.json();
                console.log('[API Response Data]', data);
                return data;
            } else {
                const text = await response.text();
                console.log('[API Response Text]', text);
                return text;
            }

        } catch (error) {
            console.error('[API Request Error]', {
                url,
                method: options.method || 'GET',
                error: error.message,
                stack: error.stack
            });

            // Show connection error toast
            if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                if (typeof showToast === 'function') {
                    showToast('Cannot connect to server. Please check if the backend is running.', 'error');
                }
            }

            throw error;
        }
    }

    // GET request
    async get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    }

    // POST request
    async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    // PUT request
    async put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data),
        });
    }

    // DELETE request
    async delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }

    // Upload file with FormData
    async upload(endpoint, formData) {
        const headers = {};
        const token = this.getAuthToken();
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        const url = `${this.baseURL}${endpoint}`;
        console.log(`[API Upload] POST ${url}`);
        console.log('[API Upload] FormData entries:', Array.from(formData.entries()).map(([key, value]) => {
            if (value instanceof File) {
                return `${key}: File(${value.name}, ${value.size} bytes)`;
            }
            return `${key}: ${value}`;
        }));

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers,
                body: formData,
            });

            console.log(`[API Upload Response] Status: ${response.status}`);

            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: `Upload Error: ${response.status}` }));
                console.error('[API Upload Error]', error);
                throw new Error(error.detail || `Upload Error: ${response.status}`);
            }

            const data = await response.json();
            console.log('[API Upload Success]', data);
            return data;

        } catch (error) {
            console.error('[API Upload Error]', error);
            throw error;
        }
    }

    // ========== DATASETS ==========
    async getDatasets() {
        console.log('[API] Fetching datasets...');
        try {
            const data = await this.get('/datasets/');
            console.log(`[API] Fetched ${data.length} datasets`);
            return data;
        } catch (error) {
            console.error('[API] Failed to fetch datasets:', error);
            // Return empty array to prevent UI crash
            return [];
        }
    }

    async getDataset(id) {
        console.log(`[API] Fetching dataset ${id}...`);
        return this.get(`/datasets/${id}`);
    }

    async createDataset(data) {
        console.log('[API] Creating dataset:', data);
        return this.post('/datasets/', data);
    }

    async uploadDataset(formData) {
        console.log('[API] Uploading dataset...');
        return this.upload('/datasets/upload', formData);
    }

    async autoAnnotate(datasetId, modelId = null, confidenceThreshold = 0.5, overwriteExisting = false) {
        console.log(`[API] Starting auto-annotation for dataset ${datasetId}...`);
        console.log(`[API] Overwrite existing: ${overwriteExisting}`);
        const payload = {
            dataset_id: datasetId,
            confidence_threshold: confidenceThreshold,
            overwrite_existing: overwriteExisting
        };

        if (modelId) {
            payload.model_id = modelId;
        }

        return this.post('/datasets/auto-annotate', payload);
    }

    async getDatasetStats() {
        console.log('[API] Fetching dataset stats...');
        try {
            const data = await this.get('/datasets/stats');
            console.log('[API] Dataset stats:', data);
            return data;
        } catch (error) {
            console.error('[API] Failed to fetch dataset stats:', error);
            // Return default stats
            return {
                total_images: 0,
                total_datasets: 0,
                total_classes: 0,
                auto_annotation_rate: 0
            };
        }
    }

    async getDatasetImages(datasetId) {
        console.log(`[API] Fetching images for dataset ${datasetId}...`);
        try {
            const data = await this.get(`/datasets/${datasetId}/images`);
            console.log(`[API] Fetched ${data.length} images`);
            return data;
        } catch (error) {
            console.error(`[API] Failed to fetch images for dataset ${datasetId}:`, error);
            return [];
        }
    }

    async getImageAnnotations(imageId, minConfidence = null) {
        console.log(`[API] Fetching annotations for image ${imageId}...`);
        if (minConfidence !== null) {
            console.log(`[API] With min confidence: ${minConfidence}`);
        }
        try {
            let endpoint = `/datasets/images/${imageId}/annotations`;
            if (minConfidence !== null) {
                endpoint += `?min_confidence=${minConfidence}`;
            }
            const data = await this.get(endpoint);
            console.log(`[API] Fetched ${data.length} annotations`);
            return data;
        } catch (error) {
            console.error(`[API] Failed to fetch annotations for image ${imageId}:`, error);
            return [];
        }
    }

    // ========== MODELS ==========
    async getModels() {
        console.log('[API] Fetching models...');
        return this.get('/models/');
    }

    async getModel(id) {
        return this.get(`/models/${id}`);
    }

    async createModel(data) {
        console.log('[API] Creating model:', data);
        return this.post('/models/', data);
    }

    async convertModel(modelId, targetFramework, optimizationLevel, precision) {
        return this.post(`/models/convert`, {
            model_id: modelId,
            target_framework: targetFramework,
            optimization_level: optimizationLevel,
            precision: precision
        });
    }

    async exportModel(modelId, format) {
        const response = await fetch(`${this.baseURL}/models/${modelId}/export?format=${format}`, {
            headers: {
                'Authorization': `Bearer ${this.getAuthToken()}`
            }
        });
        return response.blob();
    }

    // ========== TRAINING ==========
    async getTrainingJobs() {
        console.log('[API] Fetching training jobs...');
        try {
            const data = await this.get('/training/');
            console.log(`[API] Fetched ${data.length} training jobs`);
            return data;
        } catch (error) {
            console.error('[API] Failed to fetch training jobs:', error);
            return [];
        }
    }

    async getTrainingJob(id) {
        return this.get(`/training/${id}`);
    }

    async startTraining(data) {
        console.log('[API] Starting training:', data);
        return this.post('/training/', data);
    }

    async pauseTraining(id) {
        return this.post(`/training/${id}/control`, { action: 'pause' });
    }

    async stopTraining(id) {
        return this.post(`/training/${id}/control`, { action: 'cancel' });
    }

    async getTrainingMetrics(id) {
        console.log(`[API] Fetching training metrics for job ${id}...`);
        try {
            const data = await this.get(`/training/${id}/metrics`);
            return data;
        } catch (error) {
            console.error(`[API] Failed to fetch training metrics for job ${id}:`, error);
            return [];
        }
    }

    // ========== EVALUATION ==========
    async getEvaluations() {
        return this.get('/evaluation/');
    }

    async evaluateModel(data) {
        return this.post('/evaluation/', data);
    }

    async getEvaluation(id) {
        return this.get(`/evaluation/${id}`);
    }

    // ========== DEPLOYMENT ==========
    async getDeployments() {
        return this.get('/deployment/');
    }

    async createDeployment(data) {
        return this.post('/deployment/', data);
    }

    async getDeployment(id) {
        return this.get(`/deployment/${id}`);
    }

    async deleteDeployment(id) {
        return this.delete(`/deployment/${id}`);
    }

    // ========== MONITORING ==========
    async getMonitoring() {
        return this.get('/monitoring/');
    }

    async getMonitoringDashboard(deploymentId) {
        return this.get(`/monitoring/dashboard/${deploymentId}`);
    }

    async getAlerts() {
        return this.get('/monitoring/alerts');
    }

    async acknowledgeAlert(alertId) {
        return this.post(`/monitoring/alerts/${alertId}/acknowledge`);
    }

    async triggerRetraining(deploymentId) {
        return this.post(`/monitoring/feedback-loops/${deploymentId}/trigger-retrain`);
    }

    async getPerformanceMetrics(deploymentId) {
        return this.get(`/monitoring/deployments/${deploymentId}/metrics`);
    }

    // ========== VERSIONS ==========
    async getDatasetVersions(datasetId) {
        console.log(`[API] Fetching versions for dataset ${datasetId}...`);
        return this.get(`/datasets/${datasetId}/versions`);
    }

    async getVersion(versionId) {
        return this.get(`/datasets/versions/${versionId}`);
    }

    async createVersion(datasetId, versionData) {
        console.log('[API] Creating dataset version:', versionData);
        return this.post(`/datasets/${datasetId}/versions`, versionData);
    }

    async deleteVersion(versionId) {
        return this.delete(`/datasets/versions/${versionId}`);
    }

    // ========== EXPORT ==========
    async getExportJobs() {
        console.log('[API] Fetching export jobs...');
        return this.get('/export/');
    }

    async getExportJob(exportId) {
        return this.get(`/export/${exportId}`);
    }

    async createExport(exportData) {
        console.log('[API] Creating export job:', exportData);
        return this.post('/export/', exportData);
    }

    async downloadExport(exportId) {
        const url = `${this.baseURL}/export/${exportId}/download`;
        window.open(url, '_blank');
    }

    async deleteExport(exportId) {
        return this.delete(`/export/${exportId}`);
    }
}

// Create singleton instance and make it globally accessible
const apiService = new APIService();
window.apiService = apiService;

console.log('[API Service] Service ready and globally accessible');
