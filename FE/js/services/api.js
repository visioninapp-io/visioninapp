// API Service Layer - Handles all backend communications

const API_BASE_URL = 'http://localhost:8000/api/v1';

class APIService {
    constructor() {
        this.baseURL = API_BASE_URL;
        this.authToken = null;
    }

    // Set authentication token
    setAuthToken(token) {
        this.authToken = token;
        localStorage.setItem('authToken', token);
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
    }

    // Make HTTP request
    async request(endpoint, options = {}) {
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers,
        };

        if (this.getAuthToken()) {
            headers['Authorization'] = `Bearer ${this.getAuthToken()}`;
        }

        const config = {
            ...options,
            headers,
        };

        const url = `${this.baseURL}${endpoint}`;
        console.log(`API Request: ${options.method || 'GET'} ${url}`);

        try {
            const response = await fetch(url, config);

            console.log(`API Response: ${response.status} ${response.statusText}`);

            if (!response.ok) {
                let errorDetail;
                try {
                    const error = await response.json();
                    errorDetail = error.detail || `HTTP Error: ${response.status}`;
                } catch {
                    errorDetail = `HTTP Error: ${response.status} - ${response.statusText}`;
                }
                throw new Error(errorDetail);
            }

            const data = await response.json();
            console.log(`API Response Data:`, data);
            return data;
        } catch (error) {
            console.error('API Request Error:', error);
            console.error('URL:', url);
            console.error('Config:', config);
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

    // Upload file
    async upload(endpoint, formData) {
        const headers = {};
        if (this.getAuthToken()) {
            headers['Authorization'] = `Bearer ${this.getAuthToken()}`;
        }

        const response = await fetch(`${this.baseURL}${endpoint}`, {
            method: 'POST',
            headers,
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `Upload Error: ${response.status}`);
        }

        return await response.json();
    }

    // ========== DATASETS ==========
    async getDatasets() {
        return this.get('/datasets/');  // Add trailing slash to avoid 307 redirect
    }

    async getDataset(id) {
        return this.get(`/datasets/${id}`);
    }

    async createDataset(data) {
        return this.post('/datasets', data);
    }

    async uploadDataset(formData) {
        return this.upload('/datasets/upload', formData);
    }

    async autoAnnotate(datasetId, modelId) {
        return this.post(`/datasets/${datasetId}/auto-annotate`, { model_id: modelId });
    }

    async getDatasetStats() {
        return this.get('/datasets/stats');  // No trailing slash for /stats
    }

    async getDatasetImages(datasetId) {
        return this.get(`/datasets/${datasetId}/images`);  // No trailing slash for images
    }

    // ========== MODELS ==========
    async getModels() {
        return this.get('/models');
    }

    async getModel(id) {
        return this.get(`/models/${id}`);
    }

    async createModel(data) {
        return this.post('/models', data);
    }

    async convertModel(modelId, targetFramework, optimizationLevel, precision) {
        return this.post(`/models/${modelId}/convert`, {
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
        return this.get('/training/');  // Add trailing slash to avoid 307 redirect
    }

    async getTrainingJob(id) {
        return this.get(`/training/${id}`);
    }

    async startTraining(data) {
        return this.post('/training/', data);  // Add trailing slash
    }

    async pauseTraining(id) {
        return this.post(`/training/${id}/pause`);
    }

    async stopTraining(id) {
        return this.post(`/training/${id}/stop`);
    }

    async getTrainingMetrics(id) {
        return this.get(`/training/${id}/metrics`);
    }

    // ========== EVALUATION ==========
    async getEvaluations() {
        return this.get('/evaluation/');  // Add trailing slash
    }

    async evaluateModel(data) {
        return this.post('/evaluation/', data);  // Add trailing slash
    }

    async getEvaluation(id) {
        return this.get(`/evaluation/${id}`);
    }

    // ========== DEPLOYMENT ==========
    async getDeployments() {
        return this.get('/deployment/');  // Add trailing slash
    }

    async createDeployment(data) {
        return this.post('/deployment/', data);  // Add trailing slash
    }

    async getDeployment(id) {
        return this.get(`/deployment/${id}`);
    }

    async deleteDeployment(id) {
        return this.delete(`/deployment/${id}`);
    }

    // ========== MONITORING ==========
    async getMonitoring() {
        return this.get('/monitoring/');  // Add trailing slash
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
}

// Create singleton instance
const apiService = new APIService();
