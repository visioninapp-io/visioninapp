// API Service Layer - Handles all backend communications
// Enhanced with robust error handling and debugging

const API_BASE_URL = '/api/v1';

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

    async updateDataset(id, data) {
        console.log(`[API] Updating dataset ${id}:`, data);
        return this.put(`/datasets/${id}`, data);
    }

    async deleteDataset(id) {
        console.log(`[API] Deleting dataset ${id}...`);
        return this.delete(`/datasets/${id}`);
    }

    async uploadDataset(formData) {
        console.log('[API] Uploading dataset...');
        console.log('[API] FormData type:', typeof formData, formData);

        // Check if formData is valid
        if (!formData || typeof formData.get !== 'function') {
            console.error('[API] Invalid FormData object:', formData);
            throw new Error('Invalid FormData object');
        }

        try {
            // Extract dataset_id and name from FormData
            const datasetId = formData.get('dataset_id');
            const datasetName = formData.get('name');
            const datasetDescription = formData.get('description');

            console.log('[API] Upload params:', { datasetId, datasetName, datasetDescription });

            // Get files from FormData
            const files = formData.getAll('files');
            console.log('[API] Files count:', files.length);

            if (files.length === 0) {
                throw new Error('No files provided');
            }

            let targetDatasetId = datasetId;

            // If no dataset_id, create a new dataset first
            if (!datasetId || datasetId === '') {
                if (!datasetName) {
                    throw new Error('Dataset name is required for new dataset');
                }

                console.log('[API] Creating new dataset:', datasetName);
                const newDataset = await this.createDataset({
                    name: datasetName,
                    description: datasetDescription || ''
                });
                targetDatasetId = newDataset.id;
                console.log('[API] Created dataset with ID:', targetDatasetId);
            }

            // Now upload files to the dataset
            console.log(`[API] Uploading ${files.length} files to dataset ${targetDatasetId}...`);

            // Create new FormData with just the files
            const uploadFormData = new FormData();
            files.forEach(file => {
                uploadFormData.append('files', file);
            });

            try {
                // Backend expects POST /datasets/{dataset_id}/images/upload
                const result = await this.upload(`/datasets/${targetDatasetId}/images/upload`, uploadFormData);

                console.log('[API] Upload response:', result);

                // Add dataset_id to result for frontend compatibility
                if (result && !result.dataset_id) {
                    result.dataset_id = targetDatasetId;
                }

                return result;

            } catch (uploadError) {
                console.error('[API] Upload to backend failed:', uploadError);
                console.error('[API] Error details:', {
                    message: uploadError.message,
                    stack: uploadError.stack
                });

                // More detailed error for user
                throw new Error(`Upload failed: ${uploadError.message}. Check browser console for details.`);
            }

        } catch (error) {
            console.error('[API] Upload process failed:', error);
            throw error;
        }
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

            // BE returns: { dataset_id, page, images: [...], total_images }
            // Extract the images array
            if (data && data.images && Array.isArray(data.images)) {
                console.log(`[API] Fetched ${data.images.length} images`);
                return data.images;
            } else if (Array.isArray(data)) {
                // Fallback: if BE returns array directly
                console.log(`[API] Fetched ${data.length} images`);
                return data;
            } else {
                console.warn('[API] Unexpected response format:', data);
                return [];
            }
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

    async createAnnotation(annotationData) {
        console.log('[API] Creating annotation:', annotationData);
        return this.post('/datasets/annotations', annotationData);
    }

    async deleteAnnotation(annotationId) {
        console.log(`[API] Deleting annotation ${annotationId}...`);
        return this.delete(`/datasets/annotations/${annotationId}`);
    }

    async updateAnnotation(annotationId, annotationData) {
        console.log(`[API] Updating annotation ${annotationId}:`, annotationData);
        return this.put(`/datasets/annotations/${annotationId}`, annotationData);
    }

    async getDatasetAssets(datasetId) {
        console.log(`[API] Fetching assets for dataset ${datasetId}...`);
        return this.get(`/datasets/${datasetId}/assets`);
    }

    async getPresignedUploadUrls(data) {
        console.log('[API] Getting presigned upload URLs...');
        return this.post('/datasets/presigned-upload-urls', data);
    }

    async uploadCompleteBatch(datasetId, data) {
        console.log(`[API] Completing batch upload for dataset ${datasetId}...`);
        return this.post(`/datasets/${datasetId}/upload-complete-batch`, data);
    }

    async getPresignedDownloadUrlsBatch(datasetId, assetIds) {
        console.log(`[API] Getting presigned download URLs for dataset ${datasetId}...`);
        return this.post(`/datasets/${datasetId}/presigned-download-urls-batch`, { asset_ids: assetIds });
    }

    async downloadDatasetImage(datasetId, imageId) {
        console.log(`[API] Downloading image ${imageId} from dataset ${datasetId}...`);
        return this.get(`/datasets/${datasetId}/images/${imageId}/download`);
    }

    async downloadDataset(datasetId) {
        console.log(`[API] Downloading dataset ${datasetId}...`);
        return this.get(`/datasets/${datasetId}/download`);
    }

    async getDatasetLabelClasses(datasetId) {
        console.log(`[API] Fetching label classes for dataset ${datasetId}...`);
        return this.get(`/datasets/${datasetId}/label-classes`);
    }

    // ========== MODELS ==========
    async getModels() {
        console.log('[API] Fetching models...');
        return this.get('/models/');
    }

    async getTrainedModels() {
        console.log('[API] Fetching trained models...');
        return this.get('/models/trained');
    }

    async getModel(id) {
        return this.get(`/models/${id}`);
    }

    async createModel(data) {
        console.log('[API] Creating model:', data);
        return this.post('/models/', data);
    }

    async updateModel(id, data) {
        console.log(`[API] Updating model ${id}:`, data);
        return this.put(`/models/${id}`, data);
    }

    async deleteModel(id) {
        console.log(`[API] Deleting model ${id}...`);
        return this.delete(`/models/${id}`);
    }

    async getModelPresignedUpload(modelId, data) {
        console.log(`[API] Getting presigned upload URL for model ${modelId}...`);
        return this.post(`/models/${modelId}/presigned-upload`, data);
    }

    async modelUploadComplete(modelId, data) {
        console.log(`[API] Completing upload for model ${modelId}...`);
        return this.post(`/models/${modelId}/upload-complete`, data);
    }

    async getModelArtifacts(modelId) {
        console.log(`[API] Fetching artifacts for model ${modelId}...`);
        return this.get(`/models/${modelId}/artifacts`);
    }

    async getArtifactPresignedDownload(artifactId) {
        console.log(`[API] Getting presigned download URL for artifact ${artifactId}...`);
        return this.get(`/models/artifacts/${artifactId}/presigned-download`);
    }

    async predictModel(modelId, data) {
        console.log(`[API] Running prediction with model ${modelId}...`);
        return this.post(`/models/${modelId}/predict`, data);
    }

    async convertModel(modelId, targetFramework, optimizationLevel, precision) {
        // NOTE: Backend does not have a /models/convert endpoint
        // Model conversion should be handled differently or via training/export
        console.warn('[API] Model conversion endpoint not available in backend');
        throw new Error('Model conversion not implemented in backend');
    }

    async exportModel(modelId, format) {
        // Backend uses /models/{model_id}/download for model download
        // This returns a presigned URL
        console.log(`[API] Exporting model ${modelId}...`);
        try {
            const data = await this.post(`/models/${modelId}/download`);

            // If backend returns presigned URL, open it
            if (data.download_url) {
                window.open(data.download_url, '_blank');
                return data;
            }

            // Fallback: old behavior (if backend returns blob)
            const response = await fetch(`${this.baseURL}/models/${modelId}/download`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.getAuthToken()}`
                }
            });
            return response.blob();
        } catch (error) {
            console.error(`[API] Failed to export model ${modelId}:`, error);
            throw error;
        }
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

    async updateTrainingJob(id, data) {
        console.log(`[API] Updating training job ${id}:`, data);
        return this.put(`/training/${id}`, data);
    }

    async pauseTraining(id) {
        return this.post(`/training/${id}/control`, { action: 'pause' });
    }

    async stopTraining(id) {
        return this.post(`/training/${id}/control`, { action: 'cancel' });
    }

    async hyperparameterTuning(data) {
        console.log('[API] Starting hyperparameter tuning:', data);
        return this.post('/training/hyperparameter-tuning', data);
    }

    async getTrainingMetrics(id) {
        console.log(`[API] Fetching training metrics for job ${id}...`);
        try {
            // Backend uses /training/{job_id}/progress for metrics
            const data = await this.get(`/training/${id}/progress`);
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

    async getModelVersionEvaluations(modelVersionId) {
        console.log(`[API] Fetching evaluations for model version ${modelVersionId}...`);
        return this.get(`/evaluation/model-version/${modelVersionId}`);
    }

    async getLatestModelVersionEvaluation(modelVersionId) {
        console.log(`[API] Fetching latest evaluation for model version ${modelVersionId}...`);
        return this.get(`/evaluation/model-version/${modelVersionId}/latest`);
    }

    async compareEvaluations(data) {
        console.log('[API] Comparing evaluations:', data);
        return this.post('/evaluation/compare', data);
    }

    async deleteEvaluation(id) {
        console.log(`[API] Deleting evaluation ${id}...`);
        return this.delete(`/evaluation/${id}`);
    }

    // ========== DEPLOYMENT ==========
    async getDeploymentStats() {
        console.log('[API] Fetching deployment stats...');
        return this.get('/deployment/stats');
    }

    async getDeployments() {
        return this.get('/deployment/');
    }

    async createDeployment(data) {
        return this.post('/deployment/', data);
    }

    async getDeployment(id) {
        return this.get(`/deployment/${id}`);
    }

    async updateDeployment(id, data) {
        console.log(`[API] Updating deployment ${id}:`, data);
        return this.put(`/deployment/${id}`, data);
    }

    async deleteDeployment(id) {
        return this.delete(`/deployment/${id}`);
    }

    async deploymentInference(deploymentId, data) {
        console.log(`[API] Running inference on deployment ${deploymentId}...`);
        return this.post(`/deployment/${deploymentId}/inference`, data);
    }

    async deploymentHealthCheck(deploymentId) {
        console.log(`[API] Checking health of deployment ${deploymentId}...`);
        return this.post(`/deployment/${deploymentId}/health-check`);
    }

    async startDeployment(deploymentId) {
        console.log(`[API] Starting deployment ${deploymentId}...`);
        return this.post(`/deployment/${deploymentId}/start`);
    }

    async stopDeployment(deploymentId) {
        console.log(`[API] Stopping deployment ${deploymentId}...`);
        return this.post(`/deployment/${deploymentId}/stop`);
    }

    // ========== MONITORING ==========
    // NOTE: Backend does not have /monitoring endpoints implemented yet
    // These methods are placeholders for future implementation
    async getMonitoring() {
        console.warn('[API] Monitoring endpoints not implemented in backend');
        // return this.get('/monitoring/');
        return [];
    }

    async getMonitoringDashboard(deploymentId) {
        console.warn('[API] Monitoring endpoints not implemented in backend');
        // return this.get(`/monitoring/dashboard/${deploymentId}`);
        return null;
    }

    async getAlerts() {
        console.warn('[API] Monitoring endpoints not implemented in backend');
        // return this.get('/monitoring/alerts');
        return [];
    }

    async acknowledgeAlert(alertId) {
        console.warn('[API] Monitoring endpoints not implemented in backend');
        // return this.post(`/monitoring/alerts/${alertId}/acknowledge`);
        throw new Error('Monitoring not implemented');
    }

    async triggerRetraining(deploymentId) {
        console.warn('[API] Monitoring endpoints not implemented in backend');
        // return this.post(`/monitoring/feedback-loops/${deploymentId}/trigger-retrain`);
        throw new Error('Monitoring not implemented');
    }

    async getPerformanceMetrics(deploymentId) {
        console.warn('[API] Monitoring endpoints not implemented in backend');
        // return this.get(`/monitoring/deployments/${deploymentId}/metrics`);
        return null;
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

    async updateVersion(versionId, versionData) {
        console.log(`[API] Updating version ${versionId}:`, versionData);
        return this.put(`/datasets/versions/${versionId}`, versionData);
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
