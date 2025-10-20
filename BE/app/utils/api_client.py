"""
Frontend API Client Helper
JavaScript code for frontend to connect to the backend API
"""

FRONTEND_API_CLIENT = """
// VisionAI Platform API Client for Frontend
// Place this in your frontend project

const API_BASE_URL = 'http://localhost:8000/api/v1';

class VisionAIClient {
  constructor(baseURL = API_BASE_URL) {
    this.baseURL = baseURL;
    this.token = null;
  }

  // Set Firebase token for authentication
  setToken(token) {
    this.token = token;
  }

  // Make authenticated API request
  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    const config = {
      ...options,
      headers,
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'API request failed');
      }

      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }

  // Datasets API
  async getDatasetStats() {
    return this.request('/datasets/stats');
  }

  async listDatasets(skip = 0, limit = 100) {
    return this.request(`/datasets?skip=${skip}&limit=${limit}`);
  }

  async createDataset(data) {
    return this.request('/datasets', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getDataset(id) {
    return this.request(`/datasets/${id}`);
  }

  async updateDataset(id, data) {
    return this.request(`/datasets/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteDataset(id) {
    return this.request(`/datasets/${id}`, {
      method: 'DELETE',
    });
  }

  async uploadImages(datasetId, files) {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    return this.request(`/datasets/${datasetId}/images/upload`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.token}`,
      },
      body: formData,
    });
  }

  async autoAnnotate(datasetId, modelId, confidenceThreshold = 0.5) {
    return this.request('/datasets/auto-annotate', {
      method: 'POST',
      body: JSON.stringify({
        dataset_id: datasetId,
        model_id: modelId,
        confidence_threshold: confidenceThreshold,
      }),
    });
  }

  // Training API
  async listTrainingJobs(statusFilter = null) {
    const url = statusFilter
      ? `/training?status_filter=${statusFilter}`
      : '/training';
    return this.request(url);
  }

  async createTrainingJob(data) {
    return this.request('/training', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getTrainingJob(id) {
    return this.request(`/training/${id}`);
  }

  async getTrainingProgress(id) {
    return this.request(`/training/${id}/progress`);
  }

  async controlTraining(id, action) {
    return this.request(`/training/${id}/control`, {
      method: 'POST',
      body: JSON.stringify({ action }),
    });
  }

  async getTrainingMetrics(id) {
    return this.request(`/training/${id}/metrics`);
  }

  // Models API
  async listModels(framework = null, statusFilter = null) {
    let url = '/models?';
    if (framework) url += `framework=${framework}&`;
    if (statusFilter) url += `status_filter=${statusFilter}`;
    return this.request(url);
  }

  async getModel(id) {
    return this.request(`/models/${id}`);
  }

  async convertModel(sourceModelId, targetFramework, optimizationLevel = 'balanced', precision = 'FP16') {
    return this.request('/models/convert', {
      method: 'POST',
      body: JSON.stringify({
        source_model_id: sourceModelId,
        target_framework: targetFramework,
        optimization_level: optimizationLevel,
        precision: precision,
      }),
    });
  }

  async getConversionStatus(conversionId) {
    return this.request(`/models/convert/${conversionId}`);
  }

  // Evaluation API
  async listEvaluations(modelId = null) {
    const url = modelId ? `/evaluation?model_id=${modelId}` : '/evaluation';
    return this.request(url);
  }

  async createEvaluation(data) {
    return this.request('/evaluation', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getModelEvaluations(modelId) {
    return this.request(`/evaluation/model/${modelId}`);
  }

  async compareModels(modelIds) {
    return this.request('/evaluation/compare', {
      method: 'POST',
      body: JSON.stringify(modelIds),
    });
  }

  // Deployment API
  async getDeploymentStats() {
    return this.request('/deployment/stats');
  }

  async listDeployments(statusFilter = null) {
    const url = statusFilter
      ? `/deployment?status_filter=${statusFilter}`
      : '/deployment';
    return this.request(url);
  }

  async createDeployment(data) {
    return this.request('/deployment', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getDeployment(id) {
    return this.request(`/deployment/${id}`);
  }

  async runInference(deploymentId, imageData, confidenceThreshold = 0.5) {
    return this.request(`/deployment/${deploymentId}/inference`, {
      method: 'POST',
      body: JSON.stringify({
        image: imageData,
        confidence_threshold: confidenceThreshold,
      }),
    });
  }

  async getInferenceLogs(deploymentId, skip = 0, limit = 100) {
    return this.request(`/deployment/${deploymentId}/logs?skip=${skip}&limit=${limit}`);
  }

  async healthCheck(deploymentId) {
    return this.request(`/deployment/${deploymentId}/health-check`, {
      method: 'POST',
    });
  }

  // Monitoring API
  async getMonitoringDashboard() {
    return this.request('/monitoring/dashboard');
  }

  async listAlerts(statusFilter = null) {
    const url = statusFilter
      ? `/monitoring/alerts?status_filter=${statusFilter}`
      : '/monitoring/alerts';
    return this.request(url);
  }

  async acknowledgeAlert(alertId) {
    return this.request(`/monitoring/alerts/${alertId}/acknowledge`, {
      method: 'POST',
    });
  }

  async resolveAlert(alertId) {
    return this.request(`/monitoring/alerts/${alertId}/resolve`, {
      method: 'POST',
    });
  }

  async getPerformanceMetrics(deploymentId = null, hours = 24) {
    let url = `/monitoring/metrics?hours=${hours}`;
    if (deploymentId) url += `&deployment_id=${deploymentId}`;
    return this.request(url);
  }

  async listFeedbackLoops(deploymentId = null) {
    const url = deploymentId
      ? `/monitoring/feedback-loops?deployment_id=${deploymentId}`
      : '/monitoring/feedback-loops';
    return this.request(url);
  }

  async createFeedbackLoop(data) {
    return this.request('/monitoring/feedback-loops', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateFeedbackLoop(id, data) {
    return this.request(`/monitoring/feedback-loops/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async listEdgeCases(deploymentId = null, isReviewed = null) {
    let url = '/monitoring/edge-cases?';
    if (deploymentId) url += `deployment_id=${deploymentId}&`;
    if (isReviewed !== null) url += `is_reviewed=${isReviewed}`;
    return this.request(url);
  }

  async reviewEdgeCase(caseId, groundTruth) {
    return this.request(`/monitoring/edge-cases/${caseId}/review`, {
      method: 'POST',
      body: JSON.stringify({
        edge_case_id: caseId,
        ground_truth: groundTruth,
      }),
    });
  }

  async triggerRetraining(loopId) {
    return this.request(`/monitoring/feedback-loops/${loopId}/trigger-retrain`, {
      method: 'POST',
    });
  }
}

// Export for use
// const apiClient = new VisionAIClient();
// export default apiClient;
"""
