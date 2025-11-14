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
                    // Handle detail as array or string
                    if (Array.isArray(errorBody.detail)) {
                        errorDetail = errorBody.detail.map(e => e.msg || JSON.stringify(e)).join(', ');
                    } else {
                        errorDetail = errorBody.detail || `HTTP Error: ${response.status}`;
                    }
                    console.error('[API Error Response]', errorBody);
                    console.error('[API Error Detail]', errorDetail);
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

    // Helper function to get image dimensions
    async getImageDimensions(file) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            const objectUrl = URL.createObjectURL(file);

            img.onload = () => {
                URL.revokeObjectURL(objectUrl);
                resolve({ width: img.width, height: img.height });
            };

            img.onerror = () => {
                URL.revokeObjectURL(objectUrl);
                reject(new Error('Failed to load image'));
            };

            img.src = objectUrl;
        });
    }

    // Helper function to get video metadata
    async getVideoMetadata(file) {
        return new Promise((resolve, reject) => {
            const video = document.createElement('video');
            const objectUrl = URL.createObjectURL(file);

            video.onloadedmetadata = () => {
                URL.revokeObjectURL(objectUrl);
                resolve({
                    width: video.videoWidth,
                    height: video.videoHeight,
                    duration_ms: Math.round(video.duration * 1000)
                });
            };

            video.onerror = () => {
                URL.revokeObjectURL(objectUrl);
                reject(new Error('Failed to load video'));
            };

            video.src = objectUrl;
        });
    }

    /**
     * Upload files using Presigned URL method (recommended)
     * @param {FileList|Array} files - Files to upload
     * @param {number|null} datasetId - Optional: existing dataset ID
     * @param {string|null} datasetName - Required if datasetId is null
     * @param {string|null} description - Optional: dataset description
     * @param {function} onProgress - Optional: progress callback (current, total)
     * @returns {Promise} Upload result with dataset info and successful/failed uploads
     */
    async uploadWithPresignedUrl(files, datasetId = null, datasetName = null, description = null, onProgress = null) {
        const BATCH_SIZE = 500;  // 500장씩 배치 처리
        const CONCURRENT_UPLOADS = 10;  // 배치 내에서도 10개씩만 동시 업로드
        const MAX_RETRIES = 5;  // 최대 5번 재시도

        console.log(`[API] Starting upload: ${files.length} files in batches of ${BATCH_SIZE}`);
        console.log('[API] Dataset ID:', datasetId, 'Dataset Name:', datasetName);

        try {
            let targetDatasetId = datasetId;
            const allUploadedItems = [];
            const filesArray = Array.from(files);
            
            // 배치로 나누기
            for (let batchIndex = 0; batchIndex < filesArray.length; batchIndex += BATCH_SIZE) {
                const batch = filesArray.slice(batchIndex, batchIndex + BATCH_SIZE);
                const batchNumber = Math.floor(batchIndex / BATCH_SIZE) + 1;
                const totalBatches = Math.ceil(filesArray.length / BATCH_SIZE);
                
                console.log(`[API] Processing batch ${batchNumber}/${totalBatches}: ${batch.length} files`);
                
                // Step 1: 배치별 Presigned URL 요청
                const filenames = batch.map(f => f.name);
                const urlRequest = { filenames };
                
                // 첫 배치는 dataset 생성, 이후 배치는 기존 dataset 사용
                if (batchIndex === 0) {
                    if (datasetId) {
                        urlRequest.dataset_id = datasetId;
                    } else if (datasetName) {
                        urlRequest.name = datasetName;
                        if (description) {
                            urlRequest.description = description;
                        }
                    } else {
                        throw new Error('Either datasetId or datasetName must be provided');
                    }
                } else {
                    urlRequest.dataset_id = targetDatasetId;
                }
                
                console.log('[API] Requesting presigned URLs for batch:', urlRequest);
                const urlResponse = await this.post('/datasets/presigned-upload-urls', urlRequest);
                targetDatasetId = urlResponse.dataset.id;
                console.log(`[API] Received ${urlResponse.urls.length} presigned URLs for batch ${batchNumber}`);
                
                // Step 2: 배치 내 파일들을 동시성 제어하며 업로드 (Retry 포함)
                const batchUploadedItems = [];
                
                for (let i = 0; i < batch.length; i += CONCURRENT_UPLOADS) {
                    const chunk = batch.slice(i, i + CONCURRENT_UPLOADS);
                    const chunkUrls = urlResponse.urls.slice(i, i + CONCURRENT_UPLOADS);
                    
                    const chunkPromises = chunk.map(async (file, idx) => {
                        const urlInfo = chunkUrls[idx];
                        
                        // Retry 로직으로 업로드 시도
                        const uploadSuccess = await this._uploadFileWithRetry(
                            urlInfo.upload_url,
                            file,
                            MAX_RETRIES
                        );
                        
                        if (!uploadSuccess) {
                            console.error(`[API] Failed to upload ${file.name} after ${MAX_RETRIES} retries`);
                            return null;
                        }
                        
                        // 메타데이터 추출
                        let metadata = {
                            s3_key: urlInfo.s3_key,
                            original_filename: file.name,
                            file_size: file.size
                        };
                        
                        // Extract dimensions for images/videos
                        if (file.type.startsWith('image/')) {
                            try {
                                const { width, height } = await this.getImageDimensions(file);
                                metadata.width = width;
                                metadata.height = height;
                            } catch (error) {
                                console.warn('[API] Failed to get image dimensions:', error);
                            }
                        } else if (file.type.startsWith('video/')) {
                            try {
                                const { width, height, duration_ms } = await this.getVideoMetadata(file);
                                metadata.width = width;
                                metadata.height = height;
                                metadata.duration_ms = duration_ms;
                            } catch (error) {
                                console.warn('[API] Failed to get video metadata:', error);
                            }
                        }
                        
                        // 전체 진행률 업데이트
                        if (onProgress) {
                            const completed = batchIndex + i + idx + 1;
                            onProgress(completed, filesArray.length);
                        }
                        
                        return metadata;
                    });
                    
                    const chunkResults = await Promise.all(chunkPromises);
                    batchUploadedItems.push(...chunkResults.filter(r => r !== null));
                }
                
                // Step 3: 배치별 업로드 완료 알림
                if (batchUploadedItems.length > 0) {
                    console.log(`[API] Notifying backend: batch ${batchNumber} complete (${batchUploadedItems.length} files)`);
                    const completeResponse = await this.post(
                        `/datasets/${targetDatasetId}/upload-complete-batch`,
                        { items: batchUploadedItems }
                    );
                    allUploadedItems.push(...batchUploadedItems);
                }
                
                // 배치 간 짧은 대기 (Rate Limiting 방지)
                if (batchIndex + BATCH_SIZE < filesArray.length) {
                    console.log('[API] Waiting 1s before next batch...');
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            }
            
            const failedCount = filesArray.length - allUploadedItems.length;
            
            console.log(`[API] Upload complete: ${allUploadedItems.length} succeeded, ${failedCount} failed`);
            
            return {
                success: true,
                successful_count: allUploadedItems.length,
                failed_count: failedCount,
                dataset_id: targetDatasetId
            };
            
        } catch (error) {
            console.error('[API] Presigned URL upload failed:', error);
            throw error;
        }
    }

    /**
     * Exponential Backoff를 사용한 파일 업로드 재시도
     * @param {string} uploadUrl - S3 Presigned URL
     * @param {File} file - 업로드할 파일
     * @param {number} maxRetries - 최대 재시도 횟수
     * @returns {Promise<boolean>} 성공 여부
     */
    async _uploadFileWithRetry(uploadUrl, file, maxRetries = 5) {
        let attempt = 0;
        
        while (attempt < maxRetries) {
            try {
                const response = await fetch(uploadUrl, {
                    method: 'PUT',
                    body: file
                });
                
                // 성공
                if (response.ok) {
                    if (attempt > 0) {
                        console.log(`[API] ${file.name} uploaded successfully after ${attempt} retries`);
                    }
                    return true;
                }
                
                // 재시도 가능한 에러 (503 SlowDown, 500 Internal Error, 429 Too Many Requests)
                if (response.status === 503 || response.status === 500 || response.status === 429) {
                    const backoffMs = Math.min(1000 * Math.pow(2, attempt), 10000); // 최대 10초
                    console.warn(`[API] ${file.name} upload failed (${response.status}), retrying in ${backoffMs}ms... (attempt ${attempt + 1}/${maxRetries})`);
                    
                    await new Promise(resolve => setTimeout(resolve, backoffMs));
                    attempt++;
                    continue;
                }
                
                // 재시도 불가능한 에러 (400, 403, 404 등)
                console.error(`[API] ${file.name} upload failed with non-retryable error: ${response.status}`);
                return false;
                
            } catch (error) {
                // 네트워크 에러 등 - 재시도 가능
                const backoffMs = Math.min(1000 * Math.pow(2, attempt), 10000);
                console.warn(`[API] ${file.name} upload error (${error.message}), retrying in ${backoffMs}ms... (attempt ${attempt + 1}/${maxRetries})`);
                
                await new Promise(resolve => setTimeout(resolve, backoffMs));
                attempt++;
            }
        }
        
        // 최대 재시도 횟수 초과
        console.error(`[API] ${file.name} upload failed after ${maxRetries} attempts`);
        return false;
    }

    /**
     * Get presigned download URL for a single asset
     * @param {number} assetId - Asset ID
     * @returns {Promise} Object with download_url, filename, etc.
     */
    async getAssetPresignedDownload(assetId) {
        console.log(`[API] Getting presigned download URL for asset ${assetId}...`);
        return this.get(`/datasets/assets/${assetId}/presigned-download`);
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

    // ========== LABELS (S3) ==========
    /**
     * Get presigned URL for uploading label file to S3
     * @param {number} datasetId - Dataset ID
     * @param {string} imageFilename - Image filename (e.g., "image1.jpg")
     * @returns {Promise} Object with upload_url, s3_key, filename
     */
    async getPresignedLabelUploadUrl(datasetId, imageFilename) {
        console.log(`[API] Getting presigned URL for label: ${imageFilename}`);
        return this.post(`/datasets/${datasetId}/labels/presigned-upload-url`, {
            filename: imageFilename
        });
    }

    /**
     * Upload label file to S3 using presigned URL
     * @param {string} uploadUrl - Presigned upload URL
     * @param {string} labelContent - Label file content (YOLO format)
     * @returns {Promise}
     */
    async uploadLabelToS3(uploadUrl, labelContent) {
        console.log(`[API] Uploading label to S3...`);
        console.log(`[API] Upload URL:`, uploadUrl);
        console.log(`[API] Content length:`, labelContent.length, 'bytes');

        const response = await fetch(uploadUrl, {
            method: 'PUT',
            body: labelContent
            // No headers - let browser set Content-Type automatically
        });

        console.log(`[API] S3 upload response status:`, response.status);

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`[API] S3 upload error response:`, errorText);
            throw new Error(`S3 label upload failed: ${response.status} - ${errorText}`);
        }

        console.log(`[API] S3 upload successful`);
        return { success: true };
    }

    /**
     * Delete label file from S3
     * @param {number} datasetId - Dataset ID
     * @param {string} imageFilename - Image filename
     * @returns {Promise}
     */
    async deleteLabelFromS3(datasetId, imageFilename) {
        console.log(`[API] Deleting label from S3: ${imageFilename}`);
        return this.delete(`/datasets/${datasetId}/labels/${imageFilename}`);
    }

    /**
     * Complete workflow: Upload label to S3
     * @param {number} datasetId - Dataset ID
     * @param {string} imageFilename - Image filename
     * @param {string} labelContent - YOLO format label content
     * @returns {Promise}
     */
    async uploadLabel(datasetId, imageFilename, labelContent) {
        try {
            console.log(`[API] Starting label upload for dataset ${datasetId}, image: ${imageFilename}`);
            console.log(`[API] Label content preview:`, labelContent.substring(0, 200));

            // 1. Get presigned URL
            console.log(`[API] Step 1: Requesting presigned URL...`);
            const urlData = await this.getPresignedLabelUploadUrl(datasetId, imageFilename);
            console.log(`[API] Presigned URL received:`, urlData);

            // 2. Upload to S3
            console.log(`[API] Step 2: Uploading to S3...`);
            await this.uploadLabelToS3(urlData.upload_url, labelContent);

            console.log(`[API] Label uploaded successfully: ${urlData.filename} to ${urlData.s3_key}`);
            return { success: true, s3_key: urlData.s3_key };
        } catch (error) {
            console.error(`[API] Failed to upload label:`, error);
            console.error(`[API] Error details:`, error.message, error.stack);
            throw error;
        }
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

    async convertToOnnx(payload) {
        console.log('[API] Converting model to ONNX...', payload);
        return this.post('/conversion/onnx', payload);
    }

    async convertToTensorRT(payload) {
        console.log('[API] Converting model to TensorRT...', payload);
        return this.post('/conversion/trt', payload);
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

    /**
     * LLM 기반 자연어 학습 요청
     * @param {Object} data - LLM 학습 요청 데이터
     * @param {string} data.user_query - 자연어 학습 요청 (예: 'yolov8n으로 20 에포크 학습해줘')
     * @param {string} data.dataset_name - 데이터셋 이름
     * @param {string} data.dataset_s3_prefix - S3 데이터셋 경로 (예: 'datasets/myset/')
     * @param {string} [data.run_name] - 학습 작업 이름 (선택사항, 자동 생성)
     * @returns {Promise} TrainingJobResponse
     */
    async createLLMTraining(data) {
        console.log('[API] Starting LLM training:', data);
        return this.post('/training/llm', data);
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

    async getTrainingResultsFromS3(modelName) {
        console.log(`[API] Fetching training results from S3 for model ${modelName}...`);
        try {
            // Fetch result.csv from S3: /{model_name}/results.csv
            const endpoint = `/training/results/${modelName}/results.csv`;
            console.log(`[API] Fetching CSV from: ${this.baseURL}${endpoint}`);

            // Build headers manually
            const headers = {};
            const token = this.getAuthToken();
            if (token) {
                headers['Authorization'] = `Bearer ${token}`;
            }

            const response = await fetch(`${this.baseURL}${endpoint}`, {
                method: 'GET',
                headers: headers
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const csvText = await response.text();
            console.log(`[API] Fetched CSV (${csvText.length} chars)`);

            // Parse CSV to JSON
            const lines = csvText.trim().split('\n');
            if (lines.length < 2) {
                console.warn('[API] CSV file is empty or has no data rows');
                return [];
            }

            const csvHeaders = lines[0].split(',').map(h => h.trim());
            const metrics = [];

            for (let i = 1; i < lines.length; i++) {
                const values = lines[i].split(',').map(v => v.trim());
                const metric = {};

                csvHeaders.forEach((header, index) => {
                    const value = values[index];
                    // Convert numeric values
                    metric[header] = isNaN(value) ? value : parseFloat(value);
                });

                metrics.push(metric);
            }

            console.log(`[API] Parsed ${metrics.length} metric rows from CSV`);
            return metrics;

        } catch (error) {
            console.error(`[API] Failed to fetch training results from S3:`, error);
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

    async getVersion(datasetId, versionId) {
        console.log(`[API] Fetching version ${versionId} from dataset ${datasetId}...`);
        return this.get(`/datasets/${datasetId}/versions/${versionId}`);
    }

    async createVersion(datasetId, versionData) {
        console.log('[API] Creating dataset version:', versionData);
        return this.post(`/datasets/${datasetId}/versions`, versionData);
    }

    async updateVersion(datasetId, versionId, versionData) {
        console.log(`[API] Updating version ${versionId} in dataset ${datasetId}:`, versionData);
        return this.put(`/datasets/${datasetId}/versions/${versionId}`, versionData);
    }

    async deleteVersion(datasetId, versionId) {
        console.log(`[API] Deleting version ${versionId} from dataset ${datasetId}...`);
        return this.delete(`/datasets/${datasetId}/versions/${versionId}`);
    }

    // ========== EXPORT ==========
    async getExportJobs() {
        console.log('[API] Fetching export jobs...');
        try {
            const jobs = await this.get('/export/');
            console.log(`[API] Fetched ${Array.isArray(jobs) ? jobs.length : 0} export jobs`);
            return jobs;
        } catch (error) {
            console.error('[API] Failed to fetch export jobs:', error);
            return [];
        }
    }

    async getExportJob(exportId) {
        console.log(`[API] Fetching export job ${exportId}...`);
        return this.get(`/export/${exportId}`);
    }

    async createExport(exportData) {
        console.log('[API] Creating export job:', exportData);
        return this.post('/export/', exportData);
    }

    async downloadExport(exportId) {
        console.log(`[API] Downloading export ${exportId}...`);

        const url = `${this.baseURL}/export/${exportId}/download`;

        const headers = {};
        const token = this.getAuthToken();
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
            console.log('[API] Download with auth token');
        } else {
            console.warn('[API] Download without auth token – backend may return 401');
        }

        try {
            const response = await fetch(url, {
                method: 'GET',
                headers,
            });

            console.log('[API] Download response status:', response.status);

            if (!response.ok) {
                let errorMsg = `Download failed: ${response.status}`;
                try {
                    const errBody = await response.json();
                    errorMsg = errBody.detail || errorMsg;
                } catch (_) {
                }
                console.error('[API] Download error:', errorMsg);
                throw new Error(errorMsg);
            }

            const blob = await response.blob();

            let filename = `export_${exportId}.zip`;
            const disposition = response.headers.get('Content-Disposition');
            if (disposition && disposition.includes('filename=')) {
                let tmp = disposition.split('filename=')[1].trim();
                tmp = tmp.replace(/^["']|["']$/g, '');
                if (tmp) filename = tmp;
            }

            const downloadUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(downloadUrl);

            console.log('[API] Download completed:', filename);
            return true;
        } catch (error) {
            console.error('[API] Download export error:', error);
            throw error;
        }
    }

    async deleteExport(exportId) {
        console.log(`[API] Deleting export job ${exportId}...`);
        return this.delete(`/export/${exportId}`);
    }
}

// Create singleton instance and make it globally accessible
const apiService = new APIService();
window.apiService = apiService;

console.log('[API Service] Service ready and globally accessible');
