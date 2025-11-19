// Training Page Component - Real-time Training Monitoring

class TrainingPage {
    constructor() {
        this.trainingJobs = [];
        this.selectedJob = null;
        this.chart = null;
        this.rabbitmqConnected = false;
        this.metricsData = {}; // Store real-time metrics by job_id
        this.refreshInterval = null; // Auto-refresh interval
        this.hyperparameters = {}; // Store hyperparameters by job_id (from RabbitMQ train.hpo)
        this.currentSubscription = null; // Current training log subscription
        this.currentJobId = null; // Currently subscribed job_id
    }

    async init() {
        console.log('[Training Page] Initializing...');

        try {
            await this.loadTrainingJobs();

            // Render the page
            const app = document.getElementById('app');
            if (app) {
                app.innerHTML = this.render();
                await this.afterRender(); // This will initialize chart and connect to RabbitMQ
            }

            this.attachEventListeners();

            // Start periodic refresh (every 10 seconds)
            this.startPeriodicRefresh();

            console.log('[Training Page] Initialized successfully');

            // FAB 버튼 생성 (init 완료 후)
            console.log('[Training Page] About to create FAB button...');
            try {
                this.createFAB();
                console.log('[Training Page] createFAB() call completed');
            } catch (fabError) {
                console.error('[Training Page] Error creating FAB button:', fabError);
            }
        } catch (error) {
            console.error('[Training Page] Initialization error:', error);
            showToast('Failed to load training page: ' + error.message, 'error');
        }
    }

    async connectToRabbitMQ() {
        try {
            console.log('[Training Page] Connecting to RabbitMQ...');

            // Connect to RabbitMQ
            await rabbitmqService.connect();
            this.rabbitmqConnected = true;

            // Subscribe to hyperparameter messages (train.hpo routing key)
            rabbitmqService.subscribe('train.hpo', (message) => {
                this.handleHyperparameterMessage(message);
            }, 'exchange', 'jobs.cmd');

            console.log('[Training Page] ✅ Connected to RabbitMQ');
            showToast('Connected to real-time training updates', 'success');

            // Subscribe to selected job if any
            if (this.selectedJob && this.selectedJob.external_job_id) {
                this.subscribeToJobLogs(this.selectedJob.external_job_id);
            }

            // Restart periodic refresh with longer interval (now that RabbitMQ is connected)
            this.startPeriodicRefresh();

        } catch (error) {
            console.error('[Training Page] Failed to connect to RabbitMQ:', error);
            showToast('Failed to connect to real-time updates. Using manual refresh.', 'warning');
        }
    }

    /**
     * Subscribe to training logs for a specific job
     * @param {string} jobId - External job ID
     */
    subscribeToJobLogs(jobId) {
        if (!jobId) {
            console.warn('[Training Page] Cannot subscribe: jobId is empty');
            return;
        }

        // Unsubscribe from previous job if exists
        if (this.currentSubscription && this.currentJobId) {
            try {
                const oldRoutingKey = `train.log.${this.currentJobId}`;
                rabbitmqService.unsubscribe(oldRoutingKey);
                console.log(`[Training Page] Unsubscribed from job: ${this.currentJobId}`);
            } catch (e) {
                console.warn('[Training Page] Error unsubscribing from previous job:', e);
            }
        }

        // Subscribe to new job
        try {
            console.log(`[Training Page] Subscribing to training logs for job: ${jobId}`);
            this.currentSubscription = rabbitmqService.subscribeToTrainingLogs(jobId, (message) => {
                this.handleTrainingMetrics(message);
            });
            this.currentJobId = jobId;
            console.log(`[Training Page] ✅ Subscribed to job ${jobId} training logs`);
        } catch (error) {
            console.error(`[Training Page] Failed to subscribe to job ${jobId}:`, error);
        }
    }

    handleHyperparameterMessage(message) {
        /**
         * Handle hyperparameter message from RabbitMQ
         * Message format:
         * {
         *   job_id: "xxxxxxxxxx",
         *   hyperparams: {
         *     model_name: "yolo12n",
         *     epochs: 5,
         *     batch: 16,
         *     ...
         *   }
         * }
         */
        console.log('[Training Page] 📨 Received hyperparameter message:', message);
        
        try {
            const { job_id, hyperparams } = message;
            
            if (!job_id || !hyperparams) {
                console.warn('[Training Page] Invalid hyperparameter message format:', message);
                return;
            }
            
            // Store hyperparameters by job_id (as string for consistency)
            const jobIdStr = String(job_id);
            this.hyperparameters[jobIdStr] = hyperparams;
            console.log(`[Training Page] ✅ Stored hyperparameters for job_id: ${jobIdStr}`);
            console.log(`[Training Page] Current hyperparameters keys:`, Object.keys(this.hyperparameters));
            
            // Also try to match with all possible job identifiers
            // Store by multiple keys for easier matching
            this.trainingJobs.forEach(job => {
                const possibleIds = [
                    String(job.id),
                    job.external_job_id ? String(job.external_job_id) : null,
                    job.name
                ].filter(id => id);
                
                // Check if any of the possible IDs match
                if (possibleIds.includes(jobIdStr)) {
                    console.log(`[Training Page] Matched job ${job.id} (${job.name}) with job_id ${jobIdStr}`);
                    // Store also by job.id for easier lookup
                    this.hyperparameters[String(job.id)] = hyperparams;
                }
            });
            
            // Update UI to show the button
            this.updateHyperparameterButton(jobIdStr);
            
        } catch (error) {
            console.error('[Training Page] Error handling hyperparameter message:', error);
        }
    }

    handleTrainingMetrics(message) {
        /**
         * Handle real-time training metrics from RabbitMQ
         * GPU message format:
         * {
         *   job_id: "uuid-string",  // external_job_id
         *   epoch: 0,
         *   metrics: {
         *     loss: 38.22,
         *     "metrics/precision(B)": 0.0,
         *     "metrics/mAP50(B)": 0.0,
         *     ...
         *   }
         * }
         */
        console.log('[Training Page] Received training metrics:', message);

        try {
            const { job_id: external_job_id, epoch, metrics } = message;

            // Extract loss and accuracy from metrics object
            const loss = metrics?.loss || metrics?.tloss?.[0] || 0;
            const accuracy = metrics?.['metrics/mAP50(B)'] || metrics?.['metrics/precision(B)'] || 0;

            console.log(`[Training Page] Metrics: epoch=${epoch}, loss=${loss}, accuracy=${accuracy}`);

            // GPU sends 0-based epoch, convert to 1-based for display
            const displayEpoch = epoch + 1;

            // Just update the display with received metrics (no job matching)
            this.updateMetricsDisplay({ epoch: displayEpoch, loss, accuracy: accuracy * 100 });

            // // TODO: Enable job matching when external_job_id mapping is ready
            // // Find job by external_job_id (UUID stored in hyperparameters)
            // const job = this.trainingJobs.find(j =>
            //     j.hyperparameters?.external_job_id === external_job_id
            // );
            //
            // if (!job) {
            //     console.warn('[Training Page] No job found for external_job_id:', external_job_id);
            //     return;
            // }
            //
            // console.log(`[Training Page] Matched job ${job.id} (${job.name}) with metrics:`, { epoch, loss, accuracy });
            //
            // // Store metrics by internal job ID
            // if (!this.metricsData[job.id]) {
            //     this.metricsData[job.id] = [];
            // }
            // this.metricsData[job.id].push({
            //     epoch,
            //     loss,
            //     accuracy: accuracy * 100, // Convert to percentage
            //     timestamp: new Date()
            // });
            //
            // // Update job metrics
            // job.current_epoch = epoch;
            // job.current_loss = loss;
            // job.current_accuracy = accuracy * 100;
            //
            // // Update UI if this is the selected job
            // if (this.selectedJob && this.selectedJob.id === job.id) {
            //     this.updateMetricsDisplay({ epoch, loss, accuracy: accuracy * 100 });
            // }
            //
            // // Update stats display
            // this.updateStatsDisplay();

        } catch (error) {
            console.error('[Training Page] Error handling training metrics:', error);
        }
    }

    updateMetricsDisplay(metrics) {
        /**
         * Update UI with new metrics
         */
        try {
            const { epoch, loss, accuracy } = metrics;

            console.log(`[Training Page] Updating UI: epoch=${epoch}, loss=${loss?.toFixed(4)}, accuracy=${accuracy?.toFixed(2)}%`);

            // Update stat cards (correct IDs from HTML)
            const accuracyEl = document.getElementById('live-accuracy');
            const lossEl = document.getElementById('live-loss');
            const epochEl = document.getElementById('live-epoch');

            console.log('[Training Page] Elements found:', {
                accuracyEl: !!accuracyEl,
                lossEl: !!lossEl,
                epochEl: !!epochEl
            });

            // accuracy is already * 100 from handleTrainingMetrics
            if (accuracyEl) {
                accuracyEl.textContent = accuracy.toFixed(2) + '%';
                console.log(`[Training Page] Updated accuracy: ${accuracyEl.textContent}`);
            } else {
                console.warn('[Training Page] ⚠️ live-accuracy element not found!');
            }

            if (lossEl) {
                lossEl.textContent = loss.toFixed(4);
                console.log(`[Training Page] Updated loss: ${lossEl.textContent}`);
            } else {
                console.warn('[Training Page] ⚠️ live-loss element not found!');
            }

            if (epochEl) {
                epochEl.textContent = `Epoch ${epoch}/${this.selectedJob?.total_epochs || '?'}`;
                console.log(`[Training Page] Updated epoch: ${epochEl.textContent}`);
            } else {
                console.warn('[Training Page] ⚠️ live-epoch element not found!');
            }

            // Update chart
            this.updateChartWithNewData(metrics);

            console.log('[Training Page] ✅ UI updated successfully');

        } catch (error) {
            console.error('[Training Page] Error updating metrics display:', error);
        }
    }

    updateChartWithNewData(metrics) {
        /**
         * Add new data point to chart
         */
        try {
            if (!this.chart) {
                // Silently skip if chart not initialized (will be initialized soon)
                // Don't log warning as it's expected during initialization
                return;
            }

            const { epoch, loss, accuracy } = metrics;

            // Add new data point (accuracy already * 100 from handleTrainingMetrics)
            this.chart.data.labels.push(`Epoch ${epoch}`);
            this.chart.data.datasets[0].data.push(loss);
            this.chart.data.datasets[1].data.push(accuracy); // Already percentage

            // Keep only last 50 points for performance
            if (this.chart.data.labels.length > 50) {
                this.chart.data.labels.shift();
                this.chart.data.datasets[0].data.shift();
                this.chart.data.datasets[1].data.shift();
            }

            // Update chart
            this.chart.update('none'); // Use 'none' animation for better performance

            console.log(`[Training Page] Chart updated: ${this.chart.data.labels.length} points`);

        } catch (error) {
            console.error('[Training Page] Error updating chart:', error);
        }
    }

    updateStatsDisplay() {
        /**
         * Update statistics display (active jobs, etc.)
         */
        const activeJobsEl = document.getElementById('stat-active-jobs');
        if (activeJobsEl) {
            const activeCount = this.trainingJobs.filter(j => j.status === 'running').length;
            activeJobsEl.textContent = activeCount;
        }
    }

    startPeriodicRefresh() {
        /**
         * Start periodic refresh of training jobs
         * Uses longer interval when RabbitMQ is connected (real-time updates)
         * Uses shorter interval as fallback when RabbitMQ is disconnected
         */
        // Clear existing interval if any
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }

        // Use longer interval if RabbitMQ is connected (60s vs 10s)
        // RabbitMQ provides real-time metrics, so we only need periodic refresh for job status
        const interval = this.rabbitmqConnected ? 60000 : 10000;
        console.log(`[Training Page] Starting periodic refresh (${interval/1000}s interval, RabbitMQ: ${this.rabbitmqConnected ? 'ON' : 'OFF'})`);

        // Refresh periodically
        this.refreshInterval = setInterval(async () => {
            console.log('[Training Page] Auto-refreshing training jobs...');
            await this.refreshTrainingJobs();
        }, interval);
    }

    async refreshTrainingJobs() {
        /**
         * Refresh training jobs without full page re-render
         * When RabbitMQ is connected, only refresh job status (not metrics)
         */
        try {
            const jobs = await apiService.getTrainingJobs();
            if (!jobs) return;

            // Only load S3 metrics if RabbitMQ is NOT connected
            // When RabbitMQ is connected, metrics come from real-time stream
            if (!this.rabbitmqConnected) {
                // Load S3 metrics for jobs that changed
                for (const job of jobs) {
                    const existingJob = this.trainingJobs.find(j => j.id === job.id);

                    // Only reload S3 if epoch changed or status changed
                    if (!existingJob ||
                        existingJob.current_epoch !== job.current_epoch ||
                        existingJob.status !== job.status) {
                        await this.loadS3MetricsForJob(job);
                    }
                }
            }

            this.trainingJobs = jobs;

            // Check if any jobs now have hyperparameters (in case message arrived before job was loaded)
            this.checkAndUpdateHyperparametersForJobs();

            // Update selected job reference
            if (this.selectedJob) {
                const updated = this.trainingJobs.find(j => j.id === this.selectedJob.id);
                if (updated) {
                    this.selectedJob = updated;
                }
            }

            // Re-render only the jobs list without destroying the chart
            this.updateJobsListDisplay();

        } catch (error) {
            console.error('[Training Page] Error refreshing jobs:', error);
        }
    }

    checkAndUpdateHyperparametersForJobs() {
        /**
         * Check if any loaded jobs match stored hyperparameters
         * This handles the case where hyperparameter message arrived before job was loaded
         */
        if (Object.keys(this.hyperparameters).length === 0) {
            return;
        }

        console.log('[Training Page] Checking hyperparameters for loaded jobs...');
        let updated = false;

        this.trainingJobs.forEach(job => {
            const possibleIds = [
                String(job.id),
                job.external_job_id ? String(job.external_job_id) : null,
                job.name
            ].filter(id => id);

            // Check if any stored hyperparameter key matches this job
            for (const storedKey of Object.keys(this.hyperparameters)) {
                if (possibleIds.includes(storedKey)) {
                    // Also store by job.id for easier lookup
                    if (!this.hyperparameters[String(job.id)]) {
                        this.hyperparameters[String(job.id)] = this.hyperparameters[storedKey];
                        console.log(`[Training Page] Matched hyperparameters for job ${job.id} using key: ${storedKey}`);
                        updated = true;
                    }
                }
            }
        });

        if (updated) {
            console.log('[Training Page] Updated hyperparameters mapping for jobs');
        }
    }

    updateJobsListDisplay() {
        /**
         * Update only the jobs list without re-rendering entire page
         */
        const jobsContainer = document.getElementById('training-jobs-list');
        if (jobsContainer) {
            const rendered = this.renderTrainingJobs();
            jobsContainer.innerHTML = Array.isArray(rendered) ? rendered.join('') : rendered;
        }

        // Update stats
        this.updateStatsDisplay();
    }

    destroy() {
        /**
         * Cleanup when page is destroyed
         */
        console.log('[Training Page] Destroying...');

        // Disconnect from RabbitMQ
        if (this.rabbitmqConnected) {
            rabbitmqService.unsubscribe('gpu.train.log');
            rabbitmqService.unsubscribe('train.hpo');
            console.log('[Training Page] Unsubscribed from RabbitMQ');
        }

        // Clear chart
        if (this.chart) {
            this.chart.destroy();
            this.chart = null;
        }

        // Clear refresh interval
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }

        console.log('[Training Page] Destroyed and cleaned up');
    }

    async loadTrainingJobs() {
        try {
            console.log('[Training Page] Loading training jobs...');
            const response = await apiService.getTrainingJobs();

            this.trainingJobs = Array.isArray(response) ? response : (response.jobs || response.data || []);
            console.log('[Training Page] Loaded jobs:', this.trainingJobs.length);

            // Select a job to display metrics for
            if (!this.selectedJob) {
                this.selectedJob = this.trainingJobs.find(j => j.status === 'running') ||
                                  this.trainingJobs.find(j => j.status === 'completed') ||
                                  this.trainingJobs[0];
            } else {
                // Update selected job data
                const updated = this.trainingJobs.find(j => j.id === this.selectedJob.id);
                if (updated) {
                    this.selectedJob = updated;
                }
            }

            // Only load S3 metrics for the selected job initially
            // Other jobs will load S3 metrics when selected or when RabbitMQ is not connected
            if (this.selectedJob) {
                console.log('[Training Page] Loading S3 metrics for selected job:', this.selectedJob.id, this.selectedJob.name);
                await this.loadS3MetricsForJob(this.selectedJob);
            }

            // Check if any jobs now have hyperparameters (in case message arrived before job was loaded)
            this.checkAndUpdateHyperparametersForJobs();

            this.trainingJobs.forEach(job => {
                console.log(`  Job ${job.id}: ${job.name} - Status: ${job.status}, Epoch: ${job.current_epoch + 1 || 0}/${job.total_epochs || 0}`);
            });

        } catch (error) {
            console.error('[Training Page] Error loading jobs:', error);
            this.trainingJobs = [];
            throw error;
        }
    }

    render() {
        console.log('[Training Page] Rendering with', this.trainingJobs.length, 'jobs');

        // Get data for selected job
        const selectedJob = this.selectedJob || this.trainingJobs[0];
        const currentAccuracy = selectedJob?.current_accuracy || 0;
        const currentLoss = selectedJob?.current_loss || 0;
        const currentEpoch = selectedJob?.current_epoch || 0;
        const totalEpochs = selectedJob?.total_epochs || 0;
        const activeJobsCount = this.trainingJobs.filter(j => j.status === 'running').length;

        return `
            <div class="min-vh-100 bg-light">
                <div class="container py-4">
                    <!-- Page Header -->
                    <div class="mb-4">
                        <h1 class="display-5 fw-bold mb-2">Model Training</h1>
                        <p class="text-muted">Monitor training progress and performance metrics in real-time</p>
                    </div>

                    <!-- Job Selector -->
                    ${this.trainingJobs.length > 0 ? `
                    <div class="card border-0 shadow-sm mb-4">
                        <div class="card-body">
                            <div class="row align-items-center">
                                <div class="col-md-6">
                                    <label class="form-label fw-bold mb-2">Select Training Job:</label>
                                    <select class="form-select" id="job-selector" onchange="window.trainingPage.selectJob(this.value)">
                                        ${this.trainingJobs.map(job => {
                                            // current_epoch is 0-based: 0 means completed epoch 1
                                            // Display as 1-based for user: epoch 0 -> "1/N", epoch 1 -> "2/N"
                                            const displayEpoch = (job.current_epoch ?? 0) + 1;
                                            return `
                                            <option value="${job.id}" ${selectedJob?.id === job.id ? 'selected' : ''}>
                                                ${job.name} - ${job.status.toUpperCase()} (${displayEpoch}/${job.total_epochs || 0} epochs)
                                            </option>
                                        `;}).join('')}
                                    </select>
                                </div>
                                <div class="col-md-6 text-end">
                                    <button class="btn btn-primary" onclick="window.trainingPage.showStartTrainingModal()">
                                        <i class="bi bi-play-fill me-1"></i> Start New Training
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    ` : ''}

                    <div class="row g-4 mb-4">
                        <!-- Stats Cards -->
                        <div class="col-lg-4">
                            <div class="card border-0 shadow-sm mb-3">
                                <div class="card-body">
                                    <h6 class="text-muted mb-3">Current Accuracy</h6>
                                    <h2 class="fw-bold text-success mb-2" id="live-accuracy">${currentAccuracy.toFixed(1)}%</h2>
                                    <div class="progress mb-2" style="height: 8px;">
                                        <div class="progress-bar bg-success" id="accuracy-bar" style="width: ${currentAccuracy}%"></div>
                                    </div>
                                    <p class="text-muted small mb-0" id="live-epoch">Epoch ${currentEpoch}/${totalEpochs}</p>
                                </div>
                            </div>

                            <div class="card border-0 shadow-sm mb-3">
                                <div class="card-body">
                                    <h6 class="text-muted mb-3">Training Loss</h6>
                                    <h2 class="fw-bold mb-2" id="live-loss">${currentLoss.toFixed(4)}</h2>
                                    <p class="text-success small mb-0">
                                        <i class="bi bi-arrow-down"></i> ${selectedJob?.status === 'running' ? 'Decreasing' : 'Final'}
                                    </p>
                                </div>
                            </div>

                            <div class="card border-0 shadow-sm">
                                <div class="card-body">
                                    <h6 class="text-muted mb-3">Active Jobs</h6>
                                    <h2 class="fw-bold mb-2" id="active-jobs-count">${activeJobsCount}</h2>
                                    <p class="text-muted small mb-0">Currently training</p>
                                </div>
                            </div>
                        </div>

                        <!-- Chart Card -->
                        <div class="col-lg-8">
                            <div class="card border-0 shadow-sm h-100">
                                <div class="card-header bg-white">
                                    <h5 class="mb-1 fw-bold">Training Metrics - ${selectedJob?.name || 'No Job Selected'}</h5>
                                    <p class="text-muted mb-0 small">
                                        Real-time loss and accuracy tracking
                                        ${selectedJob?.status === 'running' ? '<span class="badge bg-primary ms-2">LIVE</span>' : ''}
                                    </p>
                                </div>
                                <div class="card-body">
                                    <canvas id="trainingChart" height="100"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Training Jobs List -->
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-white">
                            <div class="row align-items-center">
                                <div class="col">
                                    <h5 class="mb-1 fw-bold">All Training Jobs</h5>
                                    <p class="text-muted mb-0 small">View and manage all training pipelines</p>
                                </div>
                                <div class="col-auto">
                                    <button class="btn btn-sm btn-outline-primary" 
                                            onclick="window.trainingPage.syncCompletedStatus(event)">
                                        <i class="bi bi-arrow-clockwise me-1"></i> Sync Status
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div class="card-body" id="training-jobs-list">
                            ${(() => {
                                const rendered = this.renderTrainingJobs();
                                return Array.isArray(rendered) ? rendered.join('') : rendered;
                            })()}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderTrainingJobs() {
        if (!this.trainingJobs || this.trainingJobs.length === 0) {
            return this.renderEmptyState();
        }

        return this.trainingJobs.map(job => {
            // current_epoch is 0-based: 0 means completed epoch 1
            // Display as 1-based for user: epoch 0 -> "1/N", epoch 1 -> "2/N"
            const displayEpoch = (job.current_epoch ?? 0) + 1;
            const progress = job.total_epochs > 0
                ? Math.round(displayEpoch / job.total_epochs * 100)
                : 0;

            const statusBadge = {
                'pending': 'bg-secondary',
                'running': 'bg-primary',
                'paused': 'bg-warning',
                'completed': 'bg-success',
                'failed': 'bg-danger'
            }[job.status] || 'bg-secondary';

            const isSelected = this.selectedJob?.id === job.id;

            // Parse hyperparameters safely
            let hyperparams = {};
            if (job.hyperparameters) {
                if (typeof job.hyperparameters === 'string') {
                    try {
                        hyperparams = JSON.parse(job.hyperparameters);
                    } catch (e) {
                        console.warn('Failed to parse hyperparameters:', e);
                    }
                } else {
                    hyperparams = job.hyperparameters;
                }
            }

            return `
                <div class="card mb-3 ${isSelected ? 'border-primary' : ''}" style="cursor: pointer;" onclick="window.trainingPage.selectJob(${job.id})">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start mb-3">
                            <div>
                                <h5 class="fw-bold mb-1">
                                    ${job.name}
                                    ${isSelected ? '<i class="bi bi-check-circle-fill text-primary ms-2"></i>' : ''}
                                </h5>
                                <p class="text-muted mb-0 small">Architecture: ${job.architecture || 'N/A'}</p>
                            </div>
                            <div class="d-flex align-items-center gap-2">
                                <span class="badge ${statusBadge}">${job.status.toUpperCase()}</span>
                                <button class="btn btn-sm btn-outline-danger"
                                        onclick="event.stopPropagation(); window.trainingPage.deleteTrainingJob(${job.id}, ${job.model_id})"
                                        title="Delete model and artifacts">
                                    <i class="bi bi-trash"></i>
                                </button>
                            </div>
                        </div>

                        <div class="mb-3">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <span class="text-muted small">Progress</span>
                                <div class="d-flex align-items-center gap-2">
                                    ${(() => {
                                        const hasHyperparams = this.hasHyperparameters(job);
                                        const jobIdForHyperparams = this.getJobIdForHyperparams(job);
                                        console.log(`[Training Page] Rendering job ${job.id}: hasHyperparams=${hasHyperparams}, jobIdForHyperparams=${jobIdForHyperparams}`);
                                        if (hasHyperparams && jobIdForHyperparams) {
                                            return `
                                                <button class="btn btn-sm btn-outline-info" 
                                                        onclick="event.stopPropagation(); window.trainingPage.showHyperparameterModal('${jobIdForHyperparams}')"
                                                        title="View Hyperparameters">
                                                    <i class="bi bi-sliders"></i> Hyperparameters
                                                </button>
                                            `;
                                        }
                                        return '';
                                    })()}
                                    <span class="fw-medium small">${displayEpoch}/${job.total_epochs || 0} epochs</span>
                                </div>
                            </div>
                            <div class="progress" style="height: 8px;">
                                <div class="progress-bar ${job.status === 'running' ? 'progress-bar-striped progress-bar-animated' : ''}"
                                     style="width: ${progress}%"></div>
                            </div>
                        </div>

                        <div class="row">
                            <div class="col-md-3">
                                <p class="text-muted small mb-1">Accuracy</p>
                                <p class="fw-medium mb-0 text-success">${(job.current_accuracy || 0).toFixed(2)}%</p>
                            </div>
                            <div class="col-md-3">
                                <p class="text-muted small mb-1">Loss</p>
                                <p class="fw-medium mb-0">${(job.current_loss || 0).toFixed(4)}</p>
                            </div>
                            <div class="col-md-3">
                                <p class="text-muted small mb-1">Learning Rate</p>
                                <p class="fw-medium mb-0 font-monospace small">${hyperparams.learning_rate || '0.001'}</p>
                            </div>
                            <div class="col-md-3">
                                <p class="text-muted small mb-1">Batch Size</p>
                                <p class="fw-medium mb-0">${hyperparams.batch_size || 16}</p>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    renderEmptyState() {
        return `
            <div class="text-center py-5">
                <i class="bi bi-robot text-muted mb-3" style="font-size: 5rem;"></i>
                <h5 class="text-muted">No Training Jobs Yet</h5>
                <p class="text-muted">Start training your first model to see it here</p>
                <button class="btn btn-primary" onclick="window.trainingPage.showStartTrainingModal()">
                    <i class="bi bi-play-fill me-1"></i> Start New Training
                </button>
            </div>
        `;
    }

    attachEventListeners() {
        // Event listeners are handled via onclick in HTML
    }

    async loadS3MetricsForJob(job) {
        /**
         * Load metrics from S3 results.csv for a job
         * New path structure: models/{dataset_name}/{train_type}/{version}/results.csv
         * - train_type: "train" (일반) or "ai-train" (AI 학습)
         * - version: "v1", "v2", "v3"...
         */
        if (!job || !job.hyperparameters?.version || !job.hyperparameters?.dataset_name) {
            console.warn('[Training Page] Missing required info for S3 path:', {
                job_id: job?.id,
                job_name: job?.name,
                hyperparameters: job?.hyperparameters,
                version: job?.hyperparameters?.version,
                dataset_name: job?.hyperparameters?.dataset_name
            });
            return null;
        }

        try {
            const datasetName = job.hyperparameters.dataset_name;
            const version = job.hyperparameters.version;
            const isAI = job.hyperparameters.ai_mode || false;
            
            // 경로 생성
            const trainType = isAI ? 'ai-train' : 'train';
            const modelPath = `${datasetName}/${trainType}/${version}`;
            
            console.log(`[Training Page] Loading S3 results for job ${job.id} (${job.name}):`, {
                dataset_name: datasetName,
                version: version,
                train_type: trainType,
                model_path: modelPath,
                full_s3_path: `models/${modelPath}/results.csv`,
                hyperparameters: job.hyperparameters
            });
            const s3Metrics = await apiService.getTrainingResultsFromS3(modelPath);

            if (s3Metrics && s3Metrics.length > 0) {
                console.log(`[Training Page] Loaded ${s3Metrics.length} metrics from S3 for job ${job.id}`);

                // Store S3 metrics for later use
                job.s3Metrics = s3Metrics;

                // Update job with latest metrics from S3
                const latestMetric = s3Metrics[s3Metrics.length - 1];

                // IMPORTANT: YOLO v8 results.csv uses 1-based epoch (1, 2, 3, ..., N for N epochs)
                // But DB current_epoch uses 0-based (0 = first epoch completed)
                // Convert: CSV epoch 1 -> DB 0, CSV epoch 20 -> DB 19
                const csvEpoch = latestMetric.epoch;
                console.log(`[Training Page] CSV epoch value (1-based): ${csvEpoch}, Total epochs: ${job.total_epochs}`);

                // Convert from 1-based (CSV) to 0-based (DB)
                job.current_epoch = (csvEpoch && csvEpoch > 0) ? csvEpoch - 1 : 0;
                job.current_accuracy = (latestMetric['metrics/mAP50(B)'] || latestMetric['metrics/precision(B)'] || 0) * 100;
                job.current_loss = latestMetric['val/box_loss'] || latestMetric['train/box_loss'] || 0;

                // Extract hyperparameters from S3 if available
                if (s3Metrics[0]) {
                    const firstMetric = s3Metrics[0];
                    if (!job.hyperparameters) job.hyperparameters = {};

                    // Learning rate from lr/pg0, lr/pg1, lr/pg2
                    if (firstMetric['lr/pg0']) {
                        job.hyperparameters.learning_rate = firstMetric['lr/pg0'];
                    }
                }

                console.log('[Training Page] Updated job with S3 metrics:', {
                    csvEpoch: csvEpoch,
                    current_epoch: job.current_epoch,
                    total_epochs: job.total_epochs,
                    displayWillBe: `${job.current_epoch + 1}/${job.total_epochs}`,
                    accuracy: job.current_accuracy?.toFixed(2),
                    loss: job.current_loss?.toFixed(4)
                });

                return s3Metrics;
            }
        } catch (error) {
            console.warn(`[Training Page] Could not load S3 results for job ${job.id}:`, error);
        }
        return null;
    }

    async selectJob(jobId) {
        const id = parseInt(jobId);
        const job = this.trainingJobs.find(j => j.id === id);

        if (job) {
            console.log('[Training Page] Selected job:', job.id, job.name);
            this.selectedJob = job;

            // Subscribe to this job's training logs if RabbitMQ is connected
            if (this.rabbitmqConnected && job.external_job_id) {
                console.log(`[Training Page] Switching to job ${job.external_job_id} logs`);
                this.subscribeToJobLogs(job.external_job_id);
            } else if (!job.external_job_id) {
                console.warn('[Training Page] Selected job has no external_job_id');
            }

            // Load S3 metrics for this job
            await this.loadS3MetricsForJob(job);

            // Re-render the page
            const app = document.getElementById('app');
            if (app) {
                app.innerHTML = this.render();
                await this.afterRender();
            }
        }
    }

    async afterRender() {
        console.log('[Training Page] afterRender called');
        // Initialize chart first (before RabbitMQ messages can arrive)
        await this.initChart();
        console.log('[Training Page] Chart initialized');
        
        // Ensure RabbitMQ is connected and subscribed (only if not already connected)
        if (!this.rabbitmqConnected) {
            await this.connectToRabbitMQ();
        }
        
        // FAB 버튼 생성 (afterRender 완료 후)
        this.createFAB();
    }

    async initChart() {
        try {
            const ctx = document.getElementById('trainingChart');
            if (!ctx) {
                console.warn('[Training Page] Chart canvas not found');
                return;
            }

            // Destroy existing chart
            if (this.chart) {
                this.chart.destroy();
                this.chart = null;
            }

            // Initialize with empty or existing metrics
            let labels = [];
            let lossData = [];
            let accuracyData = [];

            if (this.selectedJob) {
                console.log('[Training Page] Initializing chart for job', this.selectedJob.id);

                // Try to use S3 metrics first (more complete)
                let metrics = this.selectedJob.s3Metrics;

                if (!metrics || metrics.length === 0) {
                    // Fallback to API metrics
                    metrics = await apiService.getTrainingMetrics(this.selectedJob.id);
                    console.log('[Training Page] Loaded API metrics:', metrics?.length || 0);
                } else {
                    console.log('[Training Page] Using S3 metrics:', metrics.length);
                }

                console.log('[Training Page] Raw metrics data:', metrics);

                if (metrics && metrics.length > 0) {
                    // Extract chart data from metrics
                    // For S3 metrics: epoch is 1-based in CSV (1, 2, 3, ..., N)
                    // Display as-is for chart labels (already correct for display)
                    labels = metrics.map((m, idx) => `Epoch ${m.epoch ?? (idx + 1)}`);
                    lossData = metrics.map(m => m['val/box_loss'] || m['train/box_loss'] || m.train_loss || 0);
                    accuracyData = metrics.map(m => {
                        const acc = m['metrics/mAP50(B)'] || m['metrics/precision(B)'] || m.train_accuracy || 0;
                        return acc * 100; // Convert to percentage
                    });
                }
            } else {
                console.log('[Training Page] Initializing empty chart - will populate with real-time data');
            }
            
            console.log('[Training Page] Chart labels:', labels);
            console.log('[Training Page] Loss data:', lossData);
            console.log('[Training Page] Accuracy data:', accuracyData);

            this.chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Loss',
                        data: lossData,
                        borderColor: 'rgb(239, 68, 68)',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        yAxisID: 'y',
                        tension: 0.4
                    }, {
                        label: 'Accuracy (%)',
                        data: accuracyData,
                        borderColor: 'rgb(16, 185, 129)',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        yAxisID: 'y1',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Loss'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Accuracy (%)'
                            },
                            grid: {
                                drawOnChartArea: false,
                            },
                        },
                    }
                }
            });

            console.log('[Training Page] Chart initialized successfully');
            console.log('[Training Page] Chart object:', this.chart);

        } catch (error) {
            console.error('[Training Page] Error initializing chart:', error);
        }
    }

    async updateChart() {
        if (!this.chart || !this.selectedJob) {
            console.log('[Training Page] Cannot update chart - chart or selectedJob missing');
            return;
        }

        try {
            const metrics = await apiService.getTrainingMetrics(this.selectedJob.id);
            if (!metrics || metrics.length === 0) {
                console.log('[Training Page] No metrics for chart update');
                return;
            }

            console.log(`[Training Page] Updating chart with ${metrics.length} metrics`);

            // Update chart data
            // API metrics may have different format, but epoch should be used as-is
            this.chart.data.labels = metrics.map((m, idx) => `Epoch ${m.epoch ?? (idx + 1)}`);
            this.chart.data.datasets[0].data = metrics.map(m => m.train_loss || 0);
            this.chart.data.datasets[1].data = metrics.map(m => (m.train_accuracy || 0));
            this.chart.update('none');
            
            console.log('[Training Page] Chart updated successfully');
        } catch (error) {
            console.error('[Training Page] Error updating chart:', error);
        }
    }

    cleanup() {
        console.log('[Training Page] Cleaning up...');
        this.removeFAB();  // FAB 버튼 제거
        if (this.chart) {
            this.chart.destroy();
            this.chart = null;
        }
    }

    createFAB() {
        console.log('[Training Page] Creating FAB button...');
        
        // 이미 존재하면 제거
        const existingFAB = document.getElementById('llm-fab-button');
        if (existingFAB) {
            console.log('[Training Page] Removing existing FAB button');
            existingFAB.remove();
        }

        // FAB 버튼 생성 (pill 형태로 변경)
        const fabButton = document.createElement('button');
        fabButton.id = 'llm-fab-button';
        fabButton.className = 'btn shadow-lg';
        fabButton.innerHTML = `
            <div class="d-flex align-items-center gap-2 px-1">
                <i class="bi bi-robot fs-4"></i>
                <span class="fw-bold" style="font-size: 1.1rem;">AI Training</span>
            </div>
        `;
        fabButton.title = 'AI 자동 학습 시작';
        
        // CSS 스타일 적용 (pill 형태, 더 작고 반투명)
        fabButton.style.cssText = `
            position: fixed;
            bottom: 24px;
            right: 24px;
            height: 48px;
            padding: 0 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1050;
            border: none;
            border-radius: 24px;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, #3b82f6, #0ea5e9);
            color: white;
            opacity: 0.9;
        `;

        // 호버 효과
        fabButton.addEventListener('mouseenter', () => {
            fabButton.style.transform = 'scale(1.05) translateY(-2px)';
            fabButton.style.boxShadow = '0 12px 24px rgba(59, 130, 246, 0.4)';
            fabButton.style.opacity = '1';
        });

        fabButton.addEventListener('mouseleave', () => {
            fabButton.style.transform = 'scale(1) translateY(0)';
            fabButton.style.boxShadow = '';
            fabButton.style.opacity = '0.9';
        });

        fabButton.addEventListener('mousedown', () => {
            fabButton.style.transform = 'scale(0.98) translateY(0)';
        });

        fabButton.addEventListener('mouseup', () => {
            fabButton.style.transform = 'scale(1.05) translateY(-2px)';
        });

        // 클릭 이벤트
        fabButton.onclick = () => {
            console.log('[Training Page] FAB button clicked');
            if (typeof showLLMModal === 'function') {
                showLLMModal();
            } else {
                console.warn('showLLMModal function not found');
                showToast('LLM 모달 기능을 로드하는 중입니다...', 'info');
            }
        };

        // 아이콘 색상
        const icon = fabButton.querySelector('i');
        if (icon) {
            icon.style.color = 'white';
        }

        document.body.appendChild(fabButton);
        console.log('[Training Page] FAB button created and appended to body');
    }

    removeFAB() {
        console.log('[Training Page] Removing FAB button...');
        const fabButton = document.getElementById('llm-fab-button');
        if (fabButton) {
            fabButton.remove();
            console.log('[Training Page] FAB button removed');
        }
    }

    // Modal and control methods
    showStartTrainingModal() {
        const modalHTML = `
            <div class="modal fade" id="startTrainingModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">
                                <i class="bi bi-play-fill me-2"></i>Start New Training
                            </h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <form id="start-training-form">
                                <div class="mb-3">
                                    <label class="form-label">Training Job Name *</label>
                                    <input type="text" class="form-control" id="training-name" required
                                           placeholder="e.g., ProductDefect-Training-v1">
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Dataset *</label>
                                    <select class="form-select" id="dataset-select" required>
                                        <option value="">-- Select Dataset --</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Select Model to Train *</label>
                                    <select class="form-select" id="model-select" required>
                                        <option value="">-- Select Model --</option>
                                    </select>
                                    <small class="text-muted">Choose an existing model or create a new one</small>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Model Architecture *</label>
                                    <select class="form-select" id="architecture-select" required>
                                        <option value="">-- Select Architecture --</option>
                                        <optgroup label="Object Detection (YOLO)">
                                            <option value="yolov8n">YOLOv8 Nano (Fastest, Smallest)</option>
                                            <option value="yolov8s">YOLOv8 Small</option>
                                            <option value="yolov8m">YOLOv8 Medium</option>
                                            <option value="yolov8l">YOLOv8 Large</option>
                                            <option value="yolov8x">YOLOv8 XLarge (Most Accurate)</option>
                                        </optgroup>
                                        <optgroup label="Image Classification (Not Yet Supported)">
                                            <option value="resnet18" disabled>ResNet18 (Coming Soon)</option>
                                            <option value="resnet50" disabled>ResNet50 (Coming Soon)</option>
                                            <option value="mobilenet_v2" disabled>MobileNet V2 (Coming Soon)</option>
                                        </optgroup>
                                    </select>
                                    <small class="text-muted">YOLO for object detection, ResNet/MobileNet for classification</small>
                                </div>
                                <div class="row">
                                    <div class="col-md-3">
                                        <label class="form-label">Epochs</label>
                                        <input type="number" class="form-control" id="epochs" value="20" min="1">
                                    </div>
                                    <div class="col-md-3">
                                        <label class="form-label">Batch Size</label>
                                        <input type="number" class="form-control" id="batch-size" value="16" min="1">
                                    </div>
                                    <div class="col-md-3">
                                        <label class="form-label">Image Size</label>
                                        <input type="number" class="form-control" id="img-size" value="640" step="32" min="320">
                                        <small class="text-muted">For YOLO</small>
                                    </div>
                                    <div class="col-md-3">
                                        <label class="form-label">Learning Rate</label>
                                        <input type="number" class="form-control" id="learning-rate" value="0.001" step="0.0001">
                                    </div>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" id="start-training-btn">
                                <i class="bi bi-play-fill me-1"></i> Start Training
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        const existingModal = document.getElementById('startTrainingModal');
        if (existingModal) existingModal.remove();

        document.body.insertAdjacentHTML('beforeend', modalHTML);
        const modal = new bootstrap.Modal(document.getElementById('startTrainingModal'));

        // Load datasets and models
        this.loadDatasetsForModal();
        this.loadModelsForModal();

        // Handle form submission
        document.getElementById('start-training-btn').addEventListener('click', () => this.handleStartTraining());

        modal.show();
    }

    async loadDatasetsForModal() {
        try {
            const datasets = await apiService.getDatasets();
            const select = document.getElementById('dataset-select');
            datasets.forEach(dataset => {
                const option = document.createElement('option');
                option.value = dataset.id;
                option.textContent = `${dataset.name} (${dataset.total_images || 0} images)`;
                select.appendChild(option);
            });
        } catch (error) {
            console.error('Error loading datasets:', error);
        }
    }

    async loadModelsForModal() {
        try {
            const models = await apiService.get('/models/');
            const select = document.getElementById('model-select');

            // Add "Create New Model" option
            const newOption = document.createElement('option');
            newOption.value = 'new';
            newOption.textContent = '+ Create New Model';
            select.appendChild(newOption);

            // Add existing models
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = `${model.name} (${model.architecture || 'Unknown'})`;
                select.appendChild(option);
            });
        } catch (error) {
            console.error('Error loading models:', error);
        }
    }

    async handleStartTraining() {
        const name = document.getElementById('training-name').value;
        const datasetId = document.getElementById('dataset-select').value;
        const modelId = document.getElementById('model-select').value;
        const architecture = document.getElementById('architecture-select').value;

        if (!name || !datasetId || !modelId || !architecture) {
            showToast('Please fill in all required fields', 'error');
            return;
        }

        const hyperparameters = {
            epochs: parseInt(document.getElementById('epochs').value),
            batch: parseInt(document.getElementById('batch-size').value),
            imgsz: parseInt(document.getElementById('img-size').value)
        };

        try {
            // If "Create New Model" was selected, create model first
            let actualModelId = modelId;
            if (modelId === 'new') {
                const modelData = {
                    name: name,  // _model 접미사 제거
                    architecture: architecture,
                    description: `Model for ${name}`
                };
                const createdModel = await apiService.post('/models/', modelData);
                actualModelId = createdModel.id;
                showToast('Model created successfully', 'success');
            }

            // If using existing model as pretrained, add to hyperparameters
            if (modelId !== 'new' && modelId) {
                hyperparameters.pretrained_model_id = parseInt(modelId);
            }

            // Start training
            const newJob = await apiService.startTraining({
                name,
                dataset_id: parseInt(datasetId),
                architecture,
                hyperparameters
            });

            bootstrap.Modal.getInstance(document.getElementById('startTrainingModal')).hide();
            showToast('Training started successfully!', 'success');

            console.log('[Training Page] New training job created:', newJob);

            // Reload page data
            await this.loadTrainingJobs();

            // Select the newly created job
            if (newJob && newJob.id) {
                this.selectedJob = this.trainingJobs.find(j => j.id === newJob.id) || newJob;
                console.log('[Training Page] Auto-selected new job:', this.selectedJob?.name);

                // Subscribe to this job's training logs if RabbitMQ is connected
                if (this.rabbitmqConnected && newJob.external_job_id) {
                    console.log(`[Training Page] Subscribing to new job: ${newJob.external_job_id}`);
                    this.subscribeToJobLogs(newJob.external_job_id);
                } else if (!newJob.external_job_id) {
                    console.warn('[Training Page] New job has no external_job_id, cannot subscribe to logs');
                }
            }

            const app = document.getElementById('app');
            if (app) {
                app.innerHTML = this.render();
                await this.afterRender();
            }

        } catch (error) {
            console.error('Error starting training:', error);
            showToast('Failed to start training: ' + error.message, 'error');
        }
    }

    async deleteTrainingJob(jobId, modelId) {
        /**
         * Delete a training job and its associated model (including S3 artifacts)
         */
        try {
            // Find the job to get its name for confirmation
            const job = this.trainingJobs.find(j => j.id === jobId);
            if (!job) {
                showToast('Training job not found', 'error');
                return;
            }

            // Show Bootstrap modal for confirmation
            this.showDeleteConfirmModal(job, async () => {
                try {
                    console.log(`[Training Page] Deleting training job ${jobId} and model ${modelId}...`);

                    // Delete the training job (this will also delete associated model if exists)
                    await apiService.deleteTrainingJob(jobId);
                    showToast('Training job deleted successfully', 'success');

                    // If this was the selected job, clear selection
                    if (this.selectedJob?.id === jobId) {
                        this.selectedJob = null;
                    }

                    // Reload training jobs
                    await this.loadTrainingJobs();

                    // Re-render the page
                    const app = document.getElementById('app');
                    if (app) {
                        app.innerHTML = this.render();
                        await this.afterRender();
                    }

                    console.log(`[Training Page] Successfully deleted job ${jobId}`);

                } catch (error) {
                    console.error('[Training Page] Error deleting training job:', error);
                    showToast(error.message || 'Failed to delete training job', 'error');
                }
            });

        } catch (error) {
            console.error('[Training Page] Error deleting training job:', error);
            showToast('Failed to delete: ' + error.message, 'error');
        }
    }

    showDeleteConfirmModal(job, onConfirm) {
        /**
         * Show a Bootstrap modal for delete confirmation
         */
        // Remove existing modal if any
        const existingModal = document.getElementById('deleteTrainingModal');
        if (existingModal) {
            existingModal.remove();
        }

        // Create modal HTML
        const modalHTML = `
            <div class="modal fade" id="deleteTrainingModal" tabindex="-1" aria-labelledby="deleteTrainingModalLabel" aria-hidden="true">
                <div class="modal-dialog modal-dialog-centered">
                    <div class="modal-content">
                        <div class="modal-header border-0 pb-0">
                            <h5 class="modal-title fw-bold" id="deleteTrainingModalLabel">
                                <i class="bi bi-exclamation-triangle-fill text-danger me-2"></i>
                                Delete Training Job
                            </h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <p class="mb-3">Are you sure you want to delete <strong>"${job.name}"</strong>?</p>
                            <div class="alert alert-warning mb-0">
                                <p class="mb-2 fw-bold">This will permanently delete:</p>
                                <ul class="mb-0 ps-3">
                                    <li>Training job from database</li>
                                    <li>Associated model and all versions</li>
                                    <li>All model artifacts from S3</li>
                                </ul>
                                <p class="mt-3 mb-0 text-danger fw-bold">
                                    <i class="bi bi-exclamation-circle me-1"></i>
                                    This action cannot be undone.
                                </p>
                            </div>
                        </div>
                        <div class="modal-footer border-0">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-danger" id="confirmDeleteBtn">
                                <i class="bi bi-trash me-1"></i>
                                Delete
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Add modal to body
        document.body.insertAdjacentHTML('beforeend', modalHTML);

        // Get modal element and buttons
        const modalElement = document.getElementById('deleteTrainingModal');
        const confirmBtn = document.getElementById('confirmDeleteBtn');

        // Initialize Bootstrap modal
        const modal = new bootstrap.Modal(modalElement);

        // Handle confirm button click
        confirmBtn.addEventListener('click', () => {
            modal.hide();
            onConfirm();
        });

        // Clean up modal after it's hidden
        modalElement.addEventListener('hidden.bs.modal', () => {
            modalElement.remove();
        });

        // Show modal
        modal.show();
    }

    async syncCompletedStatus(event) {
        /**
         * Sync completed status by checking S3 for results.csv files
         * Useful for trainings that completed before train.done consumer was added
         */
        const syncBtn = event?.target || document.querySelector('button[onclick*="syncCompletedStatus"]');
        
        try {
            if (syncBtn) {
                syncBtn.disabled = true;
                syncBtn.innerHTML = '<i class="bi bi-arrow-clockwise me-1 spinner-border spinner-border-sm"></i> Syncing...';
            }

            const response = await apiService.syncCompletedStatus();
            
            if (response.updated_count > 0) {
                showToast(
                    `Synced ${response.updated_count} training job(s) to completed status`,
                    'success'
                );
                console.log('[Training Page] Synced jobs:', response.updated_jobs);
                
                // Reload training jobs
                const jobs = await apiService.getTrainingJobs();
                if (jobs) {
                    this.trainingJobs = Array.isArray(jobs) ? jobs : (jobs.jobs || jobs.data || []);
                    
                    // Sync로 변경된 job들의 S3 metrics를 명시적으로 로드
                    // RabbitMQ 연결 여부와 관계없이 S3에서 최신 데이터 가져오기
                    for (const updatedJobInfo of response.updated_jobs || []) {
                        const job = this.trainingJobs.find(j => j.id === updatedJobInfo.id);
                        if (job) {
                            console.log(`[Training Page] Loading S3 metrics for synced job: ${job.id} (${job.name})`);
                            await this.loadS3MetricsForJob(job);
                        }
                    }
                    
                    // Update selected job reference
                    if (this.selectedJob) {
                        const updated = this.trainingJobs.find(j => j.id === this.selectedJob.id);
                        if (updated) {
                            this.selectedJob = updated;
                        }
                    }
                }
                
                // Re-render only the jobs list without destroying the chart
                this.updateJobsListDisplay();
            } else {
                showToast('No training jobs need status update', 'info');
            }
        } catch (error) {
            console.error('[Training Page] Error syncing completed status:', error);
            showToast('Failed to sync status: ' + error.message, 'error');
        } finally {
            if (syncBtn) {
                syncBtn.disabled = false;
                syncBtn.innerHTML = '<i class="bi bi-arrow-clockwise me-1"></i> Sync Status';
            }
        }
    }

    hasHyperparameters(job) {
        /**
         * Check if hyperparameters are available for this job
         * Try multiple ways to match job_id
         */
        if (!job) return false;
        
        // Try multiple possible IDs
        const possibleIds = [
            String(job.id),
            job.external_job_id ? String(job.external_job_id) : null,
            job.name
        ].filter(id => id);
        
        // Check if any of the stored hyperparameters match
        for (const id of possibleIds) {
            if (this.hyperparameters[id] !== undefined) {
                console.log(`[Training Page] Found hyperparameters for job ${job.id} using key: ${id}`);
                return true;
            }
        }
        
        // Also check in job.hyperparameters for external_job_id
        if (job.hyperparameters) {
            let hyperparams = {};
            if (typeof job.hyperparameters === 'string') {
                try {
                    hyperparams = JSON.parse(job.hyperparameters);
                } catch (e) {
                    // Ignore parse errors
                }
            } else {
                hyperparams = job.hyperparameters;
            }
            
            if (hyperparams.external_job_id) {
                const extId = String(hyperparams.external_job_id);
                if (this.hyperparameters[extId] !== undefined) {
                    console.log(`[Training Page] Found hyperparameters for job ${job.id} using external_job_id: ${extId}`);
                    return true;
                }
            }
        }
        
        return false;
    }

    getJobIdForHyperparams(job) {
        /**
         * Get job_id for hyperparameters lookup
         * Try multiple ways to find matching hyperparameters
         */
        if (!job) return null;
        
        // Try multiple possible IDs
        const possibleIds = [
            String(job.id),
            job.external_job_id ? String(job.external_job_id) : null,
            job.name
        ].filter(id => id);
        
        // Find the first matching ID
        for (const id of possibleIds) {
            if (this.hyperparameters[id] !== undefined) {
                return id;
            }
        }
        
        // Also check in job.hyperparameters for external_job_id
        if (job.hyperparameters) {
            let hyperparams = {};
            if (typeof job.hyperparameters === 'string') {
                try {
                    hyperparams = JSON.parse(job.hyperparameters);
                } catch (e) {
                    // Ignore parse errors
                }
            } else {
                hyperparams = job.hyperparameters;
            }
            
            if (hyperparams.external_job_id) {
                const extId = String(hyperparams.external_job_id);
                if (this.hyperparameters[extId] !== undefined) {
                    return extId;
                }
            }
        }
        
        // Fallback to job.id as string
        return job.id ? String(job.id) : null;
    }

    updateHyperparameterButton(jobId) {
        /**
         * Update UI to show hyperparameter button for the job
         */
        console.log(`[Training Page] Updating hyperparameter button for job_id: ${jobId}`);
        console.log(`[Training Page] Available jobs:`, this.trainingJobs.map(j => ({ id: j.id, name: j.name })));
        console.log(`[Training Page] Stored hyperparameters keys:`, Object.keys(this.hyperparameters));
        
        // Find all jobs that might match this job_id
        const matchingJobs = this.trainingJobs.filter(j => {
            const possibleIds = [
                String(j.id),
                j.external_job_id ? String(j.external_job_id) : null,
                j.name
            ].filter(id => id);
            
            return possibleIds.includes(String(jobId));
        });
        
        console.log(`[Training Page] Found ${matchingJobs.length} matching jobs for job_id ${jobId}`);
        
        if (matchingJobs.length > 0) {
            // Re-render the jobs list to show the button
            console.log(`[Training Page] Re-rendering jobs list to show hyperparameter button`);
            this.updateJobsListDisplay();
        } else {
            console.warn(`[Training Page] No matching job found for job_id: ${jobId}`);
            // Still update the display in case the job appears later
            this.updateJobsListDisplay();
        }
    }

    showHyperparameterModal(jobId) {
        /**
         * Show modal with hyperparameters
         */
        const hyperparams = this.hyperparameters[jobId];
        
        if (!hyperparams) {
            showToast('Hyperparameters not available for this job', 'warning');
            return;
        }

        // Find the job for display name
        const job = this.trainingJobs.find(j => {
            const id = this.getJobIdForHyperparams(j);
            return id === jobId;
        });

        const jobName = job ? job.name : 'Unknown Job';

        // Create modal HTML
        const modalHTML = `
            <div class="modal fade" id="hyperparameterModal" tabindex="-1">
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
                                ${this.renderHyperparameterFields(hyperparams)}
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
        const existingModal = document.getElementById('hyperparameterModal');
        if (existingModal) {
            existingModal.remove();
        }

        // Add modal to body
        document.body.insertAdjacentHTML('beforeend', modalHTML);

        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('hyperparameterModal'));
        modal.show();
    }

    renderHyperparameterFields(hyperparams) {
        /**
         * Render hyperparameter fields in a nice format
         */
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
}

// Make instance globally accessible
window.trainingPage = null;
