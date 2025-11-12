// Training Page Component - Real-time Training Monitoring

class TrainingPage {
    constructor() {
        this.trainingJobs = [];
        this.selectedJob = null;
        this.chart = null;
        this.rabbitmqConnected = false;
        this.metricsData = {}; // Store real-time metrics by job_id
    }

    async init() {
        console.log('[Training Page] Initializing...');

        try {
            await this.loadTrainingJobs();

            // Render the page
            const app = document.getElementById('app');
            if (app) {
                app.innerHTML = this.render();
                await this.afterRender();
            }

            this.attachEventListeners();

            // Connect to RabbitMQ and subscribe to training logs
            await this.connectToRabbitMQ();

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

            // Subscribe to training logs queue
            rabbitmqService.subscribeToTrainingLogs((message) => {
                this.handleTrainingMetrics(message);
            });

            console.log('[Training Page] ✅ Connected to RabbitMQ and subscribed to gpu.train.log');
            showToast('Connected to real-time training updates', 'success');

        } catch (error) {
            console.error('[Training Page] Failed to connect to RabbitMQ:', error);
            showToast('Failed to connect to real-time updates. Using manual refresh.', 'warning');
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

            // Just update the display with received metrics (no job matching)
            this.updateMetricsDisplay({ epoch, loss, accuracy: accuracy * 100 });

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
                console.warn('[Training Page] Chart not initialized, skipping chart update');
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

    destroy() {
        /**
         * Cleanup when page is destroyed
         */
        console.log('[Training Page] Destroying...');

        // Disconnect from RabbitMQ
        if (this.rabbitmqConnected) {
            rabbitmqService.unsubscribe('gpu.train.log');
            console.log('[Training Page] Unsubscribed from RabbitMQ');
        }

        // Clear chart
        if (this.chart) {
            this.chart.destroy();
            this.chart = null;
        }
    }

    async loadTrainingJobs() {
        try {
            console.log('[Training Page] Loading training jobs...');
            const jobs = await apiService.getTrainingJobs();

            this.trainingJobs = jobs || [];
            console.log('[Training Page] Loaded jobs:', this.trainingJobs.length);

            this.trainingJobs.forEach(job => {
                console.log(`  Job ${job.id}: ${job.name} - Status: ${job.status}, Epoch: ${job.current_epoch || 0}/${job.total_epochs || 0}`);
            });

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

            if (this.selectedJob) {
                console.log('[Training Page] Selected job:', this.selectedJob.id, this.selectedJob.name);
            }

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
                                        ${this.trainingJobs.map(job => `
                                            <option value="${job.id}" ${selectedJob?.id === job.id ? 'selected' : ''}>
                                                ${job.name} - ${job.status.toUpperCase()} (${job.current_epoch || 0}/${job.total_epochs || 0} epochs)
                                            </option>
                                        `).join('')}
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
                            </div>
                        </div>
                        <div class="card-body" id="jobs-container">
                            ${this.trainingJobs.length > 0 ? this.renderTrainingJobs() : this.renderEmptyState()}
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
            const progress = job.total_epochs > 0
                ? Math.round((job.current_epoch || 0) / job.total_epochs * 100)
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
                            <span class="badge ${statusBadge}">${job.status.toUpperCase()}</span>
                        </div>

                        <div class="mb-3">
                            <div class="d-flex justify-content-between mb-2">
                                <span class="text-muted small">Progress</span>
                                <span class="fw-medium small">${job.current_epoch || 0}/${job.total_epochs || 0} epochs</span>
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

    selectJob(jobId) {
        const id = parseInt(jobId);
        const job = this.trainingJobs.find(j => j.id === id);

        if (job) {
            console.log('[Training Page] Selected job:', job.id, job.name);
            this.selectedJob = job;

            // Re-render the page
            const app = document.getElementById('app');
            if (app) {
                app.innerHTML = this.render();
                this.afterRender();
            }

        }
    }

    async afterRender() {
        console.log('[Training Page] afterRender called');
        await this.initChart();
        console.log('[Training Page] Chart initialized, creating FAB...');
        // FAB 버튼 생성 (afterRender 완료 후)
        this.createFAB();
    }

    async initChart() {
        if (!this.selectedJob) {
            console.log('[Training Page] No selected job for chart');
            return;
        }

        try {
            console.log('[Training Page] Initializing chart for job', this.selectedJob.id);
            const metrics = await apiService.getTrainingMetrics(this.selectedJob.id);
            console.log('[Training Page] Loaded metrics:', metrics?.length || 0);
            console.log('[Training Page] Raw metrics data:', metrics);

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

            // Create chart only if we have metrics
            if (!metrics || metrics.length === 0) {
                console.log('[Training Page] No metrics to display');
                const context = ctx.getContext('2d');
                context.fillStyle = '#6c757d';
                context.font = '16px sans-serif';
                context.fillText('No metrics available yet. Metrics will appear once training starts.', 50, 100);
                return;
            }

            // Extract chart data
            const labels = metrics.map((m, idx) => `Epoch ${m.epoch || idx + 1}`);
            const lossData = metrics.map(m => m.train_loss || 0);
            const accuracyData = metrics.map(m => (m.train_accuracy || 0));
            
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
            this.chart.data.labels = metrics.map((m, idx) => `Epoch ${m.epoch || idx + 1}`);
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

        // FAB 버튼 생성
        const fabButton = document.createElement('button');
        fabButton.id = 'llm-fab-button';
        fabButton.className = 'btn btn-info rounded-circle shadow-lg';
        fabButton.innerHTML = '<i class="bi bi-robot fs-4"></i>';
        fabButton.title = 'LLM 모델 변환';
        
        // CSS 스타일 적용
        fabButton.style.cssText = `
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 64px;
            height: 64px;
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1050;
            border: none;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, #3b82f6, #0ea5e9);
        `;

        // 호버 효과
        fabButton.addEventListener('mouseenter', () => {
            fabButton.style.transform = 'scale(1.1)';
            fabButton.style.boxShadow = '0 8px 16px rgba(59, 130, 246, 0.4)';
        });

        fabButton.addEventListener('mouseleave', () => {
            fabButton.style.transform = 'scale(1)';
            fabButton.style.boxShadow = '';
        });

        fabButton.addEventListener('mousedown', () => {
            fabButton.style.transform = 'scale(0.95)';
        });

        fabButton.addEventListener('mouseup', () => {
            fabButton.style.transform = 'scale(1)';
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
            batch_size: parseInt(document.getElementById('batch-size').value),
            img_size: parseInt(document.getElementById('img-size').value),
            learning_rate: parseFloat(document.getElementById('learning-rate').value),
            num_classes: 10
        };

        try {
            // If "Create New Model" was selected, create model first
            let actualModelId = modelId;
            if (modelId === 'new') {
                const modelData = {
                    name: `${name}_model`,
                    architecture: architecture,
                    description: `Model for ${name}`
                };
                const createdModel = await apiService.post('/models/', modelData);
                actualModelId = createdModel.id;
                showToast('Model created successfully', 'success');
            }

            // Start training
            await apiService.startTraining({
                name,
                dataset_id: parseInt(datasetId),
                architecture,
                hyperparameters
            });

            bootstrap.Modal.getInstance(document.getElementById('startTrainingModal')).hide();
            showToast('Training started successfully!', 'success');

            // Reload page data
            await this.loadTrainingJobs();
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
}

// Make instance globally accessible
window.trainingPage = null;
