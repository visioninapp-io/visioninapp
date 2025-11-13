// RabbitMQ STOMP Service - Direct consume from frontend
// Uses RabbitMQ Web STOMP plugin (port 15674)

class RabbitMQService {
    constructor() {
        this.stompClient = null;
        this.connected = false;
        this.subscriptions = {};
        this.pendingSubscriptions = []; // 재연결 시 재구독할 구독 정보 저장
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 3000; // 3 seconds
        this.reconnectTimer = null;
        this.socket = null;

        // RabbitMQ Web STOMP configuration from config file
        this.config = RABBITMQ_CONFIG;

        console.log('[RabbitMQ] Service initialized with config:', {
            host: this.config.host,
            port: this.config.port,
            vhost: this.config.vhost
        });
    }

    /**
     * Connect to RabbitMQ Web STOMP
     */
    connect() {
        return new Promise((resolve, reject) => {
            if (this.connected && this.stompClient && this.stompClient.connected) {
                console.log('[RabbitMQ] Already connected');
                resolve();
                return;
            }

            // 기존 연결 정리
            if (this.stompClient) {
                try {
                    this.stompClient.disconnect();
                } catch (e) {
                    // Ignore
                }
            }
            if (this.socket) {
                try {
                    this.socket.close();
                } catch (e) {
                    // Ignore
                }
            }

            try {
                // WebSocket URL for RabbitMQ Web STOMP (two method for Nginx Serving with https)
                const isHttps  = window.location.protocol === 'https:';
                const protocol = isHttps ? 'wss' : 'ws';
                const portPart = isHttps ? '' : `:${this.config.port}`;
                const wsUrl    = `${protocol}://${this.config.host}${portPart}/ws`;

                console.log('[RabbitMQ] Connecting to:', wsUrl);

                // Create WebSocket
                this.socket = new WebSocket(wsUrl);

                // WebSocket 이벤트 리스너 추가
                this.socket.onopen = () => {
                    console.log('[RabbitMQ] WebSocket opened');
                };

                this.socket.onerror = (error) => {
                    console.error('[RabbitMQ] WebSocket error:', error);
                };

                this.socket.onclose = (event) => {
                    console.warn('[RabbitMQ] WebSocket closed:', event.code, event.reason);
                    this.connected = false;
                    
                    // 예상치 못한 연결 끊김 시 재연결
                    if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
                        console.log('[RabbitMQ] Unexpected disconnect, attempting reconnect...');
                        this.attemptReconnect();
                    }
                };

                // Create STOMP client
                this.stompClient = Stomp.over(this.socket);

                // STOMP 디버그 로깅 (필요시만 활성화)
                this.stompClient.debug = (msg) => {
                    // 중요한 메시지만 로깅 (너무 많은 로그 방지)
                    if (msg.includes('CONNECTED') || msg.includes('ERROR') || msg.includes('MESSAGE')) {
                        console.log('[RabbitMQ Debug]', msg);
                    }
                };

                // 연결 타임아웃 설정
                const connectionTimeout = setTimeout(() => {
                    if (!this.connected) {
                        console.error('[RabbitMQ] Connection timeout');
                        this.connected = false;
                        reject(new Error('Connection timeout'));
                    }
                }, 10000); // 10초 타임아웃

                // Connect with credentials and heartbeat
                this.stompClient.connect(
                    this.config.username,
                    this.config.password,
                    (frame) => {
                        clearTimeout(connectionTimeout);
                        console.log('[RabbitMQ] ✅ Connected successfully');
                        console.log('[RabbitMQ] Connection frame:', frame);
                        this.connected = true;
                        this.reconnectAttempts = 0; // Reset reconnect counter
                        
                        // 재연결 후 이전 구독 복구
                        this.resubscribeAll();
                        
                        resolve();
                    },
                    (error) => {
                        clearTimeout(connectionTimeout);
                        console.error('[RabbitMQ] ❌ STOMP connection error:', error);
                        console.error('[RabbitMQ] Connection error details:', {
                            message: error?.message,
                            stack: error?.stack,
                            fullError: error
                        });
                        this.connected = false;

                        // Auto-reconnect (에러 콜백에서는 재연결하지 않음 - WebSocket onclose에서 처리)
                        reject(error);
                    },
                    this.config.vhost,
                    { heartbeat: { incoming: 10000, outgoing: 10000 } } // Heartbeat 설정
                );

            } catch (error) {
                console.error('[RabbitMQ] Connection failed:', error);
                this.connected = false;
                reject(error);
            }
        });
    }

    /**
     * Subscribe to a queue or exchange
     * @param {string} destination - Queue name or routing key (e.g., 'gpu.train.log' or 'job.progress.train.download_dataset')
     * @param {function} callback - Message handler
     * @param {string} type - 'queue' or 'exchange' (default: 'queue')
     * @param {string} exchange - Exchange name (required if type is 'exchange', default: 'jobs.events')
     * @returns {string} Subscription ID
     */
    subscribe(destination, callback, type = 'queue', exchange = 'jobs.events') {
        if (!this.connected || !this.stompClient || !this.stompClient.connected) {
            console.error('[RabbitMQ] Not connected. Call connect() first.');
            // 연결되지 않았으면 구독 정보를 저장해두고 나중에 재구독
            this.pendingSubscriptions.push({ destination, callback, type, exchange });
            throw new Error('Not connected to RabbitMQ');
        }

        let stompDestination;
        if (type === 'exchange') {
            // STOMP destination for exchange: /exchange/{exchange}/{routing_key}
            stompDestination = `/exchange/${exchange}/${destination}`;
            console.log(`[RabbitMQ] Subscribing to exchange: ${exchange} with routing key: ${destination}`);
        } else {
            // STOMP destination for queue
            stompDestination = `/queue/${destination}`;
            console.log(`[RabbitMQ] Subscribing to queue: ${destination}`);
        }

        try {
            // Subscribe with auto ACK (don't manually ack)
            console.log(`[RabbitMQ] Setting up subscription callback for ${stompDestination}`);
            const subscription = this.stompClient.subscribe(stompDestination, (message) => {
                try {
                    console.log(`[RabbitMQ] 📨 Raw message received from ${stompDestination}`);

                    // Parse JSON message
                    let body;
                    try {
                        body = JSON.parse(message.body);
                    } catch (parseError) {
                        console.error('[RabbitMQ] ❌ Failed to parse message as JSON:', parseError);
                        console.error('[RabbitMQ] Message body (as string):', message.body);
                        return;
                    }
                    
                    console.log(`[RabbitMQ] ✅ Parsed message:`, body);

                    // Call callback with error handling
                    try {
                        callback(body);
                        console.log(`[RabbitMQ] ✅ Callback executed successfully`);
                    } catch (callbackError) {
                        console.error('[RabbitMQ] ❌ Error in callback function:', callbackError);
                        console.error('[RabbitMQ] Callback error stack:', callbackError.stack);
                        // Don't throw - keep connection alive
                    }

                } catch (error) {
                    console.error('[RabbitMQ] ❌ Error processing message:', error);
                    console.error('[RabbitMQ] Error stack:', error.stack);
                    // Log but don't throw - keep connection alive
                }
            }, {
                ack: 'auto'  // Auto acknowledgement mode
            });

            // Store subscription with destination as key
            const subscriptionInfo = {
                subscription: subscription,
                destination: destination,
                callback: callback,
                type: type,
                exchange: exchange,
                stompDestination: stompDestination
            };
            this.subscriptions[destination] = subscriptionInfo;
            
            // 구독 정보를 pending에서도 저장 (재연결 시 사용)
            const existingPendingIndex = this.pendingSubscriptions.findIndex(
                sub => sub.destination === destination && sub.type === type && sub.exchange === exchange
            );
            if (existingPendingIndex === -1) {
                this.pendingSubscriptions.push({ destination, callback, type, exchange });
            }
            
            console.log(`[RabbitMQ] ✅ Subscribed to ${stompDestination} with auto-ack`);

            return subscription.id;
        } catch (error) {
            console.error(`[RabbitMQ] ❌ Failed to subscribe to ${stompDestination}:`, error);
            // 구독 실패 시 pending에 추가
            this.pendingSubscriptions.push({ destination, callback, type, exchange });
            throw error;
        }
    }
    
    /**
     * 재연결 후 모든 구독 복구
     */
    resubscribeAll() {
        console.log(`[RabbitMQ] Resubscribing to ${this.pendingSubscriptions.length} subscriptions...`);
        const subscriptionsToResubscribe = [...this.pendingSubscriptions];
        this.pendingSubscriptions = [];
        
        subscriptionsToResubscribe.forEach(({ destination, callback, type, exchange }) => {
            try {
                this.subscribe(destination, callback, type, exchange);
                console.log(`[RabbitMQ] ✅ Resubscribed to ${destination}`);
            } catch (error) {
                console.error(`[RabbitMQ] ❌ Failed to resubscribe to ${destination}:`, error);
                // 실패한 구독은 다시 pending에 추가
                this.pendingSubscriptions.push({ destination, callback, type, exchange });
            }
        });
    }

    /**
     * Unsubscribe from a queue or exchange
     * @param {string} destination - Queue name or routing key
     */
    unsubscribe(destination) {
        const subscriptionInfo = this.subscriptions[destination];
        if (subscriptionInfo && subscriptionInfo.subscription) {
            try {
                subscriptionInfo.subscription.unsubscribe();
                delete this.subscriptions[destination];
                console.log(`[RabbitMQ] ✅ Unsubscribed from ${destination}`);
            } catch (error) {
                console.error(`[RabbitMQ] ❌ Error unsubscribing from ${destination}:`, error);
                // 구독 정보는 삭제 (이미 구독이 해제되었을 수 있음)
                delete this.subscriptions[destination];
            }
        } else {
            console.warn(`[RabbitMQ] ⚠️ Subscription not found for ${destination}`);
        }
    }

    /**
     * Disconnect from RabbitMQ
     */
    disconnect() {
        if (this.stompClient && this.connected) {
            // Unsubscribe all
            Object.keys(this.subscriptions).forEach(queueName => {
                this.unsubscribe(queueName);
            });

            // Disconnect
            this.stompClient.disconnect(() => {
                console.log('[RabbitMQ] Disconnected');
                this.connected = false;
            });
        }
    }

    /**
     * Attempt to reconnect to RabbitMQ
     */
    attemptReconnect() {
        // 이미 재연결 중이면 중복 방지
        if (this.reconnectTimer) {
            return;
        }
        
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('[RabbitMQ] Max reconnection attempts reached. Giving up.');
            return;
        }

        this.reconnectAttempts++;
        console.log(`[RabbitMQ] Reconnecting... (Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.connect()
                .then(() => {
                    console.log('[RabbitMQ] ✅ Reconnected successfully');
                    // resubscribeAll()이 connect() 내부에서 호출됨
                })
                .catch((error) => {
                    console.error('[RabbitMQ] Reconnection failed:', error);
                    // 재연결 실패 시 다시 시도
                    if (this.reconnectAttempts < this.maxReconnectAttempts) {
                        this.attemptReconnect();
                    }
                });
        }, this.reconnectDelay);
    }

    /**
     * Subscribe to training logs (gpu.train.log queue)
     * @param {function} callback - Handler for training metrics
     */
    subscribeToTrainingLogs(callback) {
        return this.subscribe('gpu.train.log', callback);
    }
}

// Create singleton instance
const rabbitmqService = new RabbitMQService();
window.rabbitmqService = rabbitmqService;

console.log('[RabbitMQ Service] Ready');
