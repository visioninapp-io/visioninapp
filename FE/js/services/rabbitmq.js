// RabbitMQ STOMP Service - Direct consume from frontend
// Uses RabbitMQ Web STOMP plugin (port 15674)

class RabbitMQService {
    constructor() {
        this.stompClient = null;
        this.connected = false;
        this.subscriptions = {};
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 3000; // 3 seconds

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
            if (this.connected) {
                console.log('[RabbitMQ] Already connected');
                resolve();
                return;
            }

            try {
                // WebSocket URL for RabbitMQ Web STOMP
                const wsUrl = `ws://${this.config.host}:${this.config.port}/ws`;
                console.log('[RabbitMQ] Connecting to:', wsUrl);

                // Create WebSocket
                const socket = new WebSocket(wsUrl);

                // Create STOMP client
                this.stompClient = Stomp.over(socket);

                // Disable debug logging (optional)
                this.stompClient.debug = (msg) => {
                    console.log('[RabbitMQ Debug]', msg);
                };

                // Connect with credentials
                this.stompClient.connect(
                    this.config.username,
                    this.config.password,
                    () => {
                        console.log('[RabbitMQ] ✅ Connected successfully');
                        this.connected = true;
                        this.reconnectAttempts = 0; // Reset reconnect counter
                        resolve();
                    },
                    (error) => {
                        console.error('[RabbitMQ] ❌ Connection error:', error);
                        this.connected = false;

                        // Auto-reconnect
                        this.attemptReconnect();
                        reject(error);
                    },
                    this.config.vhost
                );

            } catch (error) {
                console.error('[RabbitMQ] Connection failed:', error);
                reject(error);
            }
        });
    }

    /**
     * Subscribe to a queue
     * @param {string} queueName - Queue name (e.g., 'gpu.train.log')
     * @param {function} callback - Message handler
     * @returns {string} Subscription ID
     */
    subscribe(queueName, callback) {
        if (!this.connected) {
            console.error('[RabbitMQ] Not connected. Call connect() first.');
            throw new Error('Not connected to RabbitMQ');
        }

        console.log(`[RabbitMQ] Subscribing to queue: ${queueName}`);

        // STOMP destination for queue
        const destination = `/queue/${queueName}`;

        // Subscribe with auto ACK (don't manually ack)
        const subscription = this.stompClient.subscribe(destination, (message) => {
            try {
                console.log(`[RabbitMQ] Message received from ${queueName}`);

                // Parse JSON message
                const body = JSON.parse(message.body);
                console.log(`[RabbitMQ] Parsed message:`, body);

                // Call callback
                callback(body);

                // Don't manually ACK - STOMP auto-acks by default
                console.log(`[RabbitMQ] ✅ Message processed successfully`);

            } catch (error) {
                console.error('[RabbitMQ] Error processing message:', error);
                // Log but don't throw - keep connection alive
            }
        }, {
            ack: 'auto'  // Auto acknowledgement mode
        });

        // Store subscription
        this.subscriptions[queueName] = subscription;
        console.log(`[RabbitMQ] ✅ Subscribed to ${queueName} with auto-ack`);

        return subscription.id;
    }

    /**
     * Unsubscribe from a queue
     * @param {string} queueName - Queue name
     */
    unsubscribe(queueName) {
        const subscription = this.subscriptions[queueName];
        if (subscription) {
            subscription.unsubscribe();
            delete this.subscriptions[queueName];
            console.log(`[RabbitMQ] Unsubscribed from ${queueName}`);
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
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('[RabbitMQ] Max reconnection attempts reached. Giving up.');
            return;
        }

        this.reconnectAttempts++;
        console.log(`[RabbitMQ] Reconnecting... (Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

        setTimeout(() => {
            this.connect()
                .then(() => {
                    console.log('[RabbitMQ] ✅ Reconnected successfully');
                    // Re-subscribe to all previous subscriptions
                    // (subscriptions are stored, so they'll be re-activated)
                })
                .catch((error) => {
                    console.error('[RabbitMQ] Reconnection failed:', error);
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
