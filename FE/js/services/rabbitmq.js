// RabbitMQ STOMP Service - Direct consume from frontend
// Uses RabbitMQ Web STOMP plugin (port 15674)

class RabbitMQService {
    constructor() {
        this.stompClient = null;
        this.connected = false;
        this.subscriptions = {};

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
                        resolve();
                    },
                    (error) => {
                        console.error('[RabbitMQ] ❌ Connection error:', error);
                        this.connected = false;
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

        // Subscribe
        const subscription = this.stompClient.subscribe(destination, (message) => {
            try {
                console.log(`[RabbitMQ] Message received from ${queueName}`);

                // Parse JSON message
                const body = JSON.parse(message.body);
                console.log(`[RabbitMQ] Parsed message:`, body);

                // Call callback
                callback(body);

                // Acknowledge message
                message.ack();
            } catch (error) {
                console.error('[RabbitMQ] Error processing message:', error);
            }
        });

        // Store subscription
        this.subscriptions[queueName] = subscription;
        console.log(`[RabbitMQ] ✅ Subscribed to ${queueName}`);

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
