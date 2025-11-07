# RabbitMQ Pure Message-Driven AI Service

The AI service operates as a **pure message-driven architecture** handling ALL operations (training, conversion, inference) via RabbitMQ messages only.

## Architecture

```
Backend (BE) ──RabbitMQ──► AI Service (Multi-Consumer)
             │                       │
             │              ┌────────┼────────┐
             │              ▼        ▼        ▼
             │         Training  Conversion  Inference
             │              │        │        │
             └─RabbitMQ◄────┴────────┴────────┘
             ▼
Backend (BE) Consumer ◄── Real-time Updates & Results
```

### Message Flow:
1. **Job Submission**: BE → RabbitMQ → AI (training/conversion/inference)
2. **Real-time Updates**: AI → RabbitMQ → BE (progress, results, errors)
3. **No HTTP**: Pure message-driven communication only

## Configuration

### Environment Variables

Set these environment variables to configure RabbitMQ:

```bash
# RabbitMQ Configuration
ENABLE_RABBITMQ=true
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest
RABBITMQ_VHOST=/
RABBITMQ_SSL=false

# Queue Configuration
TRAIN_REQUEST_QUEUE=train_request_q
TRAIN_RESULT_QUEUE=train_result_q
TRAIN_UPDATE_QUEUE=train_update_q
```

### Production Configuration

For production, use secure credentials:

```bash
RABBITMQ_HOST=your-rabbitmq-server.com
RABBITMQ_USER=your-username
RABBITMQ_PASSWORD=your-secure-password
RABBITMQ_SSL=true
```

## Message Formats

### Training Request (BE → AI)
The AI service expects training requests in this format:

```json
{
  "job_id": "abc12345",
  "dataset": {
    "s3_prefix": "datasets/my-dataset/",
    "name": "my-dataset"
  },
  "output": {
    "prefix": "models/yolov8n",
    "model_name": "my-dataset.pt"
  },
  "hyperparams": {
    "model": "yolov8n",
    "epochs": 100,
    "batch": 16,
    "imgsz": 640
  }
}
```

### Epoch Update (AI → BE)
Real-time epoch progress updates:

```json
{
  "type": "epoch_update",
  "job_id": "abc12345",
  "status": "running",
  "current_epoch": 15,
  "total_epochs": 100,
  "progress": 15.0,
  "metrics": {
    "train_loss": 0.0234,
    "mAP50": 0.789,
    "box_loss": 0.0123,
    "cls_loss": 0.0089,
    "dfl_loss": 0.0022
  },
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

### Training Completion (AI → BE)
Final training results:

```json
{
  "type": "training_completed",
  "job_id": "abc12345",
  "status": "completed",
  "progress": 100.0,
  "model_path": "uploads/models/model_19/weights/best.pt",
  "final_metrics": {
    "final_mAP50": 0.892,
    "final_loss": 0.0156,
    "training_epochs": 100
  },
  "training_time": 3600.5,
  "timestamp": "2024-01-15T11:30:45.123456"
}
```

### Training Failure (AI → BE)
Error notifications:

```json
{
  "type": "training_failed",
  "job_id": "abc12345",
  "status": "failed",
  "current_epoch": 23,
  "error_message": "CUDA out of memory",
  "partial_metrics": {
    "last_loss": 0.0234,
    "last_mAP50": 0.567
  },
  "timestamp": "2024-01-15T10:45:30.123456"
}
```

## Features

### Dual Communication Support
- **HTTP Endpoints**: Direct API calls for immediate responses
- **RabbitMQ Consumer**: Async message processing for background jobs

### Automatic Fallback
- If RabbitMQ is not configured, AI service continues with HTTP-only mode
- No breaking changes to existing HTTP endpoints

### Error Handling
- Message acknowledgment/rejection based on processing success
- Failed messages can be requeued or sent to dead letter queue

## Testing

### 1. Start AI Service
```bash
cd AI
python main.py
```

### 2. Check RabbitMQ Status
Look for these log messages:
```
🐰 Initializing RabbitMQ consumer...
✅ RabbitMQ consumer started
🚀 Started RabbitMQ consumer in background thread
```

### 3. Send Test Message
Use the backend's training endpoint, which will send messages via RabbitMQ.

## Troubleshooting

### RabbitMQ Connection Failed
```
❌ Failed to connect to RabbitMQ: [Errno 111] Connection refused
```
**Solution**: Check if RabbitMQ server is running and accessible.

### Authentication Failed
```
❌ Failed to connect to RabbitMQ: (403) ACCESS_REFUSED
```
**Solution**: Check RABBITMQ_USER and RABBITMQ_PASSWORD.

### Queue Not Found
```
❌ Queue 'train_request_q' not found
```
**Solution**: The consumer automatically declares the queue, but check TRAIN_REQUEST_QUEUE setting.

## Development vs Production

### Development (Default)
- Uses localhost RabbitMQ with guest/guest credentials
- Logs warnings about insecure configuration
- Still functional for testing

### Production
- Requires proper RabbitMQ server configuration
- Uses secure credentials and SSL
- Proper error handling and monitoring

## Monitoring

The AI service logs all RabbitMQ activity:
- Connection status
- Message processing
- Errors and retries
- Training job status

Check logs for `[RMQ]` and `[Handler]` prefixed messages.
