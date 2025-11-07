"""
RabbitMQ Producer for AI Service
Sends training updates and results back to the backend
"""

import json
import logging
import os
from typing import Dict, Any, Optional
import pika
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

class RabbitMQProducer:
    """RabbitMQ producer for sending training updates back to BE"""
    
    def __init__(self, 
                 rabbitmq_url: Optional[str] = None,
                 result_queue: str = "train_result_q",
                 update_queue: str = "train_update_q"):
        """
        Initialize RabbitMQ producer
        
        Args:
            rabbitmq_url: RabbitMQ connection URL (default: from env vars)
            result_queue: Queue name for final results
            update_queue: Queue name for epoch updates
        """
        self.result_queue = result_queue
        self.update_queue = update_queue
        self.connection = None
        self.channel = None
        
        # RabbitMQ connection parameters (same as consumer)
        self.rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
        self.rabbitmq_port = int(os.getenv("RABBITMQ_PORT", "5672"))
        self.rabbitmq_user = os.getenv("RABBITMQ_USER", "guest")
        self.rabbitmq_password = os.getenv("RABBITMQ_PASSWORD", "guest")
        self.rabbitmq_vhost = os.getenv("RABBITMQ_VHOST", "/")
        
        logger.info(f"RabbitMQ Producer initialized")
        logger.info(f"Result Queue: {result_queue}, Update Queue: {update_queue}")
    
    def connect(self):
        """Establish connection to RabbitMQ"""
        try:
            credentials = pika.PlainCredentials(self.rabbitmq_user, self.rabbitmq_password)
            parameters = pika.ConnectionParameters(
                host=self.rabbitmq_host,
                port=self.rabbitmq_port,
                virtual_host=self.rabbitmq_vhost,
                credentials=credentials,
                heartbeat=30,
                blocked_connection_timeout=30,
                connection_attempts=3,
                retry_delay=1,
                socket_timeout=5
            )
            
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Declare queues (make sure they exist)
            self.channel.queue_declare(queue=self.result_queue, durable=True)
            self.channel.queue_declare(queue=self.update_queue, durable=True)
            
            # Enable delivery confirmation
            self.channel.confirm_delivery()
            
            logger.info(f" Producer connected to RabbitMQ: {self.rabbitmq_host}:{self.rabbitmq_port}")
            return True
            
        except Exception as e:
            logger.error(f" Failed to connect RabbitMQ producer: {e}")
            return False
    
    def disconnect(self):
        """Close RabbitMQ connection"""
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.close()
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            logger.info(" Producer disconnected from RabbitMQ")
        except Exception as e:
            logger.error(f"Error disconnecting producer: {e}")
    
    def _publish_message(self, queue: str, message: Dict[str, Any]) -> bool:
        """
        Publish message to specified queue
        
        Args:
            queue: Queue name
            message: Message to send
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure connection
            if not self.connection or self.connection.is_closed:
                if not self.connect():
                    return False
            
            # Add timestamp
            message["timestamp"] = datetime.utcnow().isoformat()
            
            # Serialize message
            body = json.dumps(message, ensure_ascii=False)
            
            # Message properties
            properties = pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2,  # Make message persistent
                timestamp=int(datetime.utcnow().timestamp())
            )
            
            # Publish message
            confirmed = self.channel.basic_publish(
                exchange="",
                routing_key=queue,
                body=body,
                properties=properties,
                mandatory=True
            )
            
            if confirmed:
                logger.info(f"[Producer]  Sent to {queue}: job_id={message.get('job_id')}")
                return True
            else:
                logger.error(f"[Producer]  Failed to confirm delivery to {queue}")
                return False
                
        except Exception as e:
            logger.error(f"[Producer]  Error publishing to {queue}: {e}")
            return False
    
    def send_epoch_update(self, 
                         job_id: str, 
                         epoch: int, 
                         total_epochs: int,
                         metrics: Dict[str, Any],
                         progress: float = None) -> bool:
        """
        Send epoch update to BE
        
        Args:
            job_id: Training job ID
            epoch: Current epoch number
            total_epochs: Total number of epochs
            metrics: Training metrics (loss, accuracy, etc.)
            progress: Progress percentage (0-100)
            
        Returns:
            True if successful, False otherwise
        """
        if progress is None:
            progress = (epoch / total_epochs) * 100.0
        
        message = {
            "type": "epoch_update",
            "job_id": job_id,
            "status": "running",
            "current_epoch": epoch,
            "total_epochs": total_epochs,
            "progress": progress,
            "metrics": metrics
        }
        
        logger.info(f"[Producer] Sending epoch update: job_id={job_id}, epoch={epoch}/{total_epochs}")
        return self._publish_message(self.update_queue, message)
    
    def send_training_started(self, job_id: str, total_epochs: int, config: Dict[str, Any] = None) -> bool:
        """
        Send training started notification
        
        Args:
            job_id: Training job ID
            total_epochs: Total number of epochs
            config: Training configuration
            
        Returns:
            True if successful, False otherwise
        """
        message = {
            "type": "training_started",
            "job_id": job_id,
            "status": "running",
            "current_epoch": 0,
            "total_epochs": total_epochs,
            "progress": 0.0,
            "config": config or {}
        }
        
        logger.info(f"[Producer] Sending training started: job_id={job_id}")
        return self._publish_message(self.update_queue, message)
    
    def send_training_completed(self, 
                               job_id: str, 
                               model_path: str,
                               final_metrics: Dict[str, Any],
                               training_time: float = None) -> bool:
        """
        Send training completion notification
        
        Args:
            job_id: Training job ID
            model_path: Path to trained model
            final_metrics: Final training metrics
            training_time: Total training time in seconds
            
        Returns:
            True if successful, False otherwise
        """
        message = {
            "type": "training_completed",
            "job_id": job_id,
            "status": "completed",
            "progress": 100.0,
            "model_path": model_path,
            "final_metrics": final_metrics,
            "training_time": training_time
        }
        
        logger.info(f"[Producer] Sending training completed: job_id={job_id}, model={model_path}")
        return self._publish_message(self.result_queue, message)
    
    def send_training_failed(self, 
                            job_id: str, 
                            error_message: str,
                            current_epoch: int = 0,
                            partial_metrics: Dict[str, Any] = None) -> bool:
        """
        Send training failure notification
        
        Args:
            job_id: Training job ID
            error_message: Error description
            current_epoch: Epoch where failure occurred
            partial_metrics: Metrics up to failure point
            
        Returns:
            True if successful, False otherwise
        """
        message = {
            "type": "training_failed",
            "job_id": job_id,
            "status": "failed",
            "current_epoch": current_epoch,
            "error_message": error_message,
            "partial_metrics": partial_metrics or {}
        }
        
        logger.error(f"[Producer] Sending training failed: job_id={job_id}, error={error_message}")
        return self._publish_message(self.result_queue, message)
    
    def send_training_cancelled(self, job_id: str, current_epoch: int = 0) -> bool:
        """
        Send training cancellation notification
        
        Args:
            job_id: Training job ID
            current_epoch: Epoch where cancellation occurred
            
        Returns:
            True if successful, False otherwise
        """
        message = {
            "type": "training_cancelled",
            "job_id": job_id,
            "status": "cancelled",
            "current_epoch": current_epoch
        }
        
        logger.info(f"[Producer] Sending training cancelled: job_id={job_id}")
        return self._publish_message(self.result_queue, message)
    
    def send_custom_update(self, job_id: str, update_type: str, data: Dict[str, Any]) -> bool:
        """
        Send custom update message
        
        Args:
            job_id: Training job ID
            update_type: Type of update
            data: Custom data to send
            
        Returns:
            True if successful, False otherwise
        """
        message = {
            "type": update_type,
            "job_id": job_id,
            **data
        }
        
        logger.info(f"[Producer] Sending custom update: job_id={job_id}, type={update_type}")
        return self._publish_message(self.update_queue, message)


class TrainingUpdatePublisher:
    """High-level interface for publishing training updates"""
    
    def __init__(self, producer: RabbitMQProducer):
        """
        Initialize publisher with producer instance
        
        Args:
            producer: RabbitMQ producer instance
        """
        self.producer = producer
        self.active_jobs = {}  # Track active jobs
    
    def start_job(self, job_id: str, total_epochs: int, config: Dict[str, Any] = None):
        """Start tracking a training job"""
        self.active_jobs[job_id] = {
            "total_epochs": total_epochs,
            "current_epoch": 0,
            "start_time": datetime.utcnow(),
            "config": config or {}
        }
        
        # Send started notification
        self.producer.send_training_started(job_id, total_epochs, config)
    
    def update_epoch(self, job_id: str, epoch: int, metrics: Dict[str, Any]):
        """Update epoch progress"""
        if job_id not in self.active_jobs:
            logger.warning(f"Job {job_id} not tracked, cannot send epoch update")
            return
        
        job_info = self.active_jobs[job_id]
        job_info["current_epoch"] = epoch
        
        # Send epoch update
        self.producer.send_epoch_update(
            job_id=job_id,
            epoch=epoch,
            total_epochs=job_info["total_epochs"],
            metrics=metrics
        )
    
    def complete_job(self, job_id: str, model_path: str, final_metrics: Dict[str, Any]):
        """Complete a training job"""
        if job_id not in self.active_jobs:
            logger.warning(f"Job {job_id} not tracked, cannot send completion")
            return
        
        job_info = self.active_jobs[job_id]
        training_time = (datetime.utcnow() - job_info["start_time"]).total_seconds()
        
        # Send completion notification
        self.producer.send_training_completed(
            job_id=job_id,
            model_path=model_path,
            final_metrics=final_metrics,
            training_time=training_time
        )
        
        # Remove from active jobs
        del self.active_jobs[job_id]
    
    def fail_job(self, job_id: str, error_message: str, partial_metrics: Dict[str, Any] = None):
        """Fail a training job"""
        current_epoch = 0
        if job_id in self.active_jobs:
            current_epoch = self.active_jobs[job_id]["current_epoch"]
            del self.active_jobs[job_id]
        
        # Send failure notification
        self.producer.send_training_failed(
            job_id=job_id,
            error_message=error_message,
            current_epoch=current_epoch,
            partial_metrics=partial_metrics
        )
    
    def cancel_job(self, job_id: str):
        """Cancel a training job"""
        current_epoch = 0
        if job_id in self.active_jobs:
            current_epoch = self.active_jobs[job_id]["current_epoch"]
            del self.active_jobs[job_id]
        
        # Send cancellation notification
        self.producer.send_training_cancelled(job_id, current_epoch)


# Global producer instance
rabbitmq_producer: Optional[RabbitMQProducer] = None
training_publisher: Optional[TrainingUpdatePublisher] = None

def initialize_rabbitmq_producer() -> RabbitMQProducer:
    """
    Initialize global RabbitMQ producer
    
    Returns:
        RabbitMQ producer instance
    """
    global rabbitmq_producer, training_publisher
    
    # Create producer
    rabbitmq_producer = RabbitMQProducer(
        result_queue=os.getenv("TRAIN_RESULT_QUEUE", "train_result_q"),
        update_queue=os.getenv("TRAIN_UPDATE_QUEUE", "train_update_q")
    )
    
    # Create high-level publisher
    training_publisher = TrainingUpdatePublisher(rabbitmq_producer)
    
    return rabbitmq_producer

def get_training_publisher() -> Optional[TrainingUpdatePublisher]:
    """Get global training publisher instance"""
    return training_publisher

def cleanup_rabbitmq_producer():
    """Cleanup producer connection"""
    global rabbitmq_producer
    
    if rabbitmq_producer:
        rabbitmq_producer.disconnect()
        rabbitmq_producer = None
