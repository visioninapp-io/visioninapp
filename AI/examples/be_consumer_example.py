"""
Example Backend Consumer for Training Updates
This shows how the BE should consume training updates from AI service
"""

import json
import logging
import pika
import os
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingUpdateConsumer:
    """Example consumer for BE to receive training updates from AI"""
    
    def __init__(self):
        # RabbitMQ connection parameters
        self.rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
        self.rabbitmq_port = int(os.getenv("RABBITMQ_PORT", "5672"))
        self.rabbitmq_user = os.getenv("RABBITMQ_USER", "guest")
        self.rabbitmq_password = os.getenv("RABBITMQ_PASSWORD", "guest")
        self.rabbitmq_vhost = os.getenv("RABBITMQ_VHOST", "/")
        
        # Queue names
        self.result_queue = os.getenv("TRAIN_RESULT_QUEUE", "train_result_q")
        self.update_queue = os.getenv("TRAIN_UPDATE_QUEUE", "train_update_q")
        
        self.connection = None
        self.channel = None
    
    def connect(self):
        """Connect to RabbitMQ"""
        try:
            credentials = pika.PlainCredentials(self.rabbitmq_user, self.rabbitmq_password)
            parameters = pika.ConnectionParameters(
                host=self.rabbitmq_host,
                port=self.rabbitmq_port,
                virtual_host=self.rabbitmq_vhost,
                credentials=credentials,
                heartbeat=30,
                blocked_connection_timeout=30
            )
            
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Declare queues
            self.channel.queue_declare(queue=self.result_queue, durable=True)
            self.channel.queue_declare(queue=self.update_queue, durable=True)
            
            # Set QoS
            self.channel.basic_qos(prefetch_count=1)
            
            logger.info(f"✅ Connected to RabbitMQ: {self.rabbitmq_host}:{self.rabbitmq_port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to RabbitMQ: {e}")
            return False
    
    def process_training_update(self, ch, method, properties, body):
        """Process training update message"""
        try:
            message = json.loads(body.decode('utf-8'))
            job_id = message.get('job_id')
            update_type = message.get('type')
            
            logger.info(f"[BE Consumer] Received {update_type} for job_id: {job_id}")
            
            if update_type == "epoch_update":
                self.handle_epoch_update(message)
            elif update_type == "training_started":
                self.handle_training_started(message)
            else:
                logger.warning(f"[BE Consumer] Unknown update type: {update_type}")
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except Exception as e:
            logger.error(f"[BE Consumer] Error processing update: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    
    def process_training_result(self, ch, method, properties, body):
        """Process training result message"""
        try:
            message = json.loads(body.decode('utf-8'))
            job_id = message.get('job_id')
            result_type = message.get('type')
            
            logger.info(f"[BE Consumer] Received {result_type} for job_id: {job_id}")
            
            if result_type == "training_completed":
                self.handle_training_completed(message)
            elif result_type == "training_failed":
                self.handle_training_failed(message)
            elif result_type == "training_cancelled":
                self.handle_training_cancelled(message)
            else:
                logger.warning(f"[BE Consumer] Unknown result type: {result_type}")
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except Exception as e:
            logger.error(f"[BE Consumer] Error processing result: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    
    def handle_epoch_update(self, message: Dict[str, Any]):
        """Handle epoch update from AI service"""
        job_id = message['job_id']
        epoch = message['current_epoch']
        total_epochs = message['total_epochs']
        progress = message['progress']
        metrics = message['metrics']
        
        logger.info(f"[BE] Job {job_id}: Epoch {epoch}/{total_epochs} ({progress:.1f}%)")
        logger.info(f"[BE] Metrics: Loss={metrics.get('train_loss', 0):.4f}, mAP50={metrics.get('mAP50', 0):.3f}")
        
        # TODO: Update database with new progress and metrics
        # Example:
        # db_job = db.query(TrainingJob).filter(TrainingJob.external_job_id == job_id).first()
        # if db_job:
        #     db_job.current_epoch = epoch
        #     db_job.progress_percentage = progress
        #     db_job.current_loss = metrics.get('train_loss', 0)
        #     db_job.current_accuracy = metrics.get('mAP50', 0)
        #     db_job.metrics_history[str(epoch)] = metrics
        #     db.commit()
    
    def handle_training_started(self, message: Dict[str, Any]):
        """Handle training started notification"""
        job_id = message['job_id']
        total_epochs = message['total_epochs']
        config = message.get('config', {})
        
        logger.info(f"[BE] Job {job_id}: Training started with {total_epochs} epochs")
        logger.info(f"[BE] Config: {config}")
        
        # TODO: Update database status to RUNNING
        # db_job = db.query(TrainingJob).filter(TrainingJob.external_job_id == job_id).first()
        # if db_job:
        #     db_job.status = TrainingStatus.RUNNING
        #     db_job.started_at = datetime.utcnow()
        #     db.commit()
    
    def handle_training_completed(self, message: Dict[str, Any]):
        """Handle training completion"""
        job_id = message['job_id']
        model_path = message['model_path']
        final_metrics = message['final_metrics']
        training_time = message.get('training_time', 0)
        
        logger.info(f"[BE] Job {job_id}: Training completed successfully!")
        logger.info(f"[BE] Model saved to: {model_path}")
        logger.info(f"[BE] Training time: {training_time:.1f}s")
        logger.info(f"[BE] Final metrics: {final_metrics}")
        
        # TODO: Update database status to COMPLETED
        # db_job = db.query(TrainingJob).filter(TrainingJob.external_job_id == job_id).first()
        # if db_job:
        #     db_job.status = TrainingStatus.COMPLETED
        #     db_job.completed_at = datetime.utcnow()
        #     db_job.progress_percentage = 100.0
        #     
        #     # Create model record
        #     model = Model(
        #         name=f"Model from job {job_id}",
        #         framework="YOLO",
        #         file_path=model_path,
        #         training_job_id=db_job.id,
        #         metrics=final_metrics
        #     )
        #     db.add(model)
        #     db.commit()
    
    def handle_training_failed(self, message: Dict[str, Any]):
        """Handle training failure"""
        job_id = message['job_id']
        error_message = message['error_message']
        current_epoch = message.get('current_epoch', 0)
        partial_metrics = message.get('partial_metrics', {})
        
        logger.error(f"[BE] Job {job_id}: Training failed at epoch {current_epoch}")
        logger.error(f"[BE] Error: {error_message}")
        
        # TODO: Update database status to FAILED
        # db_job = db.query(TrainingJob).filter(TrainingJob.external_job_id == job_id).first()
        # if db_job:
        #     db_job.status = TrainingStatus.FAILED
        #     db_job.error_message = error_message
        #     db_job.completed_at = datetime.utcnow()
        #     db.commit()
    
    def handle_training_cancelled(self, message: Dict[str, Any]):
        """Handle training cancellation"""
        job_id = message['job_id']
        current_epoch = message.get('current_epoch', 0)
        
        logger.info(f"[BE] Job {job_id}: Training cancelled at epoch {current_epoch}")
        
        # TODO: Update database status to CANCELLED
        # db_job = db.query(TrainingJob).filter(TrainingJob.external_job_id == job_id).first()
        # if db_job:
        #     db_job.status = TrainingStatus.CANCELLED
        #     db_job.completed_at = datetime.utcnow()
        #     db.commit()
    
    def start_consuming(self):
        """Start consuming messages"""
        if not self.connect():
            logger.error("Failed to connect to RabbitMQ")
            return
        
        try:
            # Set up consumers
            self.channel.basic_consume(
                queue=self.update_queue,
                on_message_callback=self.process_training_update
            )
            
            self.channel.basic_consume(
                queue=self.result_queue,
                on_message_callback=self.process_training_result
            )
            
            logger.info(f"🚀 Started consuming from queues: {self.update_queue}, {self.result_queue}")
            logger.info("Waiting for messages. To exit press CTRL+C")
            
            # Start consuming
            self.channel.start_consuming()
            
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
            self.channel.stop_consuming()
            self.connection.close()

def main():
    """Main function to run the consumer"""
    consumer = TrainingUpdateConsumer()
    consumer.start_consuming()

if __name__ == "__main__":
    main()
