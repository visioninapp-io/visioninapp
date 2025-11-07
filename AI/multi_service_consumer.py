"""
Multi-Service RabbitMQ Consumer for AI Service
Handles training, conversion, and inference requests via RabbitMQ messages only
Direct service instantiation - no external handlers needed
"""

import json
import logging
import threading
import time
from typing import Dict, Any, Optional
from datetime import datetime
import pika
from pika.exceptions import AMQPConnectionError
from config import settings

# Import AI services directly
from training_service import YOLOTrainingService
from conversion_service import ModelConversionService
from inference_service import YOLOInferenceService
from rabbitmq_producer import initialize_rabbitmq_producer, get_training_publisher

# Setup logging
logger = logging.getLogger(__name__)

class MultiServiceConsumer:
    """RabbitMQ consumer for multiple AI services - direct service calls"""
    
    def __init__(self):
        """Initialize multi-service consumer"""
        self.connection = None
        self.channel = None
        self.consuming = False
        self.consumer_thread = None
        self.should_stop = threading.Event()
        
        # Initialize services directly
        self.conversion_service = ModelConversionService()
        self.inference_service = YOLOInferenceService()
        # training_service is created per job
        
        # Initialize RabbitMQ producer for sending updates
        self.producer = None
        self.publisher = None
        
        # Job tracking
        self.active_jobs = {}
        
        logger.info("Multi-Service RabbitMQ Consumer initialized")
        logger.info(" Services ready: Training, Conversion, Inference")
    
    def initialize_producer(self):
        """Initialize RabbitMQ producer for sending updates"""
        try:
            self.producer = initialize_rabbitmq_producer()
            self.publisher = get_training_publisher()
            logger.info(" RabbitMQ producer initialized")
            return True
        except Exception as e:
            logger.error(f" Failed to initialize RabbitMQ producer: {e}")
            return False
    
    def connect(self):
        """Establish connection to RabbitMQ"""
        try:
            credentials = pika.PlainCredentials(settings.RABBITMQ_USER, settings.RABBITMQ_PASSWORD)
            parameters = pika.ConnectionParameters(
                host=settings.RABBITMQ_HOST,
                port=settings.RABBITMQ_PORT,
                virtual_host=settings.RABBITMQ_VHOST,
                credentials=credentials,
                heartbeat=30,
                blocked_connection_timeout=30,
                connection_attempts=3,
                retry_delay=1,
                socket_timeout=5
            )
            
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Declare all queues
            self._declare_queues()
            
            # Set QoS
            self.channel.basic_qos(prefetch_count=1)
            
            logger.info(f" Connected to RabbitMQ: {settings.RABBITMQ_HOST}:{settings.RABBITMQ_PORT}")
            return True
            
        except Exception as e:
            logger.error(f" Failed to connect to RabbitMQ: {e}")
            return False
    
    def _declare_queues(self):
        """Declare all required queues"""
        queues = [
            settings.TRAIN_REQUEST_QUEUE,
            settings.CONVERT_REQUEST_QUEUE,
            settings.INFERENCE_REQUEST_QUEUE
        ]
        
        for queue in queues:
            self.channel.queue_declare(queue=queue, durable=True)
            logger.info(f" Declared queue: {queue}")
    
    def _setup_consumers(self):
        """Set up consumers for all queues"""
        # Training requests
        self.channel.basic_consume(
            queue=settings.TRAIN_REQUEST_QUEUE,
            on_message_callback=self._handle_training_request
        )
        logger.info(f"👂 Listening for training requests on: {settings.TRAIN_REQUEST_QUEUE}")
        
        # Conversion requests
        self.channel.basic_consume(
            queue=settings.CONVERT_REQUEST_QUEUE,
            on_message_callback=self._handle_conversion_request
        )
        logger.info(f"👂 Listening for conversion requests on: {settings.CONVERT_REQUEST_QUEUE}")
        
        # Inference requests
        self.channel.basic_consume(
            queue=settings.INFERENCE_REQUEST_QUEUE,
            on_message_callback=self._handle_inference_request
        )
        logger.info(f"👂 Listening for inference requests on: {settings.INFERENCE_REQUEST_QUEUE}")
    
    def _handle_training_request(self, ch, method, properties, body):
        """Handle training request message - direct service call"""
        try:
            message = json.loads(body.decode('utf-8'))
            job_id = message.get('job_id')
            
            logger.info(f"[Training] Received request for job_id: {job_id}")
            
            if not job_id:
                logger.error("[Training] Message missing 'job_id'. Rejecting.")
                ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
                return
            
            # Start training job in background thread
            training_thread = threading.Thread(
                target=self._run_training_job,
                args=(job_id, message),
                daemon=True
            )
            training_thread.start()
            logger.info(f"[Training] Started training job {job_id}")
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except json.JSONDecodeError:
            logger.error(f"[Training] Invalid JSON: {body.decode()}. Rejecting.")
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            logger.error(f"[Training] Error processing message: {e}. Rejecting.")
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
    
    def _handle_conversion_request(self, ch, method, properties, body):
        """Handle conversion request message - direct service call"""
        try:
            message = json.loads(body.decode('utf-8'))
            job_id = message.get('job_id')
            
            logger.info(f"[Conversion] Received request for job_id: {job_id}")
            
            if not job_id:
                logger.error("[Conversion] Message missing 'job_id'. Rejecting.")
                ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
                return
            
            # Start conversion job in background thread
            conversion_thread = threading.Thread(
                target=self._run_conversion_job,
                args=(job_id, message),
                daemon=True
            )
            conversion_thread.start()
            logger.info(f"[Conversion] Started conversion job {job_id}")
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except json.JSONDecodeError:
            logger.error(f"[Conversion] Invalid JSON: {body.decode()}. Rejecting.")
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            logger.error(f"[Conversion] Error processing message: {e}. Rejecting.")
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
    
    def _handle_inference_request(self, ch, method, properties, body):
        """Handle inference request message - direct service call"""
        try:
            message = json.loads(body.decode('utf-8'))
            job_id = message.get('job_id')
            
            logger.info(f"[Inference] Received request for job_id: {job_id}")
            
            if not job_id:
                logger.error("[Inference] Message missing 'job_id'. Rejecting.")
                ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
                return
            
            # Start inference job in background thread
            inference_thread = threading.Thread(
                target=self._run_inference_job,
                args=(job_id, message),
                daemon=True
            )
            inference_thread.start()
            logger.info(f"[Inference] Started inference job {job_id}")
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except json.JSONDecodeError:
            logger.error(f"[Inference] Invalid JSON: {body.decode()}. Rejecting.")
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            logger.error(f"[Inference] Error processing message: {e}. Rejecting.")
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
    
    def _run_training_job(self, job_id: str, request_data: Dict[str, Any]):
        """Run training job directly"""
        import traceback
        
        try:
            logger.info(f"[Training {job_id}] Starting training job")
            
            # Track job
            self.active_jobs[job_id] = {
                "type": "training",
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
                "progress": 0.0
            }
            
            # Send training started notification
            if self.publisher:
                self.publisher.start_job(job_id, request_data)
            
            # Create training service for this job
            training_service = YOLOTrainingService(
                job_id=job_id,
                progress_callback=self._training_progress_callback,
                completion_callback=self._training_completion_callback
            )
            
            # Run training
            result = training_service.train(request_data)
            
            logger.info(f"[Training {job_id}]  Training completed successfully")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[Training {job_id}]  Training failed: {error_msg}")
            
            self.active_jobs[job_id] = {
                "type": "training",
                "status": "failed",
                "error": error_msg,
                "completed_at": datetime.utcnow().isoformat()
            }
            
            # Send failure notification
            if self.publisher:
                self.publisher.fail_job(job_id, error_msg, {})
            
            logger.error(f"[Training {job_id}] Traceback: {traceback.format_exc()}")
    
    def _run_conversion_job(self, job_id: str, request_data: Dict[str, Any]):
        """Run conversion job directly"""
        import traceback
        
        try:
            logger.info(f"[Conversion {job_id}] Starting conversion job")
            
            # Parse request data
            model_path = request_data["model_path"]
            target_format = request_data["target_format"]
            conversion_options = request_data.get("options", {})
            
            # Track job
            self.active_jobs[job_id] = {
                "type": "conversion",
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
                "model_path": model_path,
                "target_format": target_format,
                "progress": 0.0
            }
            
            # Send conversion started notification
            if self.publisher:
                self.publisher.producer.send_custom_update(
                    job_id, "conversion_started", {
                        "model_path": model_path,
                        "target_format": target_format,
                        "options": conversion_options
                    }
                )
            
            # Run conversion
            logger.info(f"[Conversion {job_id}] Converting {model_path} to {target_format}")
            
            result = self.conversion_service.convert_model(
                model_path=model_path,
                target_format=target_format,
                **conversion_options
            )
            
            # Update final status
            if result.get('success', False):
                self.active_jobs[job_id].update({
                    "status": "completed",
                    "progress": 100.0,
                    "completed_at": datetime.utcnow().isoformat(),
                    "output_path": result.get("output_path"),
                    "result": result
                })
                
                # Send completion notification
                if self.publisher:
                    self.publisher.producer.send_custom_update(
                        job_id, "conversion_completed", {
                            "output_path": result.get("output_path"),
                            "conversion_time": result.get("conversion_time"),
                            "model_info": result.get("model_info", {})
                        }
                    )
                
                logger.info(f"[Conversion {job_id}]  Conversion completed successfully")
                logger.info(f"[Conversion {job_id}] Output: {result.get('output_path')}")
                
            else:
                raise Exception(result.get('error', 'Unknown conversion error'))
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[Conversion {job_id}]  Conversion failed: {error_msg}")
            
            self.active_jobs[job_id].update({
                "status": "failed",
                "error": error_msg,
                "completed_at": datetime.utcnow().isoformat(),
                "progress": 0.0
            })
            
            # Send failure notification
            if self.publisher:
                self.publisher.producer.send_custom_update(
                    job_id, "conversion_failed", {
                        "error_message": error_msg
                    }
                )
            
            logger.error(f"[Conversion {job_id}] Traceback: {traceback.format_exc()}")
    
    def _run_inference_job(self, job_id: str, request_data: Dict[str, Any]):
        """Run inference job directly"""
        import traceback
        
        try:
            logger.info(f"[Inference {job_id}] Starting inference job")
            
            # Parse request data
            image_path = request_data["image_path"]
            model_path = request_data.get("model_path")
            inference_options = request_data.get("options", {})
            
            # Track job
            self.active_jobs[job_id] = {
                "type": "inference",
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
                "image_path": image_path,
                "model_path": model_path,
                "progress": 0.0
            }
            
            # Send inference started notification
            if self.publisher:
                self.publisher.producer.send_custom_update(
                    job_id, "inference_started", {
                        "image_path": image_path,
                        "model_path": model_path,
                        "options": inference_options
                    }
                )
            
            # Load model if specified
            if model_path and model_path != self.inference_service.model_path:
                logger.info(f"[Inference {job_id}] Loading model: {model_path}")
                self.inference_service.load_model(model_path)
            elif not self.inference_service.is_loaded:
                logger.info(f"[Inference {job_id}] Loading default model")
                self.inference_service.load_model()
            
            # Run inference
            logger.info(f"[Inference {job_id}] Running inference on: {image_path}")
            
            result = self.inference_service.annotate_for_backend(
                image_path=image_path,
                **inference_options
            )
            
            # Update final status
            if result.get('success', False):
                self.active_jobs[job_id].update({
                    "status": "completed",
                    "progress": 100.0,
                    "completed_at": datetime.utcnow().isoformat(),
                    "annotations": result.get("annotations", []),
                    "result": result
                })
                
                # Send completion notification
                if self.publisher:
                    self.publisher.producer.send_custom_update(
                        job_id, "inference_completed", {
                            "annotations": result.get("annotations", []),
                            "total_detections": result.get("total_detections", 0),
                            "model_info": result.get("model_info", {})
                        }
                    )
                
                logger.info(f"[Inference {job_id}]  Inference completed successfully")
                logger.info(f"[Inference {job_id}] Detections: {result.get('total_detections', 0)}")
                
            else:
                raise Exception(result.get('error', 'Unknown inference error'))
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[Inference {job_id}]  Inference failed: {error_msg}")
            
            self.active_jobs[job_id].update({
                "status": "failed",
                "error": error_msg,
                "completed_at": datetime.utcnow().isoformat(),
                "progress": 0.0
            })
            
            # Send failure notification
            if self.publisher:
                self.publisher.producer.send_custom_update(
                    job_id, "inference_failed", {
                        "error_message": error_msg
                    }
                )
            
            logger.error(f"[Inference {job_id}] Traceback: {traceback.format_exc()}")
    
    def _training_progress_callback(self, job_id: str, epoch: int, total_epochs: int, metrics: Dict[str, Any]):
        """Callback for training progress updates"""
        progress = (epoch / total_epochs) * 100
        
        # Update job tracking
        if job_id in self.active_jobs:
            self.active_jobs[job_id].update({
                "current_epoch": epoch,
                "total_epochs": total_epochs,
                "progress": progress,
                "metrics": metrics
            })
        
        # Send progress update
        if self.publisher:
            self.publisher.update_progress(job_id, epoch, total_epochs, metrics)
    
    def _training_completion_callback(self, job_id: str, result: Dict[str, Any]):
        """Callback for training completion"""
        if result.get('success', False):
            # Update job status
            if job_id in self.active_jobs:
                self.active_jobs[job_id].update({
                    "status": "completed",
                    "progress": 100.0,
                    "completed_at": datetime.utcnow().isoformat(),
                    "result": result
                })
            
            # Send completion notification
            if self.publisher:
                self.publisher.complete_job(job_id, result)
        else:
            # Handle failure
            error_msg = result.get('error', 'Training failed')
            if job_id in self.active_jobs:
                self.active_jobs[job_id].update({
                    "status": "failed",
                    "error": error_msg,
                    "completed_at": datetime.utcnow().isoformat()
                })
            
            if self.publisher:
                self.publisher.fail_job(job_id, error_msg, result.get('metrics', {}))
    
    def start_consuming(self):
        """Start consuming messages"""
        def _consume_loop():
            while not self.should_stop.is_set():
                try:
                    if not self.connect():
                        logger.error("Failed to connect to RabbitMQ. Retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    
                    self._setup_consumers()
                    self.consuming = True
                    
                    logger.info(" Started consuming messages from all queues")
                    
                    # Start consuming
                    self.channel.start_consuming()
                    
                except AMQPConnectionError as e:
                    logger.error(f" RabbitMQ connection lost: {e}. Reconnecting...")
                    time.sleep(5)
                except Exception as e:
                    logger.error(f" Unexpected error in consumer: {e}")
                    time.sleep(5)
                finally:
                    self.consuming = False
                    if self.connection and not self.connection.is_closed:
                        try:
                            self.connection.close()
                        except:
                            pass
        
        # Start consumer in background thread
        self.consumer_thread = threading.Thread(target=_consume_loop, daemon=True)
        self.consumer_thread.start()
        
        logger.info(" Multi-service consumer started")
    
    def stop_consuming(self):
        """Stop consuming messages"""
        logger.info("🛑 Stopping multi-service consumer...")
        
        self.should_stop.set()
        
        if self.consuming and self.channel and self.channel.is_open:
            try:
                self.channel.stop_consuming()
            except:
                pass
        
        if self.connection and not self.connection.is_closed:
            try:
                self.connection.close()
            except:
                pass
        
        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=5)
        
        logger.info(" Multi-service consumer stopped")


# Global consumer instance
_consumer: Optional[MultiServiceConsumer] = None

def initialize_multi_service_consumer():
    """Initialize the global multi-service consumer"""
    global _consumer
    
    _consumer = MultiServiceConsumer()
    
    # Initialize producer
    if not _consumer.initialize_producer():
        logger.error("Failed to initialize RabbitMQ producer")
        return None
    
    return _consumer

def start_multi_service_consumer():
    """Start the global consumer"""
    if _consumer:
        _consumer.start_consuming()
    else:
        logger.error("Consumer not initialized. Call initialize_multi_service_consumer() first.")

def stop_multi_service_consumer():
    """Stop the global consumer"""
    if _consumer:
        _consumer.stop_consuming()
    else:
        logger.warning("Consumer not initialized.")
