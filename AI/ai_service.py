"""
AI Service - Startup Orchestrator for Pure RabbitMQ Architecture
Simplified version that just starts the multi-service consumer
"""

import logging
import signal
import sys
import time
import torch
from config import settings
from multi_service_consumer import initialize_multi_service_consumer, start_multi_service_consumer, stop_multi_service_consumer

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIService:
    """AI Service Startup Orchestrator - Pure RabbitMQ Architecture"""
    
    def __init__(self):
        """Initialize AI service orchestrator"""
        self.running = False
        self.consumer = None
        
        logger.info("=" * 60)
        logger.info(" AI Multi-Service (Training, Conversion, Inference via RabbitMQ)")
        logger.info("=" * 60)
        
        # Print system info
        self._print_system_info()
        
        # Print configuration
        settings.print_config()
    
    def _print_system_info(self):
        """Print system information"""
        logger.info("System Information:")
        
        # CUDA info
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1e9
            
            logger.info(f"   CUDA: Available ({torch.version.cuda})")
            logger.info(f"   GPU: {device_name}")
            logger.info(f"   GPU Memory: {memory_total:.1f} GB")
            logger.info(f"   Devices: {device_count}")
        else:
            logger.info("   CUDA: Not available")
        
        # PyTorch info
        logger.info(f"   PyTorch: {torch.__version__}")
        
        # Service info
        logger.info(f"   AI Multi-Service (Training, Conversion, Inference via RabbitMQ)")
        logger.info("   Ready to process RabbitMQ messages")
        logger.info("=" * 60)
        logger.info("")
    
    def initialize_rabbitmq(self):
        """Initialize RabbitMQ multi-service consumer"""
        try:
            logger.info(" Initializing RabbitMQ multi-service consumer...")
            
            # Initialize multi-service consumer (handles producer internally)
            self.consumer = initialize_multi_service_consumer()
            if not self.consumer:
                raise Exception("Failed to initialize multi-service consumer")
            
            logger.info(" Multi-service RabbitMQ consumer initialized")
            return True
            
        except Exception as e:
            logger.error(f" Failed to initialize RabbitMQ services: {e}")
            return False
    
    def start(self):
        """Start the AI service"""
        logger.info(" Starting AI Service...")
        
        # Initialize RabbitMQ
        if not self.initialize_rabbitmq():
            logger.error(" Failed to initialize RabbitMQ - exiting")
            return False
        
        # Start multi-service RabbitMQ consumer
        try:
            start_multi_service_consumer()
            logger.info(" Multi-service RabbitMQ consumer started")
            self.running = True
            
            logger.info("=" * 60)
            logger.info(" AI Service Ready - Waiting for training/conversion/inference requests...")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f" Failed to start RabbitMQ consumer: {e}")
            return False
    
    def stop(self):
        """Stop the AI service"""
        logger.info("🛑 Stopping AI Service...")
        self.running = False
        
        try:
            # Stop RabbitMQ services
            stop_multi_service_consumer()
            logger.info(" RabbitMQ services stopped")
            
            logger.info(" AI Service stopped successfully")
            
        except Exception as e:
            logger.error(f" Error stopping AI service: {e}")
    
    def run_forever(self):
        """Run the service until interrupted"""
        if not self.start():
            return
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Keep the main thread alive
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()


def main():
    """Main entry point"""
    service = AIService()
    service.run_forever()


if __name__ == "__main__":
    main()
