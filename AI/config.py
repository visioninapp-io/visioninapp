"""
Configuration for AI Service
Environment variables and settings
"""

import os
from pathlib import Path
from typing import Optional

class Settings:
    """Application settings"""
    
    # Service settings
    HOST: str = os.getenv("AI_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("AI_PORT", "8001"))
    
    # GPU settings
    DEVICE: Optional[str] = os.getenv("CUDA_DEVICE", None)  # Auto-detect if None
    
    # Model paths - resolve relative to AI directory, not current working directory
    _AI_DIR = Path(__file__).parent.absolute()
    MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", str(_AI_DIR / "models")))
    UPLOADS_DIR: Path = Path(os.getenv("UPLOADS_DIR", str(_AI_DIR / "uploads")))
    
    # RabbitMQ settings
    RABBITMQ_HOST: str = os.getenv("RABBITMQ_HOST", "localhost")
    RABBITMQ_PORT: int = int(os.getenv("RABBITMQ_PORT", "5672"))
    RABBITMQ_USER: str = os.getenv("RABBITMQ_USER", "guest")
    RABBITMQ_PASSWORD: str = os.getenv("RABBITMQ_PASSWORD", "guest")
    RABBITMQ_VHOST: str = os.getenv("RABBITMQ_VHOST", "/")
    RABBITMQ_SSL: bool = os.getenv("RABBITMQ_SSL", "false").lower() == "true"
    
    # Queue names
    TRAIN_REQUEST_QUEUE: str = os.getenv("TRAIN_REQUEST_QUEUE", "train_request_q")
    TRAIN_RESULT_QUEUE: str = os.getenv("TRAIN_RESULT_QUEUE", "train_result_q")
    TRAIN_UPDATE_QUEUE: str = os.getenv("TRAIN_UPDATE_QUEUE", "train_update_q")
    
    # Conversion queues
    CONVERT_REQUEST_QUEUE: str = os.getenv("CONVERT_REQUEST_QUEUE", "convert_request_q")
    CONVERT_RESULT_QUEUE: str = os.getenv("CONVERT_RESULT_QUEUE", "convert_result_q")
    
    # Inference queues
    INFERENCE_REQUEST_QUEUE: str = os.getenv("INFERENCE_REQUEST_QUEUE", "inference_request_q")
    INFERENCE_RESULT_QUEUE: str = os.getenv("INFERENCE_RESULT_QUEUE", "inference_result_q")
    
    # Training settings
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
    DEFAULT_EPOCHS: int = int(os.getenv("DEFAULT_EPOCHS", "100"))
    DEFAULT_BATCH_SIZE: int = int(os.getenv("DEFAULT_BATCH_SIZE", "16"))
    DEFAULT_IMAGE_SIZE: int = int(os.getenv("DEFAULT_IMAGE_SIZE", "640"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Feature flags
    ENABLE_RABBITMQ: bool = os.getenv("ENABLE_RABBITMQ", "true").lower() == "true"
    ENABLE_TENSORRT: bool = os.getenv("ENABLE_TENSORRT", "true").lower() == "true"
    
    def __init__(self):
        """Initialize settings and create directories"""
        # Create directories
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.UPLOADS_DIR.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.UPLOADS_DIR / "models").mkdir(exist_ok=True)
        (self.UPLOADS_DIR / "datasets").mkdir(exist_ok=True)
    
    @property
    def rabbitmq_url(self) -> str:
        """Get RabbitMQ connection URL"""
        protocol = "amqps" if self.RABBITMQ_SSL else "amqp"
        return f"{protocol}://{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}{self.RABBITMQ_VHOST}"
    
    def is_rabbitmq_configured(self) -> bool:
        """Check if RabbitMQ is properly configured"""
        return (
            self.ENABLE_RABBITMQ and
            self.RABBITMQ_HOST != "localhost" and  # Assume localhost means not configured
            self.RABBITMQ_USER != "guest" and
            self.RABBITMQ_PASSWORD != "guest"
        )
    
    def print_config(self):
        """Print current configuration (without sensitive data)"""
        print("=" * 60)
        print("AI Service Configuration")
        print("=" * 60)
        print(f"Host: {self.HOST}:{self.PORT}")
        print(f"Models Directory: {self.MODELS_DIR}")
        print(f"Uploads Directory: {self.UPLOADS_DIR}")
        print(f"CUDA Device: {self.DEVICE or 'Auto-detect'}")
        print(f"Log Level: {self.LOG_LEVEL}")
        print(f"Max Concurrent Jobs: {self.MAX_CONCURRENT_JOBS}")
        print("")
        print("RabbitMQ Configuration:")
        print(f"  Enabled: {self.ENABLE_RABBITMQ}")
        if self.ENABLE_RABBITMQ:
            print(f"  Host: {self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}")
            print(f"  VHost: {self.RABBITMQ_VHOST}")
            print(f"  SSL: {self.RABBITMQ_SSL}")
            print(f"  Training Request Queue: {self.TRAIN_REQUEST_QUEUE}")
            print(f"  Training Result Queue: {self.TRAIN_RESULT_QUEUE}")
            print(f"  Training Update Queue: {self.TRAIN_UPDATE_QUEUE}")
            print(f"  Conversion Request Queue: {self.CONVERT_REQUEST_QUEUE}")
            print(f"  Conversion Result Queue: {self.CONVERT_RESULT_QUEUE}")
            print(f"  Inference Request Queue: {self.INFERENCE_REQUEST_QUEUE}")
            print(f"  Inference Result Queue: {self.INFERENCE_RESULT_QUEUE}")
            print(f"  Configured: {self.is_rabbitmq_configured()}")
        print("")
        print("Feature Flags:")
        print(f"  TensorRT: {self.ENABLE_TENSORRT}")
        print("=" * 60)

# Global settings instance
settings = Settings()
