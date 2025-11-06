from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Backend"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5500",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5500",
        "*"  # Allow all origins for development
    ]

    # Database - MySQL Configuration
    DATABASE_URL: str = ""  # 환경 변수로 설정하거나 자동으로 구성됨
    
    # MySQL Configuration (환경 변수로 설정)
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "password"
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_DATABASE: str = "vision_db"

    # Security
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # File Storage
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB

    # Redis Cache
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    CACHE_TTL: int = 300  # 5 minutes
    ENABLE_CACHE: bool = False  # Set to True when Redis is available
    
    # RabbitMQ
    RABBITMQ_HOST: str = ""
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = ""
    RABBITMQ_PASSWORD: str = ""
    RABBITMQ_VHOST: str = "/"
    RABBITMQ_SSL: bool = False

    TRAIN_REQUEST_QUEUE: str = "train_request_q"

    # AWS S3 Configuration
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "ap-northeast-2"
    AWS_BUCKET_NAME: str = ""
    USE_S3_STORAGE: bool = False  # Set to True to use S3 instead of local storage
    
    # Image Cache Settings
    IMAGE_CACHE_TTL: int = 86400  # 24 hours in seconds
    IMAGE_CACHE_PAGE_SIZE: int = 50  # Number of images per page

    AI_SERVICE_URL: str = "http://localhost:8001"   # 추후 gpu 서버로 변경


    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()