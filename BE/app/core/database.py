from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
import os

# MySQL 연결 설정
mysql_user = settings.MYSQL_USER
mysql_password = settings.MYSQL_PASSWORD
mysql_host = settings.MYSQL_HOST
mysql_port = str(settings.MYSQL_PORT)
mysql_database = settings.MYSQL_DATABASE

# DATABASE_URL이 직접 설정되어 있으면 사용, 없으면 MySQL 설정으로 구성
if settings.DATABASE_URL and "mysql" in settings.DATABASE_URL.lower():
    DATABASE_URL = settings.DATABASE_URL
else:
    # KST 타임존 설정 추가
    DATABASE_URL = f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_database}?charset=utf8mb4&init_command=SET time_zone='%2B09:00'"

# MySQL 엔진 설정 (커넥션 풀링 및 최적화)
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # 연결 유효성 검사
    pool_size=20,  # 기본 커넥션 풀 크기
    max_overflow=30,  # 최대 오버플로우 커넥션
    pool_recycle=3600,  # 1시간 후 커넥션 재사용
    echo=False  # SQL 로깅 (디버깅 시 True)
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
