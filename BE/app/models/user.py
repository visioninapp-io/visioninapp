from sqlalchemy import Column, Integer, String, DateTime, Enum
from datetime import datetime
import enum
from app.core.database import Base


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    firebase_uid = Column(String, unique=True, index=True)

    email = Column(String, unique=True, index=True)
    display_name = Column(String, nullable=True)
    photo_url = Column(String, nullable=True)

    role = Column(Enum(UserRole), default=UserRole.USER)

    is_active = Column(Integer, default=1)  # boolean as int

    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
