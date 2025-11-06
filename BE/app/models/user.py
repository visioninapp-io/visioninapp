from sqlalchemy import Column, Integer, String
from app.core.database import Base


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, index=True, comment="유저ID")
    name = Column(String(50), nullable=False, comment="이름")
    firebase_uid = Column(String(100), nullable=False, unique=True, index=True, comment="유저정보")
