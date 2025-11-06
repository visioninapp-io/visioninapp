from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float, JSON, Text, Boolean, UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base


class GeometryType(str, enum.Enum):
    """어노테이션 형태 타입"""
    BBOX = "bbox"
    POLYGON = "polygon"
    KEYPOINT = "keypoint"


class Dataset(Base):
    """데이터셋 기본 정보 (ERD 기준)"""
    __tablename__ = "dataset"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("project.id"), nullable=False, index=True, comment="소속 프로젝트")
    name = Column(String(255), nullable=False, comment="데이터셋명")
    description = Column(Text, nullable=True, comment="설명")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="생성일")

    # Relationships
    project = relationship("Project", back_populates="datasets")
    training_jobs = relationship("TrainingJob", back_populates="dataset")
    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('project_id', 'name', name='uq_dataset_project_name'),
    )


class Annotation(Base):
    __tablename__ = "annotation"

    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey("asset.id"), nullable=False, index=True, comment="대상에셋ID")
    label_class_id = Column(Integer, ForeignKey("label_class.id"), nullable=False, index=True, comment="라벨 클래스ID")
    model_version_id = Column(Integer, ForeignKey("model_version.id"), nullable=True, index=True, comment="자동 생성 모델 ID")
    
    geometry_type = Column(Enum(GeometryType), nullable=False, comment="형태")
    geometry = Column(JSON, nullable=False, comment="좌표/포인트 데이터")
    is_normalized = Column(Boolean, nullable=False, default=True, comment="정규화여부")
    source = Column(Text, nullable=True, comment="생성 주석")  # 'human', 'model'
    confidence = Column(Float, nullable=False, default=1.0, comment="모델 신뢰도")
    annotator_name = Column(Text, nullable=False, default="system", comment="라벨러")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="생성")

    # Relationships
    asset = relationship("Asset", back_populates="annotations")
    label_class = relationship("LabelClass", back_populates="annotations")
    model_version = relationship("ModelVersion", back_populates="annotations")


class DatasetVersion(Base):
    """Dataset version with preprocessing and augmentation settings"""
    __tablename__ = "dataset_version"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("dataset.id"), nullable=False, index=True, comment="소속 데이터셋")
    ontology_version_id = Column(Integer, ForeignKey("label_ontology_version.id"), nullable=False, index=True, comment="온톨로지 버전ID")
    version_tag = Column(String(50), nullable=False, comment="버전 태그")
    is_frozen = Column(Boolean, nullable=False, default=False, comment="수정 불가 여부")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="생성일")

    # Relationships
    dataset = relationship("Dataset", back_populates="versions")
    ontology_version = relationship("LabelOntologyVersion", back_populates="dataset_versions")
    splits = relationship("DatasetSplit", back_populates="dataset_version", cascade="all, delete-orphan")
    model_versions = relationship("ModelVersion", back_populates="dataset_version")


class ExportJob(Base):
    """Track dataset export jobs"""
    __tablename__ = "export_job"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("dataset.id"), nullable=True)
    version_id = Column(Integer, ForeignKey("dataset_version.id"), nullable=True)

    export_format = Column(String(50))  # yolov8, yolov5, coco, pascal_voc, etc.
    include_images = Column(Integer, default=1)  # boolean as int

    # File info
    file_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)
    download_url = Column(String(500), nullable=True)

    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    created_by = Column(String(100))
