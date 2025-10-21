from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"))

    name = Column(String)
    description = Column(Text, nullable=True)

    # Overall metrics
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    map_50 = Column(Float)  # mAP@0.5
    map_50_95 = Column(Float, nullable=True)  # mAP@0.5:0.95

    # Per-class metrics
    class_metrics = Column(JSON)  # [{class, precision, recall, f1, support}, ...]

    # Confusion matrix
    confusion_matrix = Column(JSON, nullable=True)

    # Test dataset info
    test_dataset_size = Column(Integer)
    test_dataset_name = Column(String, nullable=True)

    # Comparison with previous versions
    comparison_data = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, index=True)

    # Relationships
    model = relationship("Model", back_populates="evaluations")
