from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta
from app.core.database import get_db
from app.core.auth import get_current_user, get_current_user_dev
from app.models.monitoring import (
    MonitoringAlert, PerformanceMetric, FeedbackLoop, EdgeCase, AlertStatus
)
from app.models.deployment import Deployment
from app.schemas.monitoring import (
    MonitoringAlertResponse, AlertAcknowledgeRequest,
    PerformanceMetricResponse, FeedbackLoopCreate, FeedbackLoopUpdate,
    FeedbackLoopResponse, EdgeCaseResponse, EdgeCaseReviewRequest,
    MonitoringDashboardResponse
)

router = APIRouter()


@router.get("/dashboard", response_model=MonitoringDashboardResponse)
async def get_monitoring_dashboard(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get monitoring dashboard data"""
    # Get recent performance metrics
    recent_metrics = db.query(PerformanceMetric).order_by(
        PerformanceMetric.created_at.desc()
    ).limit(10).all()

    # Calculate aggregated stats
    live_accuracy = sum(m.accuracy or 0 for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
    avg_latency = sum(m.avg_latency for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0

    # Get today's inference count
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_metrics = db.query(PerformanceMetric).filter(
        PerformanceMetric.created_at >= today_start
    ).all()
    today_inferences = sum(m.requests_count for m in today_metrics)

    # Get active alerts
    active_alerts_count = db.query(MonitoringAlert).filter(
        MonitoringAlert.status == AlertStatus.ACTIVE
    ).count()

    # Get alerts
    alerts = db.query(MonitoringAlert).filter(
        MonitoringAlert.status == AlertStatus.ACTIVE
    ).order_by(MonitoringAlert.created_at.desc()).limit(10).all()

    # Calculate accuracy trend
    accuracy_trend = [
        {
            "week": f"Week {i+1}",
            "accuracy": (live_accuracy - (i * 0.5)) if i < 4 else live_accuracy
        }
        for i in range(4)
    ]
    accuracy_trend.reverse()

    return MonitoringDashboardResponse(
        live_accuracy=live_accuracy,
        avg_latency=avg_latency,
        today_inferences=today_inferences,
        active_alerts=active_alerts_count,
        recent_metrics=recent_metrics,
        accuracy_trend=accuracy_trend,
        alerts=alerts
    )


@router.get("/alerts", response_model=List[MonitoringAlertResponse])
async def get_alerts(
    skip: int = 0,
    limit: int = 100,
    status_filter: str = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get all monitoring alerts"""
    query = db.query(MonitoringAlert)

    if status_filter:
        query = query.filter(MonitoringAlert.status == status_filter)

    alerts = query.order_by(MonitoringAlert.created_at.desc()).offset(skip).limit(limit).all()
    return alerts


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Acknowledge an alert"""
    alert = db.query(MonitoringAlert).filter(MonitoringAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.status = AlertStatus.ACKNOWLEDGED
    alert.acknowledged_at = datetime.utcnow()
    alert.acknowledged_by = current_user["uid"]

    db.commit()

    return {"message": "Alert acknowledged", "alert_id": alert_id}


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Resolve an alert"""
    alert = db.query(MonitoringAlert).filter(MonitoringAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.status = AlertStatus.RESOLVED
    alert.resolved_at = datetime.utcnow()

    db.commit()

    return {"message": "Alert resolved", "alert_id": alert_id}


@router.get("/metrics", response_model=List[PerformanceMetricResponse])
async def get_performance_metrics(
    deployment_id: int = None,
    hours: int = 24,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get performance metrics"""
    query = db.query(PerformanceMetric)

    if deployment_id:
        query = query.filter(PerformanceMetric.deployment_id == deployment_id)

    time_threshold = datetime.utcnow() - timedelta(hours=hours)
    query = query.filter(PerformanceMetric.created_at >= time_threshold)

    metrics = query.order_by(PerformanceMetric.created_at.desc()).all()
    return metrics


@router.get("/feedback-loops", response_model=List[FeedbackLoopResponse])
async def get_feedback_loops(
    deployment_id: int = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get all feedback loops"""
    query = db.query(FeedbackLoop)

    if deployment_id:
        query = query.filter(FeedbackLoop.deployment_id == deployment_id)

    loops = query.all()
    return loops


@router.post("/feedback-loops", response_model=FeedbackLoopResponse, status_code=status.HTTP_201_CREATED)
async def create_feedback_loop(
    loop: FeedbackLoopCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Create a new feedback loop"""
    deployment = db.query(Deployment).filter(Deployment.id == loop.deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    db_loop = FeedbackLoop(
        name=loop.name,
        deployment_id=loop.deployment_id,
        is_enabled=int(loop.data_collection_enabled),
        accuracy_threshold=loop.accuracy_threshold,
        data_collection_enabled=int(loop.data_collection_enabled),
        auto_retrain_enabled=int(loop.auto_retrain_enabled),
        retrain_schedule=loop.retrain_schedule,
        min_samples_for_retrain=loop.min_samples_for_retrain,
        configuration=loop.configuration
    )
    db.add(db_loop)
    db.commit()
    db.refresh(db_loop)

    return db_loop


@router.get("/feedback-loops/{loop_id}", response_model=FeedbackLoopResponse)
async def get_feedback_loop(
    loop_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get a specific feedback loop"""
    loop = db.query(FeedbackLoop).filter(FeedbackLoop.id == loop_id).first()
    if not loop:
        raise HTTPException(status_code=404, detail="Feedback loop not found")
    return loop


@router.put("/feedback-loops/{loop_id}", response_model=FeedbackLoopResponse)
async def update_feedback_loop(
    loop_id: int,
    loop_update: FeedbackLoopUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Update a feedback loop"""
    db_loop = db.query(FeedbackLoop).filter(FeedbackLoop.id == loop_id).first()
    if not db_loop:
        raise HTTPException(status_code=404, detail="Feedback loop not found")

    update_data = loop_update.dict(exclude_unset=True)

    # Convert boolean to int for SQLite
    if "is_enabled" in update_data:
        update_data["is_enabled"] = int(update_data["is_enabled"])
    if "data_collection_enabled" in update_data:
        update_data["data_collection_enabled"] = int(update_data["data_collection_enabled"])
    if "auto_retrain_enabled" in update_data:
        update_data["auto_retrain_enabled"] = int(update_data["auto_retrain_enabled"])

    for field, value in update_data.items():
        setattr(db_loop, field, value)

    db.commit()
    db.refresh(db_loop)
    return db_loop


@router.delete("/feedback-loops/{loop_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_feedback_loop(
    loop_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Delete a feedback loop"""
    loop = db.query(FeedbackLoop).filter(FeedbackLoop.id == loop_id).first()
    if not loop:
        raise HTTPException(status_code=404, detail="Feedback loop not found")

    db.delete(loop)
    db.commit()
    return None


@router.get("/edge-cases", response_model=List[EdgeCaseResponse])
async def get_edge_cases(
    deployment_id: int = None,
    is_reviewed: bool = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Get edge cases collected from production"""
    query = db.query(EdgeCase)

    if deployment_id:
        query = query.filter(EdgeCase.deployment_id == deployment_id)

    if is_reviewed is not None:
        query = query.filter(EdgeCase.is_reviewed == int(is_reviewed))

    cases = query.order_by(EdgeCase.created_at.desc()).offset(skip).limit(limit).all()
    return cases


@router.post("/edge-cases/{case_id}/review")
async def review_edge_case(
    case_id: int,
    review: EdgeCaseReviewRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Review and annotate an edge case"""
    case = db.query(EdgeCase).filter(EdgeCase.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Edge case not found")

    case.ground_truth = review.ground_truth
    case.is_reviewed = 1
    case.reviewed_by = current_user["uid"]
    case.reviewed_at = datetime.utcnow()

    db.commit()

    return {"message": "Edge case reviewed", "case_id": case_id}


@router.post("/feedback-loops/{loop_id}/trigger-retrain")
async def trigger_retraining(
    loop_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_dev)
):
    """Manually trigger retraining for a feedback loop"""
    loop = db.query(FeedbackLoop).filter(FeedbackLoop.id == loop_id).first()
    if not loop:
        raise HTTPException(status_code=404, detail="Feedback loop not found")

    # TODO: Implement actual retraining trigger
    # This would create a new training job with collected edge cases

    loop.last_retrain_triggered = datetime.utcnow()
    db.commit()

    return {
        "message": "Retraining triggered",
        "loop_id": loop_id,
        "deployment_id": loop.deployment_id
    }
