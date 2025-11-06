"""프로젝트 관련 헬퍼 함수"""
from sqlalchemy.orm import Session
from app.models.project import Project


def get_or_create_default_project(db: Session, user_uid: str) -> Project:
    """
    사용자의 기본 프로젝트를 가져오거나 생성
    
    Args:
        db: 데이터베이스 세션
        user_uid: 사용자 UID (Firebase UID)
        
    Returns:
        Project: 사용자의 기본 프로젝트
    """
    # 사용자별 기본 프로젝트 이름 생성 (UID 앞 8자리 사용)
    project_name = f"Default Project ({user_uid[:8]})"
    
    # 기존 프로젝트 찾기
    project = db.query(Project).filter(
        Project.name == project_name
    ).first()
    
    if not project:
        # 프로젝트가 없으면 생성 (ID는 자동으로 AUTO_INCREMENT)
        project = Project(
            name=project_name,
            description="Automatically created default project"
        )
        db.add(project)
        db.commit()
        db.refresh(project)
    
    return project

