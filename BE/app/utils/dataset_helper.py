"""데이터셋 관련 헬퍼 함수"""
from sqlalchemy.orm import Session
from typing import Optional
from app.models.dataset import Dataset, DatasetVersion
from app.models.dataset_split import DatasetSplit, DatasetSplitType
from app.models.label_ontology_version import LabelOntologyVersion
from app.models.label_class import LabelClass
from app.models.asset import Asset, AssetType
from app.models.project import Project


def get_or_create_label_ontology_version(
    db: Session, 
    project_id: int, 
    version_tag: str = "v1.0"
) -> LabelOntologyVersion:
    """프로젝트의 레이블 온톨로지 버전을 가져오거나 생성"""
    ontology = db.query(LabelOntologyVersion).filter(
        LabelOntologyVersion.project_id == project_id,
        LabelOntologyVersion.version_tag == version_tag
    ).first()
    
    if not ontology:
        ontology = LabelOntologyVersion(
            project_id=project_id,
            version_tag=version_tag,
            is_frozen=False
        )
        db.add(ontology)
        db.flush()
    
    return ontology


def get_latest_label_ontology_version(
    db: Session, 
    project_id: int
) -> Optional[LabelOntologyVersion]:
    """프로젝트의 최신 레이블 온톨로지 버전 조회"""
    return db.query(LabelOntologyVersion).filter(
        LabelOntologyVersion.project_id == project_id
    ).order_by(LabelOntologyVersion.created_at.desc()).first()


def get_or_create_dataset_version(
    db: Session,
    dataset_id: int,
    version_tag: str = "v1.0"
) -> DatasetVersion:
    """데이터셋 버전을 가져오거나 생성"""
    # Dataset 조회
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    # DatasetVersion 찾기 (version_tag로만 검색)
    version = db.query(DatasetVersion).filter(
        DatasetVersion.dataset_id == dataset_id,
        DatasetVersion.version_tag == version_tag
    ).first()
    
    if not version:
        # 온톨로지 v1.0 고정 (단일 온톨로지 정책)
        ontology = get_or_create_label_ontology_version(
            db, 
            dataset.project_id, 
            "v1.0"
        )
        
        version = DatasetVersion(
            dataset_id=dataset_id,
            ontology_version_id=ontology.id,
            version_tag=version_tag,
            is_frozen=False
        )
        db.add(version)
        db.flush()
    
    return version


def get_or_create_dataset_split(
    db: Session,
    dataset_version_id: int,
    split_type: DatasetSplitType = DatasetSplitType.UNASSIGNED,
    ratio: float = 1.0
) -> DatasetSplit:
    """데이터셋 스플릿을 가져오거나 생성"""
    split = db.query(DatasetSplit).filter(
        DatasetSplit.dataset_version_id == dataset_version_id,
        DatasetSplit.split == split_type
    ).first()
    
    if not split:
        split = DatasetSplit(
            dataset_version_id=dataset_version_id,
            split=split_type,
            ratio=ratio
        )
        db.add(split)
        db.flush()
    
    return split


def get_latest_dataset_version(db: Session, dataset_id: int) -> Optional[DatasetVersion]:
    """데이터셋의 최신 버전 조회"""
    return db.query(DatasetVersion).filter(
        DatasetVersion.dataset_id == dataset_id
    ).order_by(DatasetVersion.created_at.desc()).first()


def create_new_dataset_version(
    db: Session,
    dataset_id: int,
    version_tag: Optional[str] = None
) -> DatasetVersion:
    """에셋 추가 시 새 버전 생성"""
    # Dataset 조회
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    # 버전 태그가 지정되지 않으면 자동 생성
    if version_tag is None:
        # 최신 버전 찾기
        latest_version = get_latest_dataset_version(db, dataset_id)
        
        if latest_version:
            # 버전 태그에서 숫자 추출 (v1.0 -> 1, v2.0 -> 2)
            import re
            match = re.search(r'v?(\d+)', latest_version.version_tag)
            if match:
                version_num = int(match.group(1))
                version_tag = f"v{version_num + 1}.0"
            else:
                # 숫자를 찾을 수 없으면 기본값
                version_tag = "v2.0"
        else:
            # 첫 번째 버전
            version_tag = "v1.0"
    
    # 온톨로지 v1.0 고정 (단일 온톨로지 정책)
    ontology = get_or_create_label_ontology_version(
        db, 
        dataset.project_id, 
        "v1.0"
    )
    
    # 새 DatasetVersion 생성
    version = DatasetVersion(
        dataset_id=dataset_id,
        ontology_version_id=ontology.id,
        version_tag=version_tag,
        is_frozen=False
    )
    db.add(version)
    db.flush()
    
    return version


def format_asset_as_image(asset: Asset) -> dict:
    """Asset을 기존 Image API 응답 형식으로 변환 (하위 호환)"""
    filename = asset.storage_uri.split('/')[-1] if asset.storage_uri else f"asset_{asset.id}"
    
    return {
        "id": asset.id,
        "filename": filename,
        "file_path": asset.storage_uri,
        "file_size": asset.bytes,
        "width": asset.width,
        "height": asset.height,
        "is_annotated": len(asset.annotations) > 0 if hasattr(asset, 'annotations') else False,
        "created_at": asset.created_at.isoformat() if asset.created_at else None
    }


def format_assets_as_images(assets: list[Asset]) -> list[dict]:
    """Asset 리스트를 Image 형식 리스트로 변환"""
    return [format_asset_as_image(asset) for asset in assets]


def get_assets_from_dataset(
    db: Session,
    dataset_id: int,
    split_type: Optional[DatasetSplitType] = None,
    asset_type: Optional[AssetType] = None,
    page: int = 1,
    limit: int = 1000
) -> tuple[list[Asset], int]:
    """
    데이터셋에서 Asset 조회
    
    Returns:
        (assets, total_count)
    """
    # v0 버전 조회 (고정 정책)
    version = get_or_create_dataset_version(db, dataset_id, 'v0')
    
    if not version:
        # 버전이 없으면 빈 리스트 반환
        return [], 0
    
    # DatasetSplit 조회
    query = db.query(Asset).join(DatasetSplit).filter(
        DatasetSplit.dataset_version_id == version.id
    )
    
    # Split 타입 필터
    if split_type:
        query = query.filter(DatasetSplit.split == split_type)
    
    # Asset 타입 필터
    if asset_type:
        query = query.filter(Asset.type == asset_type)
    
    # 총 개수
    total_count = query.count()
    
    # 페이징
    offset = (page - 1) * limit
    assets = query.offset(offset).limit(limit).all()
    
    return assets, total_count


def enrich_version_response(db: Session, version: DatasetVersion) -> dict:
    """DatasetVersion에 계산된 필드들을 추가하여 응답 형식으로 변환"""
    # Get splits with asset counts
    splits_data = []
    total_assets = 0
    
    splits = db.query(DatasetSplit).filter(
        DatasetSplit.dataset_version_id == version.id
    ).all()
    
    for split in splits:
        asset_count = db.query(Asset).filter(
            Asset.dataset_split_id == split.id
        ).count()
        
        splits_data.append({
            "split": split.split.value,
            "ratio": split.ratio,
            "asset_count": asset_count
        })
        total_assets += asset_count
    
    return {
        "id": version.id,
        "dataset_id": version.dataset_id,
        "ontology_version_id": version.ontology_version_id,
        "version_tag": version.version_tag,
        "is_frozen": version.is_frozen,
        "created_at": version.created_at,
        "splits": splits_data,
        "total_assets": total_assets
    }
