"""데이터셋 관련 헬퍼 함수"""
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional
from app.models.dataset import Dataset, DatasetVersion, Annotation
from app.models.dataset_split import DatasetSplit, DatasetSplitType
from app.models.label_ontology_version import LabelOntologyVersion
from app.models.label_class import LabelClass
from app.models.asset import Asset, AssetType
from app.models.project import Project

import boto3, yaml, re
from app.core.config import settings

def get_or_create_label_ontology_for_dataset_version(
    db: Session,
    dataset_version_id: int,
    version_tag: str = "v1.0",
) -> LabelOntologyVersion:
    """데이터셋 버전 단위의 레이블 온톨로지 버전을 가져오거나 생성"""
    ontology = (
        db.query(LabelOntologyVersion)
        .filter(
            LabelOntologyVersion.dataset_version_id == dataset_version_id,
            LabelOntologyVersion.version_tag == version_tag,
        )
        .first()
    )
    if not ontology:
        ontology = LabelOntologyVersion(
            dataset_version_id=dataset_version_id,
            version_tag=version_tag,
            is_frozen=False,
        )
        db.add(ontology)
        db.flush()
    return ontology


def get_latest_label_ontology_version(
    db: Session,
    dataset_version_id: int
) -> Optional[LabelOntologyVersion]:
    """데이터셋 버전의 최신 레이블 온톨로지 버전 조회"""
    return (
        db.query(LabelOntologyVersion)
        .filter(LabelOntologyVersion.dataset_version_id == dataset_version_id)
        .order_by(LabelOntologyVersion.created_at.desc())
        .first()
    )


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
        # 1) 우선 ontology 없이 버전부터 만든다 (nullable=True가 전제)
        version = DatasetVersion(
            dataset_id=dataset_id,
            ontology_version_id=None,
            version_tag=version_tag,
            is_frozen=False
        )
        db.add(version)
        db.flush()  # version.id 확보

        # 2) dataset_version 단위 ontology 생성/획득
        ontology = get_or_create_label_ontology_for_dataset_version(
            db, dataset_version_id=version.id, version_tag="v1.0"
        )
        # 3) version에 ontology FK 세팅
        version.ontology_version_id = ontology.id
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
    """에셋 추가 시 새 버전 생성 (dataset_version 단위 온톨로지 귀속)"""
    # 1) Dataset 확인
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")

    # 2) version_tag 자동 생성 로직 유지
    if version_tag is None:
        latest_version = get_latest_dataset_version(db, dataset_id)
        if latest_version and latest_version.version_tag:
            m = re.search(r'v?(\d+)', latest_version.version_tag)
            if m:
                version_num = int(m.group(1))
                version_tag = f"v{version_num + 1}.0"
            else:
                version_tag = "v2.0"
        else:
            version_tag = "v1.0"

    # 3) 우선 온톨로지 없이 DatasetVersion 생성 (nullable FK 전제)
    version = DatasetVersion(
        dataset_id=dataset_id,
        ontology_version_id=None,  # ← 먼저 None으로 생성
        version_tag=version_tag,
        is_frozen=False
    )
    db.add(version)
    db.flush()  # version.id 확보

    # 4) dataset_version 단위 온톨로지 생성/획득 후 FK 연결
    ontology = get_or_create_label_ontology_for_dataset_version(
        db,
        dataset_version_id=version.id,
        version_tag="v1.0"  # 정책상 v1.0 고정
    )
    version.ontology_version_id = ontology.id
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
    limit: int = 1000,
    version_id: Optional[int] = None,
    version_tag: Optional[str] = None
) -> tuple[list[Asset], int]:
    """
    데이터셋에서 Asset 조회
    
    Args:
        dataset_id: Dataset ID
        split_type: Optional split type filter
        asset_type: Optional asset type filter
        page: Page number (1-based)
        limit: Page size
        version_id: Optional specific version ID to filter by
        version_tag: Optional version tag (e.g., 'v0', 'v1.0') - used if version_id not provided
    
    Returns:
        (assets, total_count)
    """
    # Get version - prefer version_id, then version_tag, then default to 'v0'
    if version_id:
        version = db.query(DatasetVersion).filter(
            DatasetVersion.id == version_id,
            DatasetVersion.dataset_id == dataset_id
        ).first()
    elif version_tag:
        version = get_or_create_dataset_version(db, dataset_id, version_tag)
    else:
        # Default to v0 (backward compatibility)
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

def ensure_yolo_index_for_dataset(db: Session, dataset_id: int, label_class: LabelClass) -> int:
    """
    주어진 dataset_id(=v0 기준)에서 label_class가 처음 쓰이면
    해당 데이터셋의 온톨로지 범위에서 yolo_index를 0부터 순차 부여한다.
    이미 값이 있으면 그대로 반환한다.
    """
    # 0도 유효한 값이므로 None만 새로 부여 대상
    if label_class.yolo_index is not None:
        return label_class.yolo_index

    # 1) v0 DatasetVersion 확보(없으면 생성)
    dataset_version = get_or_create_dataset_version(db, dataset_id, 'v0')
    if not dataset_version:
        raise ValueError(f"dataset_id={dataset_id} has no dataset_version (v0)")

    # 2) v0가 참조하는 Ontology 확보(없으면 생성 후 연결)
    ontology = dataset_version.ontology_version
    if ontology is None:
        ontology = get_or_create_label_ontology_for_dataset_version(
            db, dataset_version_id=dataset_version.id, version_tag="v1.0"
        )
        # 양방향 FK 유지 시: DV.fk 채워주고 flush
        dataset_version.ontology_version_id = ontology.id
        db.flush()

    # 3) 해당 Ontology 내에서만 최대 yolo_index 조회 → 0-base 부여
    max_idx = (
        db.query(func.max(LabelClass.yolo_index))
          .filter(LabelClass.ontology_version_id == ontology.id)
          .scalar()
    )
    next_idx = 0 if max_idx is None else int(max_idx) + 1

    # 4) 인덱스 부여 (label_class의 ontology는 생성 시점에 이미 매핑되어 있다고 가정)
    label_class.yolo_index = next_idx
    db.add(label_class)
    db.flush()  # 커밋은 호출부에서

    print(f"[YOLO_INDEX] dataset_id={dataset_id}, class='{label_class.display_name}', index={next_idx}")
    return next_idx


def upload_data_yaml_for_dataset(db: Session, dataset_id: int) -> str:
    """
    이 데이터셋에서 '실제로 사용된' 클래스만 yolo_index 순으로 names에 넣어
    YOLO 학습용 data.yaml을 S3에 업로드한다.
    - 보장 로직 없음: dataset_version/ontology 없으면 에러 발생
    - 어노테이션 없으면 nc:0, names:[] (의도된 동작)
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise ValueError(f"dataset_id={dataset_id} not found")
    dataset_name = dataset.name or f"dataset_{dataset_id}"

    dv = (
        db.query(DatasetVersion)
          .filter(DatasetVersion.dataset_id == dataset_id)
          .order_by(DatasetVersion.created_at.asc())
          .first()
    )
    if not dv:
        raise ValueError(f"dataset_id={dataset_id} has no dataset_version")
    if dv.ontology_version_id is None:
        raise ValueError(f"dataset_id={dataset_id} has no ontology_version attached")

    # 실제로 사용된(어노테이션 존재) + 현재 온톨로지에 속하는 클래스만, yolo_index 순
    rows_used = (
        db.query(LabelClass)
          .join(Annotation, Annotation.label_class_id == LabelClass.id)
          .join(Asset, Asset.id == Annotation.asset_id)
          .join(DatasetSplit, DatasetSplit.id == Asset.dataset_split_id)
          .join(DatasetVersion, DatasetVersion.id == DatasetSplit.dataset_version_id)
          .filter(
              DatasetVersion.dataset_id == dataset_id,
              LabelClass.ontology_version_id == dv.ontology_version_id,
              LabelClass.yolo_index.isnot(None),
          )
          .distinct()
          .order_by(LabelClass.yolo_index.asc(), LabelClass.id.asc())
          .all()
    )

    names = [lc.display_name for lc in rows_used]  # 어노테이션 없으면 빈 리스트(정상)
    names_inline = yaml.safe_dump(
        names, allow_unicode=True, default_flow_style=True, default_style="'"
    ).strip()
    body = f"nc: {len(names)}\nnames: {names_inline}\n"

    s3 = boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
    )
    key = f"datasets/{dataset_name}/data.yaml"
    s3.put_object(
        Bucket=settings.AWS_BUCKET_NAME,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="text/yaml",
    )
    return key