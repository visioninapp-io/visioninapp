import logging
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.dataset import Annotation, GeometryType, Dataset
from app.models.asset import Asset
from app.models.label_class import LabelClass
from app.utils.file_storage import file_storage
from app.utils.dataset_helper import ensure_yolo_index_for_dataset, upload_data_yaml_for_dataset

log = logging.getLogger(__name__)


def handle_inference_done(payload: dict):
    """
    inference.done 이벤트 처리

    Expected payload:
    {
        "job_id": "uuid",
        "dataset_id": 1,
        "status": "completed" | "failed",
        "s3_labels_path": "pothole/labels",
        "processed_images": [
            {
                "asset_id": 123,
                "filename": "pothole_1.jpg",
                "label_file": "pothole_1.txt",
                "annotation_count": 5
            }
        ],
        "error_message": null
    }
    """
    job_id = payload.get("job_id")
    dataset_id = payload.get("dataset_id")
    status = payload.get("status")
    s3_labels_path = payload.get("s3_labels_path")
    processed_images = payload.get("processed_images", [])
    error_message = payload.get("error_message")
    overwrite_existing = payload.get("overwrite_existing", False)

    log.info(f"[INFERENCE-DONE] Processing job_id={job_id}, dataset_id={dataset_id}, status={status}")

    if status == "failed":
        log.error(f"[INFERENCE-DONE] Job failed: {error_message}")
        return

    # DB 세션 생성
    db = SessionLocal()

    try:
        # 데이터셋 확인
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            log.error(f"[INFERENCE-DONE] Dataset not found: {dataset_id}")
            return

        # Get dataset version and ontology
        from app.utils.dataset_helper import get_or_create_dataset_version
        dataset_version = get_or_create_dataset_version(db, dataset_id, 'v0')
        if not dataset_version:
            log.error(f"[INFERENCE-DONE] Dataset version not found for dataset {dataset_id}")
            return

        ontology = dataset_version.ontology_version
        if not ontology:
            log.error(f"[INFERENCE-DONE] Ontology not found for dataset {dataset_id}")
            return

        log.info(f"[INFERENCE-DONE] Using ontology_id={ontology.id} for dataset {dataset_id}")

        # S3에서 data.yaml 읽어서 클래스 이름 가져오기
        data_yaml_key = f"datasets/{dataset.name}/data.yaml"
        class_names = {}
        try:
            yaml_content = file_storage.download_from_s3(data_yaml_key)
            if yaml_content:
                import yaml
                data_yaml = yaml.safe_load(yaml_content.decode("utf-8"))
                names = data_yaml.get("names", [])
                # names를 yolo_index로 매핑 (index 0부터 시작)
                class_names = {i: name for i, name in enumerate(names)}
                log.info(f"[INFERENCE-DONE] Loaded {len(class_names)} classes from data.yaml: {class_names}")
        except Exception as e:
            log.warning(f"[INFERENCE-DONE] Failed to load data.yaml from S3: {str(e)}")

        total_annotations = 0

        # S3에서 label 파일 읽기 및 DB 저장
        for img_data in processed_images:
            asset_id = img_data.get("asset_id")
            label_filename = img_data.get("label_file")  # 예: "pothole_1.txt"

            if not label_filename:
                continue

            # S3에서 label 파일 다운로드
            # s3_labels_path should be like "datasets/pothole/labels" or "pothole/labels"
            # Ensure it starts with "datasets/" prefix
            if not s3_labels_path.startswith("datasets/"):
                s3_label_key = f"datasets/{s3_labels_path}/{label_filename}"
            else:
                s3_label_key = f"{s3_labels_path}/{label_filename}"
            log.info(f"[INFERENCE-DONE] Downloading label from S3: {s3_label_key}")

            try:
                label_content = file_storage.download_from_s3(s3_label_key)
                if not label_content:
                    log.warning(f"[INFERENCE-DONE] Label file not found: {s3_label_key}")
                    continue

                # YOLO format 파싱 (class_id x_center y_center width height)
                lines = label_content.decode("utf-8").strip().split("\n")
                if not lines or (len(lines) == 1 and not lines[0].strip()):
                    log.info(f"[INFERENCE-DONE] Empty label file: {s3_label_key}")
                    continue

                # overwrite_existing 처리
                if overwrite_existing:
                    existing_annotations = db.query(Annotation).filter(Annotation.asset_id == asset_id).all()
                    if existing_annotations:
                        log.info(f"[INFERENCE-DONE] Deleting {len(existing_annotations)} existing annotations for asset {asset_id}")
                        for ann in existing_annotations:
                            db.delete(ann)
                        db.flush()

                # Annotation 생성
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        log.warning(f"[INFERENCE-DONE] Invalid line format (expected 5 parts): {line}")
                        continue

                    yolo_class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # yolo_index로 LabelClass 찾기 (ontology version으로 필터링)
                    label_class = db.query(LabelClass).filter(
                        LabelClass.ontology_version_id == ontology.id,
                        LabelClass.yolo_index == yolo_class_id
                    ).first()

                    # LabelClass가 없으면 자동 생성 (data.yaml의 names 사용)
                    if not label_class:
                        log.warning(f"[INFERENCE-DONE] LabelClass not found for ontology_id={ontology.id}, yolo_index={yolo_class_id}")

                        # data.yaml에서 클래스 이름 가져오기
                        class_name = class_names.get(yolo_class_id)
                        if not class_name:
                            class_name = f"class_{yolo_class_id}"

                        # 랜덤 색상 생성
                        import random
                        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

                        # LabelClass 생성
                        label_class = LabelClass(
                            ontology_version_id=ontology.id,
                            display_name=class_name,
                            color=color,
                            yolo_index=yolo_class_id
                        )
                        db.add(label_class)
                        db.flush()  # ID 생성
                        log.info(f"[INFERENCE-DONE] Created new LabelClass: {class_name} (yolo_index={yolo_class_id}, id={label_class.id})")

                    # Geometry 데이터
                    geometry_data = {
                        "bbox": {
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height
                        }
                    }

                    # Annotation 저장
                    db_annotation = Annotation(
                        asset_id=asset_id,
                        label_class_id=label_class.id,
                        geometry_type=GeometryType.BBOX,
                        geometry=geometry_data,
                        is_normalized=True,
                        source="model",
                        confidence=0.95,  # GPU에서 전달하지 않으면 기본값
                        annotator_name="auto-annotation"
                    )
                    db.add(db_annotation)
                    total_annotations += 1

            except Exception as e:
                log.error(f"[INFERENCE-DONE] Error processing label {s3_label_key}: {e}", exc_info=True)
                continue

        # Commit
        db.commit()
        log.info(f"[INFERENCE-DONE] Saved {total_annotations} annotations to DB")

        # data.yaml 업데이트
        upload_data_yaml_for_dataset(db, dataset_id)
        log.info(f"[INFERENCE-DONE] Updated data.yaml for dataset {dataset_id}")

    except Exception as e:
        log.error(f"[INFERENCE-DONE] Error handling inference done: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()
