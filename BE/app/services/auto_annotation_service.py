<<<<<<< HEAD
"""
자동 어노테이션 서비스

YOLO 모델을 사용하여 이미지에 자동으로 어노테이션을 생성합니다.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AutoAnnotationService:
    """자동 어노테이션 서비스"""

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: YOLO 모델 파일 경로 (.pt)
                       None이면 기본 경로 사용
        """
        self.model = None
        self.model_path = model_path or self._get_default_model_path()
        self.is_loaded = False

    def _get_default_model_path(self) -> Path:
        """기본 모델 경로 반환"""
        # AI/models/best.pt 사용
        # BE/app/services/auto_annotation_service.py -> go up to project root
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        model_path = project_root / "AI" / "models" / "best.pt"
        logger.debug(f"Default model path: {model_path}")
        return model_path

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        YOLO 모델 로드

        Args:
            model_path: 모델 파일 경로 (None이면 기본 경로 사용)

        Returns:
            성공 여부
        """
        try:
            from ultralytics import YOLO

            # 경로가 지정되면 업데이트
            if model_path:
                self.model_path = model_path

            # 지정된 경로가 있는지 확인
            model_path_obj = Path(self.model_path) if self.model_path else None
            
            # .pth 파일이면 .pt로 변환 시도 또는 경고
            if model_path_obj and model_path_obj.exists():
                if model_path_obj.suffix == '.pth':
                    logger.warning(f"⚠️  PyTorch 모델 (.pth) 감지: {self.model_path}")
                    logger.warning(f"   YOLO는 .pt 형식만 지원합니다.")
                    logger.warning(f"   YOLO 아키텍처로 모델을 다시 훈련하거나 YOLOv8n을 사용합니다.")
                    # .pth는 로드 실패하므로 기본 모델로 fallback
                    model_path_obj = None
                elif model_path_obj.suffix == '.pt':
                    logger.info(f"모델 로드 중: {self.model_path}")
                    self.model = YOLO(str(self.model_path))
                    self.is_loaded = True
                    logger.info(f"✅ 커스텀 모델 로드 완료")
                    logger.info(f"   클래스 수: {len(self.model.names)}")
                    return True
            
            # 커스텀 모델이 없거나 .pth 파일이면 yolov8n 사용 (자동 다운로드)
            if not model_path_obj or not model_path_obj.exists():
                logger.warning(f"커스텀 모델을 찾을 수 없습니다: {self.model_path}")
            
            logger.info("기본 YOLOv8n 모델을 사용합니다 (필요시 자동 다운로드)")
            
            self.model = YOLO("yolov8n.pt")  # Ultralytics will auto-download if needed
            self.model_path = "yolov8n.pt"
            self.is_loaded = True
            
            logger.info(f"✅ YOLOv8n 모델 로드 완료")
            logger.info(f"   클래스 수: {len(self.model.names)}")
            logger.info(f"   참고: YOLO 커스텀 모델을 사용하려면 'yolov8' 아키텍처로 훈련하세요")

            return True

        except ImportError:
            logger.error("ultralytics 패키지가 설치되지 않았습니다.")
            logger.info("pip install ultralytics 를 실행하세요.")
            return False

        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return False

    def predict_image(
        self,
        image_path: str,
        conf_threshold: float = 0.25
    ) -> List[Dict]:
        """
        단일 이미지에 대한 추론 실행

        Args:
            image_path: 이미지 파일 경로
            conf_threshold: 신뢰도 임계값 (0-1)

        Returns:
            어노테이션 리스트
            [{
                'class_id': int,
                'class_name': str,
                'confidence': float,
                'bbox': {
                    'x_center': float,  # 정규화 (0-1)
                    'y_center': float,
                    'width': float,
                    'height': float
                },
                'bbox_xyxy': [x1, y1, x2, y2]  # 절대 좌표
            }, ...]
        """
        if not self.is_loaded:
            if not self.load_model():
                return []

        try:
            # 추론 실행
            results = self.model.predict(
                source=str(image_path),
                conf=conf_threshold,
                verbose=False
            )

            if not results or len(results) == 0:
                return []

            # 결과 파싱
            annotations = []
            result = results[0]
            boxes = result.boxes

            # 이미지 크기 (정규화를 위해)
            img_height, img_width = result.orig_shape

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.model.names[cls_id]

                # XYXY 좌표 (절대 좌표)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # YOLO 포맷 (정규화된 중심점 + 너비/높이)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                annotation = {
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': conf,
                    'bbox': {
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    },
                    'bbox_xyxy': [x1, y1, x2, y2]
                }

                annotations.append(annotation)

            logger.info(f"✅ 추론 완료: {image_path}")
            logger.info(f"   탐지된 객체 수: {len(annotations)}")

            return annotations

        except Exception as e:
            logger.error(f"추론 실패: {e}")
            return []

    def predict_batch(
        self,
        image_paths: List[str],
        conf_threshold: float = 0.25,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, List[Dict]]:
        """
        배치 이미지 추론

        Args:
            image_paths: 이미지 파일 경로 리스트
            conf_threshold: 신뢰도 임계값
            progress_callback: 진행률 콜백 함수(current, total)

        Returns:
            {image_path: annotations} 딕셔너리
        """
        results = {}
        total = len(image_paths)

        for i, image_path in enumerate(image_paths):
            annotations = self.predict_image(image_path, conf_threshold)
            results[image_path] = annotations

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def get_model_info(self) -> Dict:
        """
        모델 정보 반환

        Returns:
            {
                'model_path': str,
                'is_loaded': bool,
                'num_classes': int,
                'class_names': list,
                'model_type': str
            }
        """
        info = {
            'model_path': str(self.model_path),
            'is_loaded': self.is_loaded,
            'num_classes': 0,
            'class_names': [],
            'model_type': 'YOLOv8'
        }

        if self.is_loaded and self.model:
            info['num_classes'] = len(self.model.names)
            info['class_names'] = list(self.model.names.values())

        return info


# 싱글톤 인스턴스
_auto_annotation_service = None


def get_auto_annotation_service() -> AutoAnnotationService:
    """
    AutoAnnotationService 싱글톤 인스턴스 반환
    """
    global _auto_annotation_service

    if _auto_annotation_service is None:
        _auto_annotation_service = AutoAnnotationService()

    return _auto_annotation_service
=======
"""
자동 어노테이션 서비스

YOLO 모델을 사용하여 이미지에 자동으로 어노테이션을 생성합니다.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AutoAnnotationService:
    """자동 어노테이션 서비스"""

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: YOLO 모델 파일 경로 (.pt)
                       None이면 기본 경로 사용
        """
        self.model = None
        self.model_path = model_path or self._get_default_model_path()
        self.is_loaded = False

    def _get_default_model_path(self) -> Path:
        """기본 모델 경로 반환"""
        # AI/models/best.pt 사용
        project_root = Path(__file__).parent.parent.parent.parent
        model_path = project_root / "AI" / "models" / "best.pt"
        return model_path

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        YOLO 모델 로드

        Args:
            model_path: 모델 파일 경로 (None이면 기본 경로 사용)

        Returns:
            성공 여부
        """
        try:
            from ultralytics import YOLO

            # 경로가 지정되면 업데이트
            if model_path:
                self.model_path = model_path

            if not Path(self.model_path).exists():
                logger.error(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
                logger.info("AI/scripts/train_yolo.py를 먼저 실행하세요.")
                return False

            logger.info(f"모델 로드 중: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            self.is_loaded = True

            logger.info(f"✅ 모델 로드 완료")
            logger.info(f"   클래스 수: {len(self.model.names)}")

            return True

        except ImportError:
            logger.error("ultralytics 패키지가 설치되지 않았습니다.")
            logger.info("pip install ultralytics 를 실행하세요.")
            return False

        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return False

    def predict_image(
        self,
        image_path: str,
        conf_threshold: float = 0.25
    ) -> List[Dict]:
        """
        단일 이미지에 대한 추론 실행

        Args:
            image_path: 이미지 파일 경로
            conf_threshold: 신뢰도 임계값 (0-1)

        Returns:
            어노테이션 리스트
            [{
                'class_id': int,
                'class_name': str,
                'confidence': float,
                'bbox': {
                    'x_center': float,  # 정규화 (0-1)
                    'y_center': float,
                    'width': float,
                    'height': float
                },
                'bbox_xyxy': [x1, y1, x2, y2]  # 절대 좌표
            }, ...]
        """
        if not self.is_loaded:
            if not self.load_model():
                return []

        try:
            # 추론 실행
            results = self.model.predict(
                source=str(image_path),
                conf=conf_threshold,
                verbose=False
            )

            if not results or len(results) == 0:
                return []

            # 결과 파싱
            annotations = []
            result = results[0]
            boxes = result.boxes

            # 이미지 크기 (정규화를 위해)
            img_height, img_width = result.orig_shape

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.model.names[cls_id]

                # XYXY 좌표 (절대 좌표)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # YOLO 포맷 (정규화된 중심점 + 너비/높이)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                annotation = {
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': conf,
                    'bbox': {
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    },
                    'bbox_xyxy': [x1, y1, x2, y2]
                }

                annotations.append(annotation)

            logger.info(f"✅ 추론 완료: {image_path}")
            logger.info(f"   탐지된 객체 수: {len(annotations)}")

            return annotations

        except Exception as e:
            logger.error(f"추론 실패: {e}")
            return []

    def predict_batch(
        self,
        image_paths: List[str],
        conf_threshold: float = 0.25,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, List[Dict]]:
        """
        배치 이미지 추론

        Args:
            image_paths: 이미지 파일 경로 리스트
            conf_threshold: 신뢰도 임계값
            progress_callback: 진행률 콜백 함수(current, total)

        Returns:
            {image_path: annotations} 딕셔너리
        """
        results = {}
        total = len(image_paths)

        for i, image_path in enumerate(image_paths):
            annotations = self.predict_image(image_path, conf_threshold)
            results[image_path] = annotations

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def get_model_info(self) -> Dict:
        """
        모델 정보 반환

        Returns:
            {
                'model_path': str,
                'is_loaded': bool,
                'num_classes': int,
                'class_names': list,
                'model_type': str
            }
        """
        info = {
            'model_path': str(self.model_path),
            'is_loaded': self.is_loaded,
            'num_classes': 0,
            'class_names': [],
            'model_type': 'YOLOv8'
        }

        if self.is_loaded and self.model:
            info['num_classes'] = len(self.model.names)
            info['class_names'] = list(self.model.names.values())

        return info


# 싱글톤 인스턴스
_auto_annotation_service = None


def get_auto_annotation_service() -> AutoAnnotationService:
    """
    AutoAnnotationService 싱글톤 인스턴스 반환
    """
    global _auto_annotation_service

    if _auto_annotation_service is None:
        _auto_annotation_service = AutoAnnotationService()

    return _auto_annotation_service
>>>>>>> feature/llm-pipeline
