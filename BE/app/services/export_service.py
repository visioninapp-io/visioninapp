"""
Dataset Export Service

Handles exporting datasets to various formats (YOLO, COCO, Pascal VOC, etc.)
"""

import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import xml.etree.ElementTree as ET
from sqlalchemy.orm import Session


class ExportService:
    """Service for exporting datasets to various formats"""

    def __init__(self):
        self.base_dir = Path("uploads")
        self.export_dir = Path("exports")
        self.export_dir.mkdir(exist_ok=True)

    def export_dataset(
        self,
        export_id: int,
        dataset_id: Optional[int],
        version_id: Optional[int],
        export_format: str,
        include_images: bool,
        db: Session
    ) -> Dict[str, Any]:
        """
        Export dataset to specified format

        Returns dict with file_path and file_size
        """
        from app.models.dataset import Dataset, DatasetVersion, Image, Annotation

        # Get dataset or version
        if version_id:
            version = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
            if not version:
                raise ValueError(f"Version {version_id} not found")
            dataset_id = version.dataset_id

        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Get images and annotations
        images = db.query(Image).filter(Image.dataset_id == dataset_id).all()
        annotations_map = {}  # image_id -> List[Annotation]

        for img in images:
            anns = db.query(Annotation).filter(Annotation.image_id == img.id).all()
            annotations_map[img.id] = anns

        # Create export directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"{dataset.name}_{export_format}_{timestamp}"
        export_path = self.export_dir / export_name
        export_path.mkdir(parents=True, exist_ok=True)

        # Export based on format
        if export_format == "yolov8" or export_format == "yolov5":
            self._export_yolo(export_path, dataset, images, annotations_map, include_images)
        elif export_format == "coco":
            self._export_coco(export_path, dataset, images, annotations_map, include_images)
        elif export_format == "pascal_voc":
            self._export_pascal_voc(export_path, dataset, images, annotations_map, include_images)
        elif export_format == "csv":
            self._export_csv(export_path, dataset, images, annotations_map, include_images)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        # Create ZIP file
        zip_path = self.export_dir / f"{export_name}.zip"
        self._create_zip(export_path, zip_path)

        # Clean up directory
        shutil.rmtree(export_path)

        # Get file size
        file_size = zip_path.stat().st_size

        return {
            "file_path": str(zip_path),
            "file_size": file_size
        }

    def _export_yolo(
        self,
        export_path: Path,
        dataset,
        images: List,
        annotations_map: Dict,
        include_images: bool
    ):
        """Export in YOLO format"""
        # Create directories
        (export_path / "images").mkdir(exist_ok=True)
        (export_path / "labels").mkdir(exist_ok=True)

        # Create class mapping
        class_names = dataset.class_names if dataset.class_names else []

        # Write data.yaml
        data_yaml = {
            "path": str(export_path),
            "train": "images",
            "val": "images",
            "nc": len(class_names),
            "names": class_names
        }

        with open(export_path / "data.yaml", "w") as f:
            for key, value in data_yaml.items():
                if isinstance(value, list):
                    f.write(f"{key}: {value}\n")
                else:
                    f.write(f"{key}: {value}\n")

        # Export each image
        for img in images:
            # Copy image if requested
            if include_images:
                src_path = self.base_dir / img.file_path
                if src_path.exists():
                    dst_path = export_path / "images" / img.filename
                    shutil.copy(src_path, dst_path)

            # Write label file
            anns = annotations_map.get(img.id, [])
            if anns:
                label_filename = Path(img.filename).stem + ".txt"
                label_path = export_path / "labels" / label_filename

                with open(label_path, "w") as f:
                    for ann in anns:
                        # YOLO format: class_id x_center y_center width height
                        f.write(f"{ann.class_id} {ann.x_center} {ann.y_center} {ann.width} {ann.height}\n")

    def _export_coco(
        self,
        export_path: Path,
        dataset,
        images: List,
        annotations_map: Dict,
        include_images: bool
    ):
        """Export in COCO JSON format"""
        # Create images directory
        if include_images:
            (export_path / "images").mkdir(exist_ok=True)

        # Build COCO format
        coco_data = {
            "info": {
                "description": dataset.description or dataset.name,
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

        # Add categories
        class_names = dataset.class_names if dataset.class_names else []
        for idx, class_name in enumerate(class_names):
            coco_data["categories"].append({
                "id": idx,
                "name": class_name,
                "supercategory": "none"
            })

        # Add images and annotations
        annotation_id = 1
        for img_idx, img in enumerate(images):
            # Add image info
            coco_data["images"].append({
                "id": img.id,
                "file_name": img.filename,
                "width": img.width or 640,
                "height": img.height or 640,
                "date_captured": img.created_at.isoformat() if img.created_at else ""
            })

            # Copy image if requested
            if include_images:
                src_path = self.base_dir / img.file_path
                if src_path.exists():
                    dst_path = export_path / "images" / img.filename
                    shutil.copy(src_path, dst_path)

            # Add annotations
            anns = annotations_map.get(img.id, [])
            for ann in anns:
                # Convert normalized coordinates to absolute
                img_w = img.width or 640
                img_h = img.height or 640

                # COCO uses [x_min, y_min, width, height]
                x_min = (ann.x_center - ann.width / 2) * img_w
                y_min = (ann.y_center - ann.height / 2) * img_h
                width = ann.width * img_w
                height = ann.height * img_h

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": img.id,
                    "category_id": ann.class_id,
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "iscrowd": 0
                })
                annotation_id += 1

        # Write annotations.json
        with open(export_path / "annotations.json", "w") as f:
            json.dump(coco_data, f, indent=2)

    def _export_pascal_voc(
        self,
        export_path: Path,
        dataset,
        images: List,
        annotations_map: Dict,
        include_images: bool
    ):
        """Export in Pascal VOC XML format"""
        # Create directories
        (export_path / "Annotations").mkdir(exist_ok=True)
        if include_images:
            (export_path / "JPEGImages").mkdir(exist_ok=True)

        # Export each image
        for img in images:
            # Copy image if requested
            if include_images:
                src_path = self.base_dir / img.file_path
                if src_path.exists():
                    dst_path = export_path / "JPEGImages" / img.filename
                    shutil.copy(src_path, dst_path)

            # Create XML annotation
            anns = annotations_map.get(img.id, [])
            if anns:
                xml_filename = Path(img.filename).stem + ".xml"
                xml_path = export_path / "Annotations" / xml_filename

                # Create XML structure
                annotation = ET.Element("annotation")

                ET.SubElement(annotation, "folder").text = "JPEGImages"
                ET.SubElement(annotation, "filename").text = img.filename

                size = ET.SubElement(annotation, "size")
                ET.SubElement(size, "width").text = str(img.width or 640)
                ET.SubElement(size, "height").text = str(img.height or 640)
                ET.SubElement(size, "depth").text = "3"

                # Add objects
                for ann in anns:
                    obj = ET.SubElement(annotation, "object")
                    ET.SubElement(obj, "name").text = ann.class_name
                    ET.SubElement(obj, "pose").text = "Unspecified"
                    ET.SubElement(obj, "truncated").text = "0"
                    ET.SubElement(obj, "difficult").text = "0"

                    # Convert normalized to absolute coordinates
                    img_w = img.width or 640
                    img_h = img.height or 640

                    xmin = int((ann.x_center - ann.width / 2) * img_w)
                    ymin = int((ann.y_center - ann.height / 2) * img_h)
                    xmax = int((ann.x_center + ann.width / 2) * img_w)
                    ymax = int((ann.y_center + ann.height / 2) * img_h)

                    bndbox = ET.SubElement(obj, "bndbox")
                    ET.SubElement(bndbox, "xmin").text = str(max(0, xmin))
                    ET.SubElement(bndbox, "ymin").text = str(max(0, ymin))
                    ET.SubElement(bndbox, "xmax").text = str(min(img_w, xmax))
                    ET.SubElement(bndbox, "ymax").text = str(min(img_h, ymax))

                # Write XML file
                tree = ET.ElementTree(annotation)
                tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    def _export_csv(
        self,
        export_path: Path,
        dataset,
        images: List,
        annotations_map: Dict,
        include_images: bool
    ):
        """Export in CSV format"""
        import csv

        # Copy images if requested
        if include_images:
            (export_path / "images").mkdir(exist_ok=True)

        # Create CSV file
        csv_path = export_path / "annotations.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                "filename", "width", "height",
                "class_id", "class_name",
                "x_center", "y_center", "box_width", "box_height",
                "confidence"
            ])

            # Write data
            for img in images:
                # Copy image if requested
                if include_images:
                    src_path = self.base_dir / img.file_path
                    if src_path.exists():
                        dst_path = export_path / "images" / img.filename
                        shutil.copy(src_path, dst_path)

                # Write annotations
                anns = annotations_map.get(img.id, [])
                for ann in anns:
                    writer.writerow([
                        img.filename,
                        img.width or 640,
                        img.height or 640,
                        ann.class_id,
                        ann.class_name,
                        ann.x_center,
                        ann.y_center,
                        ann.width,
                        ann.height,
                        ann.confidence
                    ])

    @staticmethod
    def _create_zip(source_dir: Path, output_zip: Path):
        """Create a ZIP archive from a directory"""
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in source_dir.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(source_dir)
                    zipf.write(file, arcname)
