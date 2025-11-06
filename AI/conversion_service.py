"""
AI Model Conversion Service
Provides unified interface for converting YOLO models to different formats (ONNX, TensorRT)
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelConversionService:
    """Service for converting YOLO models to different formats"""
    
    def __init__(self):
        """Initialize the conversion service"""
        self.supported_formats = ["onnx", "tensorrt", "trt"]
        self.supported_precisions = ["fp32", "fp16", "int8"]
        
    def convert_model(
        self,
        model_path: Union[str, Path],
        target_format: str,
        output_dir: Optional[Union[str, Path]] = None,
        **conversion_options
    ) -> Dict[str, Any]:
        """
        Convert a YOLO model to the specified format
        
        Args:
            model_path: Path to source .pt model file
            target_format: Target format ("onnx", "tensorrt", "trt")
            output_dir: Directory to save converted model (optional)
            **conversion_options: Format-specific conversion options
            
        Returns:
            Dictionary with conversion results and metadata
        """
        try:
            # Validate inputs
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            if not model_path.suffix == '.pt':
                raise ValueError(f"Expected .pt file, got: {model_path.suffix}")
                
            target_format = target_format.lower()
            if target_format not in self.supported_formats:
                raise ValueError(f"Unsupported format: {target_format}. Supported: {self.supported_formats}")
            
            # Set output directory
            if output_dir is None:
                output_dir = model_path.parent / "converted"
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Starting conversion: {model_path} -> {target_format}")
            start_time = time.time()
            
            # Load YOLO model
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            
            # Perform conversion based on target format
            if target_format == "onnx":
                result = self._convert_to_onnx(model, output_dir, **conversion_options)
            elif target_format in ["tensorrt", "trt"]:
                result = self._convert_to_tensorrt(model, output_dir, **conversion_options)
            else:
                raise ValueError(f"Conversion to {target_format} not implemented")
            
            conversion_time = time.time() - start_time
            
            # Prepare response
            response = {
                "success": True,
                "source_model": str(model_path),
                "target_format": target_format,
                "output_path": result["output_path"],
                "conversion_time": conversion_time,
                "model_info": {
                    "source_size_mb": model_path.stat().st_size / (1024 * 1024),
                    "output_size_mb": Path(result["output_path"]).stat().st_size / (1024 * 1024),
                    "classes": len(model.names),
                    "class_names": list(model.names.values())
                },
                "conversion_options": conversion_options,
                "metadata": result.get("metadata", {}),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Conversion completed in {conversion_time:.2f}s")
            logger.info(f"   Output: {result['output_path']}")
            logger.info(f"   Size: {response['model_info']['source_size_mb']:.1f}MB -> {response['model_info']['output_size_mb']:.1f}MB")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Conversion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "source_model": str(model_path) if 'model_path' in locals() else None,
                "target_format": target_format if 'target_format' in locals() else None,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _convert_to_onnx(
        self,
        model,
        output_dir: Path,
        imgsz: int = 640,
        batch_size: int = 1,
        dynamic: bool = False,
        simplify: bool = True,
        opset: int = 17,
        **kwargs
    ) -> Dict[str, Any]:
        """Convert YOLO model to ONNX format"""
        
        logger.info(f"Converting to ONNX with options:")
        logger.info(f"  - Image size: {imgsz}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Dynamic shapes: {dynamic}")
        logger.info(f"  - Simplify: {simplify}")
        logger.info(f"  - ONNX opset: {opset}")
        
        # Export to ONNX
        export_path = model.export(
            format="onnx",
            imgsz=imgsz,
            batch=batch_size,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            **kwargs
        )
        
        # Move to output directory with proper naming
        onnx_filename = f"{model.model.yaml.get('name', 'model')}_{imgsz}_batch{batch_size}.onnx"
        if dynamic:
            onnx_filename = onnx_filename.replace('.onnx', '_dynamic.onnx')
            
        final_path = output_dir / onnx_filename
        Path(export_path).rename(final_path)
        
        return {
            "output_path": str(final_path),
            "metadata": {
                "imgsz": imgsz,
                "batch_size": batch_size,
                "dynamic": dynamic,
                "simplify": simplify,
                "opset": opset
            }
        }
    
    def _convert_to_tensorrt(
        self,
        model,
        output_dir: Path,
        imgsz: int = 640,
        batch_size: int = 1,
        precision: str = "fp16",
        workspace: int = 4,
        int8_calibration_data: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Convert YOLO model to TensorRT format"""
        
        # Validate precision
        if precision not in self.supported_precisions:
            raise ValueError(f"Unsupported precision: {precision}. Supported: {self.supported_precisions}")
        
        # Check for INT8 calibration data
        if precision == "int8" and not int8_calibration_data:
            logger.warning("INT8 precision requested but no calibration data provided. Using FP16 instead.")
            precision = "fp16"
        
        logger.info(f"Converting to TensorRT with options:")
        logger.info(f"  - Image size: {imgsz}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Precision: {precision}")
        logger.info(f"  - Workspace: {workspace}GB")
        
        # Prepare export arguments
        export_args = {
            "format": "engine",
            "imgsz": imgsz,
            "batch": batch_size,
            "workspace": workspace,
            **kwargs
        }
        
        # Add precision-specific arguments
        if precision == "fp16":
            export_args["half"] = True
        elif precision == "int8":
            export_args["int8"] = True
            if int8_calibration_data:
                export_args["data"] = int8_calibration_data
        
        # Export to TensorRT
        export_path = model.export(**export_args)
        
        # Move to output directory with proper naming
        engine_filename = f"{model.model.yaml.get('name', 'model')}_{imgsz}_batch{batch_size}_{precision}.engine"
        final_path = output_dir / engine_filename
        Path(export_path).rename(final_path)
        
        return {
            "output_path": str(final_path),
            "metadata": {
                "imgsz": imgsz,
                "batch_size": batch_size,
                "precision": precision,
                "workspace": workspace,
                "int8_calibration": int8_calibration_data is not None
            }
        }
    
    def get_conversion_info(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about a model for conversion planning"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            
            return {
                "model_path": str(model_path),
                "model_size_mb": model_path.stat().st_size / (1024 * 1024),
                "classes": len(model.names),
                "class_names": list(model.names.values()),
                "supported_formats": self.supported_formats,
                "supported_precisions": self.supported_precisions,
                "recommended_settings": {
                    "onnx": {
                        "imgsz": 640,
                        "batch_size": 1,
                        "dynamic": False,
                        "simplify": True,
                        "opset": 17
                    },
                    "tensorrt": {
                        "imgsz": 640,
                        "batch_size": 1,
                        "precision": "fp16",
                        "workspace": 4
                    }
                }
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "model_path": str(model_path) if 'model_path' in locals() else None
            }
    
    def batch_convert(
        self,
        model_paths: List[Union[str, Path]],
        target_format: str,
        output_dir: Optional[Union[str, Path]] = None,
        **conversion_options
    ) -> List[Dict[str, Any]]:
        """Convert multiple models to the specified format"""
        results = []
        
        for model_path in model_paths:
            logger.info(f"Converting {model_path} ({len(results)+1}/{len(model_paths)})")
            result = self.convert_model(
                model_path=model_path,
                target_format=target_format,
                output_dir=output_dir,
                **conversion_options
            )
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r.get("success", False))
        logger.info(f"Batch conversion completed: {successful}/{len(results)} successful")
        
        return results


# Convenience functions for direct usage
def convert_to_onnx(
    model_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    **options
) -> Dict[str, Any]:
    """Convert YOLO model to ONNX format"""
    service = ModelConversionService()
    return service.convert_model(model_path, "onnx", output_dir, **options)


def convert_to_tensorrt(
    model_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    **options
) -> Dict[str, Any]:
    """Convert YOLO model to TensorRT format"""
    service = ModelConversionService()
    return service.convert_model(model_path, "tensorrt", output_dir, **options)


def get_model_info(model_path: Union[str, Path]) -> Dict[str, Any]:
    """Get model information for conversion planning"""
    service = ModelConversionService()
    return service.get_conversion_info(model_path)
