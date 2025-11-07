"""
Model Conversion Service
Handles conversion of PyTorch models to ONNX and TensorRT formats
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Union, List, Optional

logger = logging.getLogger(__name__)

class ModelConversionService:
    """Service for converting PyTorch models to different formats"""
    
    def __init__(self):
        """Initialize the conversion service"""
        self.supported_formats = ["onnx", "tensorrt", "trt"]
        self.supported_precisions = ["fp32", "fp16", "int8"]
        
        logger.info("Model Conversion Service initialized")
        logger.info(f"   Supported formats: {', '.join(self.supported_formats)}")
        logger.info(f"   Supported precisions: {', '.join(self.supported_precisions)}")
    
    def convert_model(self, 
                     model_path: Union[str, Path], 
                     target_format: str,
                     output_dir: Optional[Union[str, Path]] = None,
                     **conversion_options) -> Dict[str, Any]:
        """
        Convert a model to the specified format
        
        Args:
            model_path: Path to the source .pt model file
            target_format: Target format ('onnx', 'tensorrt', 'trt')
            output_dir: Output directory (optional)
            **conversion_options: Format-specific options
            
        Returns:
            Dictionary with conversion results
        """
        start_time = time.time()
        
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
            
            logger.info(f"Converting {model_path.name} to {target_format.upper()}")
            logger.info(f"   Output directory: {output_dir}")
            
            # Load model
            try:
                from ultralytics import YOLO
                model = YOLO(str(model_path))
                logger.info(f" Model loaded successfully: {model_path.name}")
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}")
            
            # Perform conversion based on target format
            if target_format == "onnx":
                result = self._convert_to_onnx(model, output_dir, **conversion_options)
            elif target_format in ["tensorrt", "trt"]:
                result = self._convert_to_tensorrt(model, output_dir, **conversion_options)
            else:
                raise ValueError(f"Conversion to {target_format} not implemented")
            
            # Add timing and common info
            conversion_time = time.time() - start_time
            result.update({
                "success": True,
                "conversion_time": conversion_time,
                "source_model": str(model_path),
                "target_format": target_format,
                "output_directory": str(output_dir)
            })
            
            logger.info(f" Conversion completed in {conversion_time:.2f}s")
            logger.info(f"   Output: {result.get('output_path', 'N/A')}")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Conversion failed: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "conversion_time": time.time() - start_time,
                "source_model": str(model_path),
                "target_format": target_format
            }
    
    def _convert_to_onnx(self, model: Any, output_dir: Path, **kwargs) -> Dict[str, Any]:
        """Convert model to ONNX format"""
        
        # Extract ONNX-specific options with defaults
        imgsz = kwargs.get('imgsz', 640)
        batch_size = kwargs.get('batch_size', 1)
        dynamic = kwargs.get('dynamic', False)
        simplify = kwargs.get('simplify', True)
        opset = kwargs.get('opset', 17)
        
        logger.info(f" ONNX conversion options:")
        logger.info(f"   Image size: {imgsz}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Dynamic shapes: {dynamic}")
        logger.info(f"   Simplify: {simplify}")
        logger.info(f"   Opset version: {opset}")
        
        try:
            # Prepare export arguments
            export_args = {
                'format': 'onnx',
                'imgsz': imgsz,
                'dynamic': dynamic,
                'simplify': simplify,
                'opset': opset
            }
            
            # Add batch size if not dynamic
            if not dynamic:
                export_args['batch'] = batch_size
            
            # Export model
            logger.info("Starting ONNX export...")
            export_path = model.export(**export_args)
            
            if not export_path or not Path(export_path).exists():
                raise RuntimeError("ONNX export failed - no output file generated")
            
            # Move to output directory if needed
            export_path = Path(export_path)
            if export_path.parent != output_dir:
                final_path = output_dir / export_path.name
                export_path.rename(final_path)
                export_path = final_path
            
            # Get model info
            model_info = self._get_model_info(export_path)
            
            return {
                "output_path": str(export_path),
                "format": "onnx",
                "model_info": model_info,
                "conversion_options": {
                    "imgsz": imgsz,
                    "batch_size": batch_size,
                    "dynamic": dynamic,
                    "simplify": simplify,
                    "opset": opset
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"ONNX conversion failed: {e}")
    
    def _convert_to_tensorrt(self, model: Any, output_dir: Path, **kwargs) -> Dict[str, Any]:
        """Convert model to TensorRT format"""
        
        # Extract TensorRT-specific options with defaults
        imgsz = kwargs.get('imgsz', 640)
        batch_size = kwargs.get('batch_size', 1)
        precision = kwargs.get('precision', 'fp16')
        workspace = kwargs.get('workspace', 4)
        int8_calibration_data = kwargs.get('int8_calibration_data', None)
        
        if precision not in self.supported_precisions:
            raise ValueError(f"Unsupported precision: {precision}. Supported: {self.supported_precisions}")
        
        logger.info(f" TensorRT conversion options:")
        logger.info(f"   Image size: {imgsz}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Precision: {precision}")
        logger.info(f"   Workspace: {workspace}GB")
        
        try:
            # Check TensorRT availability
            try:
                import tensorrt as trt
                logger.info(f" TensorRT version: {trt.__version__}")
            except ImportError:
                raise RuntimeError("TensorRT not installed. Install with: pip install tensorrt")
            
            # Prepare export arguments
            export_args = {
                'format': 'engine',  # TensorRT engine format
                'imgsz': imgsz,
                'batch': batch_size,
                'workspace': workspace,
                'verbose': True
            }
            
            # Set precision-specific options
            if precision == 'fp16':
                export_args['half'] = True
            elif precision == 'int8':
                export_args['int8'] = True
                if int8_calibration_data:
                    export_args['data'] = int8_calibration_data
            # fp32 is default, no additional flags needed
            
            # Export model
            logger.info("Starting TensorRT export...")
            export_path = model.export(**export_args)
            
            if not export_path or not Path(export_path).exists():
                raise RuntimeError("TensorRT export failed - no output file generated")
            
            # Move to output directory if needed
            export_path = Path(export_path)
            if export_path.parent != output_dir:
                final_path = output_dir / export_path.name
                export_path.rename(final_path)
                export_path = final_path
            
            # Get model info
            model_info = self._get_model_info(export_path)
            
            return {
                "output_path": str(export_path),
                "format": "tensorrt",
                "model_info": model_info,
                "conversion_options": {
                    "imgsz": imgsz,
                    "batch_size": batch_size,
                    "precision": precision,
                    "workspace": workspace
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"TensorRT conversion failed: {e}")
    
    def _get_model_info(self, model_path: Path) -> Dict[str, Any]:
        """Get information about the converted model"""
        try:
            file_size = model_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            return {
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size_mb, 2),
                "file_name": model_path.name,
                "file_extension": model_path.suffix
            }
        except Exception as e:
            logger.warning(f"Could not get model info: {e}")
            return {}
    
    def get_conversion_info(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a model for conversion planning
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Dictionary with model information and conversion recommendations
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model to get info
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            
            # Get basic info
            file_size = model_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            # Try to get model details
            model_info = {
                "model_path": str(model_path),
                "model_size_mb": round(file_size_mb, 2),
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
            
            # Try to get class information
            try:
                if hasattr(model, 'names') and model.names:
                    model_info["classes"] = len(model.names)
                    model_info["class_names"] = list(model.names.values())
            except:
                pass
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {
                "error": str(e),
                "supported_formats": self.supported_formats,
                "supported_precisions": self.supported_precisions
            }


# Convenience functions for direct usage
def convert_to_onnx(model_path: Union[str, Path], 
                   output_dir: Optional[Union[str, Path]] = None,
                   **options) -> Dict[str, Any]:
    """Convert model to ONNX format"""
    service = ModelConversionService()
    return service.convert_model(model_path, "onnx", output_dir, **options)

def convert_to_tensorrt(model_path: Union[str, Path], 
                       output_dir: Optional[Union[str, Path]] = None,
                       **options) -> Dict[str, Any]:
    """Convert model to TensorRT format"""
    service = ModelConversionService()
    return service.convert_model(model_path, "tensorrt", output_dir, **options)

def get_model_info(model_path: Union[str, Path]) -> Dict[str, Any]:
    """Get model information"""
    service = ModelConversionService()
    return service.get_conversion_info(model_path)