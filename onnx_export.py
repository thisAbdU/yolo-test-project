#!/usr/bin/env python3
"""
ONNX Export for YOLO

Converts PyTorch YOLO model to ONNX format with validation.
Due to compatibility issues, creates a working ONNX model.
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YOLOONNXExporter:
    """YOLO to ONNX exporter."""
    
    def __init__(self, model_path: str):
        """Initialize with model path."""
        self.model_path = Path(model_path)
    
    def export_to_onnx(self, output_path: str, input_size: tuple = (640, 640)):
        """
        Export YOLO model to ONNX format.
        
        Args:
            output_path: Output ONNX file path
            input_size: Input image size (height, width)
            
        Returns:
            Path to exported ONNX model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating YOLO ONNX model: {output_path}")
        
        try:
            # Try to load real YOLO model first
            try:
                from ultralytics import YOLO
                yolo_model = YOLO(str(self.model_path))
                logger.info("✓ Loaded real YOLO model")
                
                # Export using Ultralytics
                yolo_model.export(format="onnx", imgsz=input_size, dynamic=False)
                
                # Move to desired location
                default_onnx = self.model_path.parent / f"{self.model_path.stem}.onnx"
                if default_onnx.exists():
                    default_onnx.rename(output_path)
                
                logger.info(f"✓ Real ONNX export completed: {output_path}")
                
            except Exception as e:
                logger.warning(f"Real export failed: {e}")
                logger.info("Creating working ONNX model...")
                
                # Create working ONNX model
                self._create_working_onnx(output_path, input_size)
            
            # Validate the model
            self._validate_onnx_model(output_path, input_size)
            
            return output_path
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
    
    def _create_working_onnx(self, output_path: Path, input_size: tuple):
        """Create ONNX model from real YOLO PyTorch model."""
        logger.info("Creating ONNX model from real YOLO PyTorch model...")
        
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(str(self.model_path))
            
            # Export using Ultralytics built-in export functionality
            yolo_model.export(
                format="onnx",
                imgsz=input_size,
                dynamic=False,
                simplify=True
            )
            
            # Move to desired location
            default_onnx = self.model_path.parent / f"{self.model_path.stem}.onnx"
            if default_onnx.exists():
                default_onnx.rename(output_path)
            
            logger.info(f"✓ Real YOLO ONNX model created: {output_path}")
            
        except Exception as e:
            logger.error(f"Real ONNX export failed: {e}")
            raise
    
    def _validate_onnx_model(self, onnx_path: Path, input_size: tuple):
        """Validate ONNX model."""
        logger.info("Validating ONNX model...")
        
        # Load and check model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("✓ ONNX model validation passed")
        
        # Test with ONNX Runtime
        session = ort.InferenceSession(str(onnx_path))
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        logger.info(f"✓ Input: {input_info.name}, shape: {input_info.shape}")
        logger.info(f"✓ Output: {output_info.name}, shape: {output_info.shape}")
        
        # Test inference
        test_input = np.random.randn(1, 3, *input_size).astype(np.float32)
        outputs = session.run(None, {input_info.name: test_input})
        logger.info(f"✓ ONNX inference test passed")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export YOLO to ONNX")
    parser.add_argument("--model", type=str, default="data/images/yolo11n.pt",
                       help="Path to PyTorch YOLO model")
    parser.add_argument("--output", type=str, default="models/yolo11n.onnx",
                       help="Output ONNX file path")
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 640],
                       help="Input size [height width]")
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        return 1
    
    try:
        exporter = YOLOONNXExporter(args.model)
        onnx_path = exporter.export_to_onnx(args.output, tuple(args.input_size))
        
        logger.info(f"🎉 ONNX export completed: {onnx_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
