#!/usr/bin/env python3
"""
Complete YOLO PyTorch to ONNX Pipeline

Runs the complete pipeline: PyTorch inference → ONNX export → ONNX inference → IoU comparison.
"""

import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_pytorch_inference(model_path: str, image_path: str):
    """Run PyTorch YOLO inference."""
    logger.info("🔥 Running PyTorch YOLO inference...")
    
    from pytorch_inference import PyTorchYOLOInference
    
    try:
        inference = PyTorchYOLOInference(model_path)
        detections = inference.predict(image_path)
        inference.print_results(detections, image_path)
        logger.info("✅ PyTorch inference completed")
        return detections
    except Exception as e:
        logger.error(f"❌ PyTorch inference failed: {e}")
        return None


def export_to_onnx(model_path: str, onnx_path: str):
    """Export PyTorch model to ONNX."""
    logger.info("🔄 Converting PyTorch model to ONNX...")
    
    from onnx_export import YOLOONNXExporter
    
    try:
        exporter = YOLOONNXExporter(model_path)
        onnx_path = exporter.export_to_onnx(onnx_path)
        logger.info("✅ ONNX export completed")
        return onnx_path
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")
        return None


def run_onnx_inference(onnx_path: str, image_path: str):
    """Run ONNX YOLO inference."""
    logger.info("⚡ Running ONNX YOLO inference...")
    
    from onnx_inference import ONNXYOLOInference
    
    try:
        inference = ONNXYOLOInference(onnx_path)
        detections = inference.predict(image_path)
        inference.print_results(detections, image_path)
        logger.info("✅ ONNX inference completed")
        return detections
    except Exception as e:
        logger.error(f"❌ ONNX inference failed: {e}")
        return None


def compare_results(pytorch_detections, onnx_detections, image_path: str):
    """Compare PyTorch and ONNX results."""
    logger.info("🔍 Comparing PyTorch and ONNX results...")
    
    from iou_comparison import IoUComparison
    
    try:
        comparison = IoUComparison()
        results = comparison.compare_single_image(pytorch_detections, onnx_detections)
        comparison.print_comparison_report(results, image_path)
        
        # Create comparison image
        comparison_image = comparison.create_comparison_image(image_path, pytorch_detections, onnx_detections)
        if comparison_image is not None:
            output_path = Path(image_path).parent / f"{Path(image_path).stem}_comparison{Path(image_path).suffix}"
            import cv2
            cv2.imwrite(str(output_path), comparison_image)
            logger.info(f"✅ Comparison image saved: {output_path}")
        
        logger.info("✅ Comparison completed")
        return results
    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")
        return None


def main():
    """Main pipeline function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete YOLO PyTorch to ONNX pipeline")
    parser.add_argument("--model", type=str, default="data/images/yolo11n.pt",
                       help="Path to PyTorch YOLO model")
    parser.add_argument("--image", type=str, default="data/images/image_1.png",
                       help="Path to input image")
    parser.add_argument("--onnx", type=str, default="models/yolo11n.onnx",
                       help="Path to output ONNX model")
    parser.add_argument("--skip-pytorch", action="store_true",
                       help="Skip PyTorch inference")
    parser.add_argument("--skip-export", action="store_true",
                       help="Skip ONNX export")
    parser.add_argument("--skip-onnx", action="store_true",
                       help="Skip ONNX inference")
    parser.add_argument("--skip-comparison", action="store_true",
                       help="Skip IoU comparison")
    
    args = parser.parse_args()
    
    # Check inputs
    if not Path(args.model).exists():
        logger.error(f"❌ Model not found: {args.model}")
        return 1
    
    if not Path(args.image).exists():
        logger.error(f"❌ Image not found: {args.image}")
        return 1
    
    logger.info("🚀 Starting YOLO PyTorch to ONNX Pipeline")
    logger.info(f"📁 Model: {args.model}")
    logger.info(f"🖼️  Image: {args.image}")
    logger.info(f"📄 ONNX: {args.onnx}")
    
    pytorch_detections = None
    onnx_detections = None
    
    # Step 1: PyTorch inference
    if not args.skip_pytorch:
        pytorch_detections = run_pytorch_inference(args.model, args.image)
    
    # Step 2: ONNX export
    if not args.skip_export:
        onnx_path = export_to_onnx(args.model, args.onnx)
        if onnx_path is None:
            logger.error("❌ Pipeline stopped: ONNX export failed")
            return 1
    
    # Step 3: ONNX inference
    if not args.skip_onnx:
        if not Path(args.onnx).exists():
            logger.error(f"❌ ONNX model not found: {args.onnx}")
            return 1
        
        onnx_detections = run_onnx_inference(args.onnx, args.image)
    
    # Step 4: IoU comparison
    if not args.skip_comparison and pytorch_detections and onnx_detections:
        compare_results(pytorch_detections, onnx_detections, args.image)
    
    logger.info("🎉 Pipeline completed successfully!")
    logger.info("\n📊 Results Summary:")
    logger.info(f"   PyTorch detections: {len(pytorch_detections) if pytorch_detections else 0}")
    logger.info(f"   ONNX detections: {len(onnx_detections) if onnx_detections else 0}")
    
    return 0


if __name__ == "__main__":
    exit(main())
