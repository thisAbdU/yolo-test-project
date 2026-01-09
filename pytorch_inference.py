#!/usr/bin/env python3
"""
PyTorch YOLO Inference

Loads yolo11n.pt model and runs inference on images.
Due to PyTorch/torchvision compatibility issues, this uses a fallback approach.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PyTorchYOLOInference:
    """PyTorch YOLO inference with fallback for compatibility issues."""
    
    def __init__(self, model_path: str):
        """Initialize with model path."""
        self.model_path = Path(model_path)
        self.model = None
        self.class_names = self._get_coco_classes()
        self._load_model()
    
    def _get_coco_classes(self):
        """COCO class names."""
        return {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
            44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
            49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
            54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
            59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        }
    
    def _load_model(self):
        """Load YOLO model with compatibility handling."""
        try:
            # Try to load with Ultralytics first
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            logger.info("✓ Loaded YOLO model with Ultralytics")
        except Exception as e:
            logger.warning(f"Ultralytics loading failed: {e}")
            logger.info("Using mock inference for demonstration")
            self.model = None
    
    def predict(self, image_path: str, conf_threshold: float = 0.25, save_output: bool = True):
        """
        Run inference on image.
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold
            save_output: Whether to save annotated image
            
        Returns:
            List of detection dictionaries
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if self.model is None:
            # Mock inference for demonstration
            return self._mock_inference(image_path, conf_threshold, save_output)
        
        try:
            # Real YOLO inference
            results = self.model(str(image_path), conf=conf_threshold)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        detection = {
                            'bbox': box.xyxy[0].cpu().numpy().tolist(),
                            'confidence': float(box.conf[0].cpu().numpy()),
                            'class_id': int(box.cls[0].cpu().numpy()),
                            'class_name': self.model.names[int(box.cls[0].cpu().numpy())]
                        }
                        detections.append(detection)
            
            # Save output
            if save_output and detections:
                self._save_detections(image_path, detections)
            
            return detections
            
        except Exception as e:
            logger.error(f"Real inference failed: {e}")
            return self._fallback_inference(image_path, conf_threshold, save_output)
    
    def _fallback_inference(self, image_path: Path, conf_threshold: float, save_output: bool):
        """Fallback inference when real model fails."""
        logger.error("Real PyTorch inference failed - no fallback available")
        logger.info("Please check model path and dependencies")
        return []
    
    def _save_detections(self, image_path: Path, detections: list, suffix: str = "_detections"):
        """Save image with detections."""
        image = cv2.imread(str(image_path))
        annotated = image.copy()
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            class_id = det['class_id']
            
            color = colors[class_id % len(colors)]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        output_path = image_path.parent / f"{image_path.stem}{suffix}{image_path.suffix}"
        cv2.imwrite(str(output_path), annotated)
        logger.info(f"Saved PyTorch detections: {output_path}")
    
    def print_results(self, detections: list, image_path: str):
        """Print results to console."""
        print(f"\n{'='*60}")
        print(f"PyTorch YOLO Inference Results")
        print(f"Image: {image_path}")
        print(f"Number of detections: {len(detections)}")
        print(f"{'='*60}")
        
        if not detections:
            print("No objects detected.")
            return
        
        print(f"{'ID':<4} {'Class':<15} {'Confidence':<12} {'Bounding Box':<25}")
        print(f"{'-'*60}")
        
        for det in detections:
            bbox = det['bbox']
            bbox_str = f"({int(bbox[0])},{int(bbox[1])})-({int(bbox[2])},{int(bbox[3])})"
            print(f"{det['class_id']:<4} {det['class_name']:<15} {det['confidence']:<12.3f} {bbox_str:<25}")
        
        print(f"{'='*60}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PyTorch YOLO inference")
    parser.add_argument("--model", type=str, default="data/images/yolo11n.pt",
                       help="Path to PyTorch YOLO model")
    parser.add_argument("--image", type=str, default="data/images/image_1.png",
                       help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        return 1
    
    if not Path(args.image).exists():
        logger.error(f"Image not found: {args.image}")
        return 1
    
    try:
        # Initialize inference
        inference = PyTorchYOLOInference(args.model)
        
        # Run inference
        detections = inference.predict(args.image, args.conf)
        
        # Print results
        inference.print_results(detections, args.image)
        
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
