#!/usr/bin/env python3
"""
ONNX YOLO Inference

Runs inference using ONNX YOLO model with proper preprocessing and postprocessing.
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ONNXYOLOInference:
    """ONNX YOLO inference with proper preprocessing."""
    
    COCO_CLASSES = {
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
    
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (0, 128, 128), (128, 128, 0), (255, 165, 0)
    ]
    
    def __init__(self, onnx_path: str):
        """Initialize ONNX inference session."""
        self.onnx_path = Path(onnx_path)
        self.session = ort.InferenceSession(str(self.onnx_path))
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Extract input size
        if len(self.input_shape) >= 4:
            self.input_size = (self.input_shape[2], self.input_shape[3])
        else:
            self.input_size = (640, 640)
        
        logger.info(f"✓ Loaded ONNX model: {self.onnx_path}")
        logger.info(f"✓ Input size: {self.input_size}")
    
    def predict(self, image_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45, save_output: bool = True):
        """
        Run inference on image.
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            save_output: Whether to save annotated image
            
        Returns:
            List of detection dictionaries
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_shape = image.shape[:2]
        
        # Preprocess
        input_tensor = self._preprocess_image(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        raw_output = outputs[0]
        
        # Postprocess
        detections = self._postprocess_output(raw_output, original_shape, conf_threshold, iou_threshold)
        
        # Save output
        if save_output and detections:
            self._save_detections(image_path, image, detections)
        
        return detections
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX inference."""
        # Resize
        resized = cv2.resize(image, self.input_size)
        
        # BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # HWC to CHW
        chw_tensor = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batch_tensor = np.expand_dims(chw_tensor, axis=0)
        
        return batch_tensor
    
    def _postprocess_output(self, output: np.ndarray, original_shape: tuple, conf_threshold: float, iou_threshold: float):
        """Postprocess YOLO output."""
        # Remove batch dimension
        if len(output.shape) == 3:
            output = output[0]
        
        detections = []
        
        for detection in output:
            # Extract components
            bbox = detection[:4]  # [x_center, y_center, width, height]
            obj_conf = detection[4]
            class_scores = detection[5:]
            
            # Get best class
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            
            # Calculate final confidence
            final_conf = obj_conf * class_conf
            
            # Filter by confidence
            if final_conf < conf_threshold:
                continue
            
            # Convert center to corner format
            x_center, y_center, width, height = bbox
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Scale to original image size
            scale_x = original_shape[1] / self.input_size[1]
            scale_y = original_shape[0] / self.input_size[0]
            
            x1 *= scale_x
            y1 *= scale_y
            x2 *= scale_x
            y2 *= scale_y
            
            detection_dict = {
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(final_conf),
                'class_id': int(class_id),
                'class_name': self.COCO_CLASSES.get(class_id, f'class_{class_id}')
            }
            
            detections.append(detection_dict)
        
        # Apply NMS
        if detections:
            detections = self._apply_nms(detections, iou_threshold)
        
        return detections
    
    def _apply_nms(self, detections: list, iou_threshold: float) -> list:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Remove detections with high IoU
            remaining = []
            for det in detections:
                if self._calculate_iou(current['bbox'], det['bbox']) < iou_threshold:
                    remaining.append(det)
            
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, box1: list, box2: list) -> float:
        """Calculate Intersection over Union."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _save_detections(self, image_path: Path, image: np.ndarray, detections: list):
        """Save image with detections."""
        annotated = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            class_id = det['class_id']
            
            color = self.COLORS[class_id % len(self.COLORS)]
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
        
        output_path = image_path.parent / f"{image_path.stem}_onnx{image_path.suffix}"
        cv2.imwrite(str(output_path), annotated)
        logger.info(f"Saved ONNX detections: {output_path}")
    
    def print_results(self, detections: list, image_path: str):
        """Print results to console."""
        print(f"\n{'='*60}")
        print(f"ONNX YOLO Inference Results")
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
    
    parser = argparse.ArgumentParser(description="ONNX YOLO inference")
    parser.add_argument("--model", type=str, default="models/yolo11n.onnx",
                       help="Path to ONNX model")
    parser.add_argument("--image", type=str, default="data/images/image_1.png",
                       help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="IoU threshold for NMS")
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        return 1
    
    if not Path(args.image).exists():
        logger.error(f"Image not found: {args.image}")
        return 1
    
    try:
        # Initialize inference
        inference = ONNXYOLOInference(args.model)
        
        # Run inference
        detections = inference.predict(args.image, args.conf, args.iou)
        
        # Print results
        inference.print_results(detections, args.image)
        
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
