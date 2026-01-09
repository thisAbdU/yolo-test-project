#!/usr/bin/env python3
"""
IoU Comparison between PyTorch and ONNX Models

Compares detection results and calculates IoU metrics.
"""

import numpy as np
import cv2
from pathlib import Path
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IoUComparison:
    """Compare PyTorch and ONNX YOLO results."""
    
    def __init__(self, iou_threshold: float = 0.5):
        """Initialize comparison with IoU threshold."""
        self.iou_threshold = iou_threshold
    
    def calculate_iou(self, box1: list, box2: list) -> float:
        """Calculate IoU between two bounding boxes."""
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
    
    def match_detections(self, pytorch_detections: list, onnx_detections: list):
        """Match detections between PyTorch and ONNX results."""
        matches = []
        unmatched_pytorch = list(range(len(pytorch_detections)))
        unmatched_onnx = list(range(len(onnx_detections)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(pytorch_detections), len(onnx_detections)))
        for i, det1 in enumerate(pytorch_detections):
            for j, det2 in enumerate(onnx_detections):
                # Only match same class
                if det1['class_id'] == det2['class_id']:
                    iou_matrix[i, j] = self.calculate_iou(det1['bbox'], det2['bbox'])
        
        # Find best matches
        used_pytorch = set()
        used_onnx = set()
        
        # Sort by IoU
        matches_list = []
        for i in range(len(pytorch_detections)):
            for j in range(len(onnx_detections)):
                if iou_matrix[i, j] > 0:
                    matches_list.append((i, j, iou_matrix[i, j]))
        
        matches_list.sort(key=lambda x: x[2], reverse=True)
        
        for i, j, iou in matches_list:
            if i not in used_pytorch and j not in used_onnx and iou >= self.iou_threshold:
                matches.append((i, j, iou))
                used_pytorch.add(i)
                used_onnx.add(j)
        
        unmatched_pytorch = [i for i in range(len(pytorch_detections)) if i not in used_pytorch]
        unmatched_onnx = [j for j in range(len(onnx_detections)) if j not in used_onnx]
        
        return matches, unmatched_pytorch, unmatched_onnx
    
    def compare_single_image(self, pytorch_detections: list, onnx_detections: list):
        """Compare results for a single image."""
        matches, unmatched_pytorch, unmatched_onnx = self.match_detections(
            pytorch_detections, onnx_detections
        )
        
        total_pytorch = len(pytorch_detections)
        total_onnx = len(onnx_detections)
        matched = len(matches)
        
        # Calculate metrics
        precision = matched / total_onnx if total_onnx > 0 else 0.0
        recall = matched / total_pytorch if total_pytorch > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_iou = np.mean([match[2] for match in matches]) if matches else 0.0
        
        # Calculate confidence differences
        conf_diffs = []
        for i, j, iou in matches:
            conf_diff = abs(pytorch_detections[i]['confidence'] - onnx_detections[j]['confidence'])
            conf_diffs.append(conf_diff)
        avg_conf_diff = np.mean(conf_diffs) if conf_diffs else 0.0
        
        comparison = {
            'pytorch_detections': total_pytorch,
            'onnx_detections': total_onnx,
            'matched_detections': matched,
            'unmatched_pytorch': len(unmatched_pytorch),
            'unmatched_onnx': len(unmatched_onnx),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_iou': avg_iou,
            'avg_confidence_diff': avg_conf_diff,
            'matches': matches
        }
        
        return comparison
    
    def create_comparison_image(self, image_path: str, pytorch_detections: list, onnx_detections: list):
        """Create side-by-side comparison image."""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Draw PyTorch detections
        pytorch_image = self._draw_detections(image, pytorch_detections, (255, 0, 0))
        
        # Draw ONNX detections
        onnx_image = self._draw_detections(image, onnx_detections, (0, 255, 0))
        
        # Create side-by-side comparison
        h, w = image.shape[:2]
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = pytorch_image
        comparison[:, w:] = onnx_image
        
        # Add labels
        cv2.putText(comparison, "PyTorch", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "ONNX", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add statistics
        cv2.putText(comparison, f"Detections: {len(pytorch_detections)}", (10, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(comparison, f"Detections: {len(onnx_detections)}", (w + 10, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Draw matching lines
        matches, _, _ = self.match_detections(pytorch_detections, onnx_detections)
        for i, j, iou in matches:
            pytorch_bbox = pytorch_detections[i]['bbox']
            onnx_bbox = onnx_detections[j]['bbox']
            
            # Calculate centers
            pytorch_center = (int((pytorch_bbox[0] + pytorch_bbox[2]) / 2), 
                               int((pytorch_bbox[1] + pytorch_bbox[3]) / 2))
            onnx_center = (int((onnx_bbox[0] + onnx_bbox[2]) / 2) + w, 
                           int((onnx_bbox[1] + onnx_bbox[3]) / 2))
            
            # Draw line
            cv2.line(comparison, pytorch_center, onnx_center, (255, 255, 0), 1)
            
            # Add IoU text
            mid_point = ((pytorch_center[0] + onnx_center[0]) // 2, 
                        (pytorch_center[1] + onnx_center[1]) // 2)
            cv2.putText(comparison, f"{iou:.2f}", mid_point, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return comparison
    
    def _draw_detections(self, image: np.ndarray, detections: list, color: tuple):
        """Draw detections on image."""
        annotated = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
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
        
        return annotated
    
    def print_comparison_report(self, comparison: dict, image_path: str):
        """Print comparison report."""
        print(f"\n{'='*60}")
        print(f"IoU Comparison Report")
        print(f"Image: {image_path}")
        print(f"{'='*60}")
        print(f"PyTorch Detections: {comparison['pytorch_detections']}")
        print(f"ONNX Detections: {comparison['onnx_detections']}")
        print(f"Matched Detections: {comparison['matched_detections']}")
        print(f"Unmatched PyTorch: {comparison['unmatched_pytorch']}")
        print(f"Unmatched ONNX: {comparison['unmatched_onnx']}")
        print(f"")
        print(f"Metrics:")
        print(f"  Precision: {comparison['precision']:.4f}")
        print(f"  Recall: {comparison['recall']:.4f}")
        print(f"  F1 Score: {comparison['f1_score']:.4f}")
        print(f"  Average IoU: {comparison['avg_iou']:.4f}")
        print(f"  Avg Confidence Diff: {comparison['avg_confidence_diff']:.4f}")
        print(f"{'='*60}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="IoU comparison")
    parser.add_argument("--image", type=str, default="data/images/image_1.png",
                       help="Path to input image")
    parser.add_argument("--pytorch-results", type=str, default=None,
                       help="Path to PyTorch results (JSON)")
    parser.add_argument("--onnx-results", type=str, default=None,
                       help="Path to ONNX results (JSON)")
    parser.add_argument("--onnx-model", type=str, default="data/outputs/yolo11n_real.onnx",
                       help="Path to ONNX model")
    parser.add_argument("--iou", type=float, default=0.5,
                       help="IoU threshold for matching")
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        logger.error(f"Image not found: {args.image}")
        return 1
    
    try:
        # Load PyTorch results
        if args.pytorch_results and Path(args.pytorch_results).exists():
            import json
            with open(args.pytorch_results) as f:
                pytorch_detections = json.load(f)
        else:
            # Run PyTorch inference
            from pytorch_inference import PyTorchYOLOInference
            pytorch_inference = PyTorchYOLOInference("data/images/yolo11n.pt")
            pytorch_detections = pytorch_inference.predict(args.image)
        
        # Load ONNX results
        if args.onnx_results and Path(args.onnx_results).exists():
            import json
            with open(args.onnx_results) as f:
                onnx_detections = json.load(f)
        else:
            # Run ONNX inference
            from onnx_inference import ONNXYOLOInference
            onnx_inference = ONNXYOLOInference(args.onnx_model)
            onnx_detections = onnx_inference.predict(args.image, save_output=False)
        
        # Compare results
        comparison = IoUComparison(args.iou)
        results = comparison.compare_single_image(pytorch_detections, onnx_detections)
        
        # Print report
        comparison.print_comparison_report(results, args.image)
        
        # Create comparison image
        comparison_image = comparison.create_comparison_image(args.image, pytorch_detections, onnx_detections)
        if comparison_image is not None:
            output_path = Path(args.image).parent / f"{Path(args.image).stem}_comparison{Path(args.image).suffix}"
            cv2.imwrite(str(output_path), comparison_image)
            logger.info(f"Comparison image saved: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
