#!/usr/bin/env python3
"""
YOLOv8 object detection using Hailo-8 with GStreamer pipeline
Compatible with Raspberry Pi camera
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import argparse
from datetime import datetime

# Import Hailo pipeline infrastructure
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# COCO class names for YOLOv8
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Color palette for different classes (BGR format)
CLASS_COLORS = {
    'person': (0, 255, 0),      # Green
    'car': (255, 0, 0),          # Blue
    'bicycle': (0, 255, 255),    # Yellow
    'motorcycle': (255, 0, 255), # Magenta
    'bus': (128, 0, 128),        # Purple
    'truck': (255, 128, 0),      # Orange
    'dog': (0, 128, 255),        # Light Blue
    'cat': (128, 255, 0),        # Light Green
    'chair': (128, 128, 0),      # Olive
    'couch': (0, 128, 128),      # Teal
    # Default color for other classes
    'default': (255, 255, 255)   # White
}

class YOLOv8DetectionApp(app_callback_class):
    """
    Custom class for YOLOv8 detection tracking and visualization
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.frame_count = 0
        self.detection_stats = {}
        self.total_detections = 0
        self.fps_start_time = None
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        # Initialize stats for all classes
        for class_name in COCO_CLASSES:
            self.detection_stats[class_name] = 0
            
    def update_fps(self):
        """Calculate and update FPS"""
        import time
        current_time = time.time()
        
        if self.fps_start_time is None:
            self.fps_start_time = current_time
            self.fps_frame_count = 0
        else:
            self.fps_frame_count += 1
            elapsed = current_time - self.fps_start_time
            
            if elapsed > 1.0:  # Update FPS every second
                self.current_fps = self.fps_frame_count / elapsed
                self.fps_start_time = current_time
                self.fps_frame_count = 0
                
    def get_color_for_class(self, class_name):
        """Get color for a specific class"""
        return CLASS_COLORS.get(class_name, CLASS_COLORS['default'])

def yolov8_callback(pad, info, user_data):
    """
    Callback function for processing YOLOv8 detections
    """
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
        
    # Update frame count
    user_data.increment()
    user_data.frame_count = user_data.get_count()
    
    # Update FPS
    user_data.update_fps()
    
    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)
    
    # Get video frame if enabled
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)
    
    # Get detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # Process detections
    frame_detections = []
    detection_count_by_class = {}
    
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        # Filter by confidence threshold
        if confidence >= user_data.args.conf_threshold:
            # Map label to COCO class name if needed
            # The model might output class indices instead of names
            if label.isdigit():
                class_idx = int(label)
                if 0 <= class_idx < len(COCO_CLASSES):
                    label = COCO_CLASSES[class_idx]
            
            # Update detection count
            if label in detection_count_by_class:
                detection_count_by_class[label] += 1
            else:
                detection_count_by_class[label] = 1
            
            # Store detection info
            frame_detections.append({
                'label': label,
                'bbox': bbox,
                'confidence': confidence
            })
    
    # Update total statistics
    user_data.total_detections = len(frame_detections)
    for class_name, count in detection_count_by_class.items():
        if class_name in user_data.detection_stats:
            user_data.detection_stats[class_name] = count
    
    # Draw detections on frame if enabled
    if user_data.use_frame and frame is not None:
        # Draw each detection
        for det in frame_detections:
            label = det['label']
            bbox = det['bbox']
            confidence = det['confidence']
            
            # Get bbox coordinates
            x1 = int(bbox.xmin() * width)
            y1 = int(bbox.ymin() * height)
            x2 = int(bbox.xmax() * width)
            y2 = int(bbox.ymax() * height)
            
            # Get color for this class
            color = user_data.get_color_for_class(label)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_text = f"{label}: {confidence:.2f}"
            
            # Calculate text size
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(frame, (x1, y1 - text_height - 4), 
                         (x1 + text_width + 4, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label_text, (x1 + 2, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw statistics overlay
        draw_stats_overlay(frame, user_data)
        
        # Convert frame to BGR for display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)
    
    # Print detection summary (only if verbose or if detections found)
    if user_data.args.verbose or len(frame_detections) > 0:
        print(f"\rFrame {user_data.frame_count}: {len(frame_detections)} detections | "
              f"FPS: {user_data.current_fps:.1f}", end="")
        
        if user_data.args.verbose and len(frame_detections) > 0:
            print()  # New line
            for class_name, count in detection_count_by_class.items():
                print(f"  - {class_name}: {count}")
    
    return Gst.PadProbeReturn.OK

def draw_stats_overlay(frame, user_data):
    """Draw statistics overlay on the frame"""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    
    # Background for stats
    cv2.rectangle(overlay, (10, 10), (250, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw title
    cv2.putText(frame, "YOLOv8 Detection", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw FPS
    cv2.putText(frame, f"FPS: {user_data.current_fps:.1f}", (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # Draw total detections
    cv2.putText(frame, f"Detections: {user_data.total_detections}", (20, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    # Draw frame count
    cv2.putText(frame, f"Frame: {user_data.frame_count}", (20, 110),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # If there are active detections, show them on the right side
    if user_data.total_detections > 0:
        y_offset = 35
        x_offset = width - 200
        
        # Background for detection list
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (x_offset - 10, 10), 
                     (width - 10, min(10 + 30 * len(user_data.detection_stats), height - 10)), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
        
        # Draw active detections
        for class_name, count in user_data.detection_stats.items():
            if count > 0:
                color = user_data.get_color_for_class(class_name)
                cv2.putText(frame, f"{class_name}: {count}", (x_offset, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 25
                
                if y_offset > height - 20:
                    break

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 object detection with Hailo-8 using GStreamer')
    parser.add_argument('--input', '-i', type=str, default='rpi',
                       help='Input source: rpi (camera), usb, or file path')
    parser.add_argument('--model', '-m', type=str, default='yolov8s',
                       help='Model name (default: yolov8s)')
    parser.add_argument('--conf-threshold', '-c', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--use-frame', action='store_true', default=True,
                       help='Enable frame visualization (default: True)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--show-fps', type=int, default=30,
                       help='Display framerate (default: 30)')
    
    args = parser.parse_args()
    
    # Print startup info
    print("=" * 60)
    print("YOLOv8 Object Detection with Hailo-8")
    print("=" * 60)
    print(f"Input source: {args.input}")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"Frame visualization: {'Enabled' if args.use_frame else 'Disabled'}")
    print("=" * 60)
    print("\nStarting detection pipeline...")
    print("Press Ctrl+C to stop\n")
    
    # Create user data instance
    user_data = YOLOv8DetectionApp(args)
    user_data.use_frame = args.use_frame
    
    # Create and run the GStreamer app
    app = GStreamerDetectionApp(yolov8_callback, user_data)
    app.run()

if __name__ == "__main__":
    main()
