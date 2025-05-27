import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# COCO class names for reference
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

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.camera_info_printed = False
        self.detected_labels = set()  # Track unique labels
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0
        
    def get_camera_info(self):
        return "RPi Camera (imx219) - 640x480"

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
        
    # Using the user_data to count the number of frames
    user_data.increment()
    frame_count = user_data.get_count()
    
    # Calculate FPS
    user_data.fps_frame_count += 1
    current_time = time.time()
    elapsed = current_time - user_data.fps_start_time
    if elapsed > 1.0:
        user_data.current_fps = user_data.fps_frame_count / elapsed
        user_data.fps_start_time = current_time
        user_data.fps_frame_count = 0
    
    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)
    
    # Print camera info once
    if not user_data.camera_info_printed and format is not None:
        print(f"\n=== Camera Info ===")
        print(f"Format: {format}")
        print(f"Resolution: {width}x{height}")
        print(f"Camera: {user_data.get_camera_info()}")
        print(f"==================\n")
        user_data.camera_info_printed = True
    
    # Get video frame if enabled
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)
    
    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # Parse ALL detections (not just persons)
    detection_count = 0
    detections_by_class = {}
    
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        # Skip low confidence detections
        if confidence < 0.5:
            continue
            
        # Handle different label formats
        display_label = label
        
        # If label is a number, try to map to COCO class
        if label.isdigit():
            idx = int(label)
            if 0 <= idx < len(COCO_CLASSES):
                display_label = COCO_CLASSES[idx]
        # Handle plural labels (e.g., "persons" -> "person")
        elif label == "persons":
            display_label = "person"
        
        # Track unique labels
        if label not in user_data.detected_labels:
            user_data.detected_labels.add(label)
            print(f"[NEW LABEL DETECTED]: '{label}' -> '{display_label}'")
        
        # Count detections by class
        if display_label not in detections_by_class:
            detections_by_class[display_label] = []
        
        detections_by_class[display_label].append({
            'bbox': bbox,
            'confidence': confidence,
            'track_id': 0
        })
        
        # Get track ID if available
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            detections_by_class[display_label][-1]['track_id'] = track[0].get_id()
        
        detection_count += 1
    
    # Draw on frame if enabled
    if user_data.use_frame and frame is not None:
        # Draw FPS and camera info
        cv2.putText(frame, f"FPS: {user_data.current_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Camera: {user_data.get_camera_info()}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count} | Detections: {detection_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw all detections
        y_offset = 120
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        color_idx = 0
        
        for class_name, class_detections in detections_by_class.items():
            color = colors[color_idx % len(colors)]
            color_idx += 1
            
            # Draw class summary
            cv2.putText(frame, f"{class_name}: {len(class_detections)}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
            
            # Draw bounding boxes
            for det in class_detections:
                bbox = det['bbox']
                conf = det['confidence']
                track_id = det['track_id']
                
                # Convert normalized coordinates to pixel coordinates
                x1 = int(bbox.xmin() * width)
                y1 = int(bbox.ymin() * height)
                x2 = int(bbox.xmax() * width)
                y2 = int(bbox.ymax() * height)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label_text = f"{class_name} {conf:.2f}"
                if track_id > 0:
                    label_text += f" ID:{track_id}"
                
                cv2.putText(frame, label_text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)
    
    # Print summary every 30 frames
    if frame_count % 30 == 0 and detection_count > 0:
        print(f"\nFrame {frame_count} summary:")
        print(f"  Total detections: {detection_count}")
        print(f"  FPS: {user_data.current_fps:.1f}")
        for class_name, class_dets in detections_by_class.items():
            print(f"  - {class_name}: {len(class_dets)}")
        print(f"  All detected labels so far: {sorted(user_data.detected_labels)}")
    
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    print("=" * 60)
    print("RPi Camera Object Detection")
    print("=" * 60)
    print("Starting detection pipeline...")
    print("Camera: RPi Camera Module (imx219)")
    print("Resolution: 640x480")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    user_data.use_frame = True  # Enable frame visualization
    
    # Create and run the app
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
