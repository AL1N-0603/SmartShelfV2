#!/usr/bin/env python3
"""
YOLOv8 object detection using RPi camera with GStreamer pipeline
Based on the working detection.py approach
Now includes person counting and data logging functionality
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time
import argparse
import csv
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from datetime import datetime, timezone
import pytz
import atexit
import signal
import sys

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

# Color palette for different classes (BGR format for OpenCV)
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
    'bottle': (255, 128, 128),   # Light Red
    'cup': (128, 128, 255),      # Light Purple
    'default': (255, 255, 255)   # White
}

class YOLOv8CameraApp(app_callback_class):
    """
    YOLOv8 detection app for RPi camera with person counting and logging
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.camera_info_printed = False
        self.detected_labels = set()
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0
        self.total_detections = 0
        self.frame_count = 0
        self.detection_stats = {}
        
        # Logging-related attributes
        self.last_log_time = 0
        self.last_plot_time = 0
        self.current_date = None
        self.csv_file = None
        self.csv_writer = None
        self.current_person_count = 0
        
        # Create directories
        Path("logs").mkdir(exist_ok=True)
        Path("plots").mkdir(exist_ok=True)
        
        # Initialize stats for all classes
        for class_name in COCO_CLASSES:
            self.detection_stats[class_name] = 0
            
        # Setup graceful shutdown
        self.setup_graceful_shutdown()
        
    def setup_graceful_shutdown(self):
        """Setup handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print("\nReceived shutdown signal, cleaning up...")
            self.cleanup()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(self.cleanup)
        
    def cleanup(self):
        """Cleanup resources and generate final plot"""
        if self.csv_file:
            self.csv_file.close()
        if self.current_date:
            try:
                self.generate_plot(self.current_date)
                print(f"Generated final plot for {self.current_date}")
            except Exception as e:
                print(f"Error generating final plot: {e}")
                
    def get_current_date(self):
        """Get current date string"""
        return datetime.now().strftime("%Y-%m-%d")
        
    def get_csv_filename(self, date_str):
        """Get CSV filename for given date"""
        return f"logs/person_counts_{date_str}.csv"
        
    def get_plot_filename(self, date_str):
        """Get plot filename for given date"""
        return f"plots/person_counts_{date_str}.png"
        
    def ensure_csv_file_open(self):
        """Ensure CSV file is open for current date"""
        current_date = self.get_current_date()
        
        # Check if we need to open a new file (new day or first time)
        if self.current_date != current_date:
            # Close existing file if open
            if self.csv_file:
                self.csv_file.close()
                # Generate plot for the previous day
                if self.current_date:
                    try:
                        self.generate_plot(self.current_date)
                        print(f"Generated plot for {self.current_date}")
                    except Exception as e:
                        print(f"Error generating plot for {self.current_date}: {e}")
            
            # Open new file for current date
            self.current_date = current_date
            csv_filename = self.get_csv_filename(current_date)
            
            # Check if file exists to determine if we need headers
            file_exists = Path(csv_filename).exists()
            
            self.csv_file = open(csv_filename, 'a', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header if new file
            if not file_exists:
                self.csv_writer.writerow(['timestamp', 'person_count'])
                print(f"Created new CSV file: {csv_filename}")
            else:
                print(f"Appending to existing CSV file: {csv_filename}")
                
    def log_person_count(self, person_count):
        """Log person count to CSV file"""
        self.ensure_csv_file_open()
        
        # Coerce count to integer and clamp negatives to zero
        count = max(0, int(person_count))
        
        # Get current timestamp in Perth timezone
        perth_tz = pytz.timezone('Australia/Perth')
        timestamp = datetime.now(perth_tz).isoformat()
        
        # Write to CSV
        self.csv_writer.writerow([timestamp, count])
        self.csv_file.flush()  # Ensure data is written
        
    def should_log_frame(self):
        """Check if we should log this frame (1 second interval)"""
        current_time = time.time()
        if current_time - self.last_log_time >= 1.0:
            self.last_log_time = current_time
            return True
        return False
        
    def should_generate_plot(self):
        """Check if we should generate plot (5 minute interval)"""
        current_time = time.time()
        if current_time - self.last_plot_time >= 300.0:  # 5 minutes
            self.last_plot_time = current_time
            return True
        return False
        
    def generate_plot(self, date_str):
        """Generate plot for given date"""
        csv_filename = self.get_csv_filename(date_str)
        plot_filename = self.get_plot_filename(date_str)
        
        if not Path(csv_filename).exists():
            return
            
        try:
            # Read CSV data with timestamp parsing
            df = pd.read_csv(csv_filename, parse_dates=[0])
            
            if df.empty:
                return
                
            # Convert timestamp to Perth timezone
            df['timestamp'] = df['timestamp'].dt.tz_convert('Australia/Perth')
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['timestamp'], df['person_count'], 'b-', linewidth=1.5, alpha=0.8)
            ax.fill_between(df['timestamp'], df['person_count'], alpha=0.3)
            
            # Format plot
            ax.set_xlabel('Local time (AWST)')
            ax.set_ylabel('Person Count')
            ax.set_title(f'Person Detection Count - {date_str}')
            ax.grid(True, alpha=0.3)
            
            # Force integer y-ticks and set minimum at 0
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
            ax.set_ylim(bottom=0)
            
            # Format x-axis with hourly ticks
            if len(df) > 1:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                fig.autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Error generating plot: {e}")
            
    def get_camera_info(self):
        return "RPi Camera (imx219)"
        
    def get_color_for_class(self, class_name):
        """Get color for a specific class"""
        return CLASS_COLORS.get(class_name, CLASS_COLORS['default'])
        
    def update_fps(self):
        """Calculate and update FPS"""
        self.fps_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        
        if elapsed > 1.0:  # Update FPS every second
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_start_time = current_time
            self.fps_frame_count = 0

def yolov8_callback(pad, info, user_data):
    """
    Callback function for YOLOv8 detection on RPi camera stream
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
    
    # Print camera info once
    if not user_data.camera_info_printed and format is not None:
        print(f"\n=== YOLOv8 Detection Started ===")
        print(f"Camera: {user_data.get_camera_info()}")
        print(f"Format: {format}")
        print(f"Resolution: {width}x{height}")
        print(f"Model: {user_data.args.model}")
        print(f"Confidence threshold: {user_data.args.conf_threshold}")
        print(f"GUI: {'Disabled' if user_data.args.no_gui else 'Enabled'}")
        print(f"Person counting: Enabled (1 sample/sec)")
        print(f"Plotting: Every 5 minutes")
        print(f"================================\n")
        user_data.camera_info_printed = True
    
    # Get video frame if enabled
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)
    
    # Get detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # Reset current frame stats
    for class_name in user_data.detection_stats:
        user_data.detection_stats[class_name] = 0
    
    # Process detections
    frame_detections = []
    detections_by_class = {}
    person_count = 0
    
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        # Skip low confidence detections
        if confidence < user_data.args.conf_threshold:
            continue
            
        # Handle different label formats
        display_label = label
        
        # If label is a number, map to COCO class
        if label.isdigit():
            idx = int(label)
            if 0 <= idx < len(COCO_CLASSES):
                display_label = COCO_CLASSES[idx]
        # Handle plural labels
        elif label == "persons":
            display_label = "person"
        elif label.endswith('s') and label[:-1] in COCO_CLASSES:
            display_label = label[:-1]
        
        # Count persons for logging
        if display_label == "person" or label == "0":  # YOLO class id 0 is person
            person_count += 1
        
        # Track unique labels for debugging
        if label not in user_data.detected_labels:
            user_data.detected_labels.add(label)
            print(f"[NEW LABEL]: '{label}' -> '{display_label}'")
        
        # Count detections by class
        if display_label not in detections_by_class:
            detections_by_class[display_label] = []
        
        # Get track ID if available
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
        
        detections_by_class[display_label].append({
            'bbox': bbox,
            'confidence': confidence,
            'track_id': track_id
        })
        
        frame_detections.append({
            'label': display_label,
            'bbox': bbox,
            'confidence': confidence,
            'track_id': track_id
        })
    
    # Update statistics
    user_data.total_detections = len(frame_detections)
    user_data.current_person_count = person_count
    for class_name, class_dets in detections_by_class.items():
        if class_name in user_data.detection_stats:
            user_data.detection_stats[class_name] = len(class_dets)
    
    # Log person count every second
    if user_data.should_log_frame():
        user_data.log_person_count(person_count)
        
        # Generate plot every 5 minutes
        if user_data.should_generate_plot():
            try:
                user_data.generate_plot(user_data.current_date)
                print(f"Generated plot update for {user_data.current_date}")
            except Exception as e:
                print(f"Error generating plot: {e}")
    
    # Draw on frame if enabled
    if user_data.use_frame and frame is not None:
        # Draw detections
        for det in frame_detections:
            label = det['label']
            bbox = det['bbox']
            confidence = det['confidence']
            track_id = det['track_id']
            
            # Get color for this class
            color = user_data.get_color_for_class(label)
            
            # Convert normalized coordinates to pixel coordinates
            x1 = int(bbox.xmin() * width)
            y1 = int(bbox.ymin() * height)
            x2 = int(bbox.xmax() * width)
            y2 = int(bbox.ymax() * height)
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_text = f"{label}: {confidence:.2f}"
            if track_id > 0:
                label_text += f" ID:{track_id}"
            
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
        draw_stats_overlay(frame, user_data, width, height)
        
        # Convert frame to BGR for display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)
    
    # Print summary periodically
    if user_data.args.verbose or (user_data.frame_count % 30 == 0 and user_data.total_detections > 0):
        print(f"\rFrame {user_data.frame_count}: {user_data.total_detections} detections | "
              f"FPS: {user_data.current_fps:.1f} | Persons: {person_count}", end="")
        
        if user_data.args.verbose and user_data.total_detections > 0:
            print()  # New line
            for class_name, count in detections_by_class.items():
                print(f"  - {class_name}: {count}")
    
    return Gst.PadProbeReturn.OK

def draw_stats_overlay(frame, user_data, width, height):
    """Draw statistics overlay on the frame"""
    # Create semi-transparent overlay for main stats
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw title
    cv2.putText(frame, "YOLOv8 Detection - RPi Camera", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw camera info
    cv2.putText(frame, f"Camera: {user_data.get_camera_info()}", (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw FPS
    cv2.putText(frame, f"FPS: {user_data.current_fps:.1f}", (20, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # Draw total detections
    cv2.putText(frame, f"Detections: {user_data.total_detections}", (20, 110),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    # Draw person count (highlighted)
    cv2.putText(frame, f"Persons: {user_data.current_person_count}", (20, 135),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw frame count
    cv2.putText(frame, f"Frame: {user_data.frame_count}", (20, 155),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # If there are active detections, show them on the right side
    if user_data.total_detections > 0:
        y_offset = 35
        x_offset = width - 250
        
        # Background for detection list
        active_classes = sum(1 for count in user_data.detection_stats.values() if count > 0)
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (x_offset - 10, 10), 
                     (width - 10, min(10 + 30 * active_classes + 20, height - 10)), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
        
        # Title for detection list
        cv2.putText(frame, "Detected Objects:", (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
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
    parser = argparse.ArgumentParser(description='YOLOv8 object detection with RPi Camera')
    parser.add_argument('--model', '-m', type=str, default='yolov8s',
                       help='Model name (default: yolov8s)')
    parser.add_argument('--hef', type=str, default='hailo_model_zoo/yolov8s_h8l.hef',
                       help='Path to HEF file')
    parser.add_argument('--conf-threshold', '-c', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--input', '-i', type=str, default='rpi',
                       help='Input source (default: rpi for RPi camera)')
    parser.add_argument('--use-frame', action='store_true', default=True,
                       help='Enable frame visualization (default: True)')
    parser.add_argument('--no-gui', action='store_true',
                       help='Disable GUI display for headless operation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Save output video to file')
    
    args = parser.parse_args()
    
    # Disable frame visualization if no-gui is set
    if args.no_gui:
        args.use_frame = False
    
    # Set HEF file path if provided
    if args.hef and os.path.exists(args.hef):
        os.environ['HAILO_HEF_PATH'] = args.hef
        print(f"Using HEF file: {args.hef}")
    
    # Print startup info
    print("=" * 60)
    print("YOLOv8 Object Detection with RPi Camera + Person Counting")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"HEF file: {args.hef}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"Input source: {args.input} (RPi Camera)")
    print(f"Frame visualization: {'Enabled' if args.use_frame else 'Disabled'}")
    print(f"GUI mode: {'Disabled (headless)' if args.no_gui else 'Enabled'}")
    print(f"Person counting: Enabled (logging to CSV)")
    print(f"Plot generation: Every 5 minutes + on shutdown")#!/usr/bin/env python3
"""
YOLOv8 object detection using RPi camera with GStreamer pipeline
Based on the working detection.py approach
Now includes person counting and data logging functionality
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time
import argparse
import csv
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timezone
import atexit
import signal
import sys

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

# Color palette for different classes (BGR format for OpenCV)
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
    'bottle': (255, 128, 128),   # Light Red
    'cup': (128, 128, 255),      # Light Purple
    'default': (255, 255, 255)   # White
}

class YOLOv8CameraApp(app_callback_class):
    """
    YOLOv8 detection app for RPi camera with person counting and logging
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.camera_info_printed = False
        self.detected_labels = set()
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0
        self.total_detections = 0
        self.frame_count = 0
        self.detection_stats = {}
        
        # Logging-related attributes
        self.last_log_time = 0
        self.last_plot_time = 0
        self.current_date = None
        self.csv_file = None
        self.csv_writer = None
        self.current_person_count = 0
        
        # Create directories
        Path("logs").mkdir(exist_ok=True)
        Path("plots").mkdir(exist_ok=True)
        
        # Initialize stats for all classes
        for class_name in COCO_CLASSES:
            self.detection_stats[class_name] = 0
            
        # Setup graceful shutdown
        self.setup_graceful_shutdown()
        
    def setup_graceful_shutdown(self):
        """Setup handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print("\nReceived shutdown signal, cleaning up...")
            self.cleanup()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(self.cleanup)
        
    def cleanup(self):
        """Cleanup resources and generate final plot"""
        if self.csv_file:
            self.csv_file.close()
        if self.current_date:
            try:
                self.generate_plot(self.current_date)
                print(f"Generated final plot for {self.current_date}")
            except Exception as e:
                print(f"Error generating final plot: {e}")
                
    def get_current_date(self):
        """Get current date string"""
        return datetime.now().strftime("%Y-%m-%d")
        
    def get_csv_filename(self, date_str):
        """Get CSV filename for given date"""
        return f"logs/person_counts_{date_str}.csv"
        
    def get_plot_filename(self, date_str):
        """Get plot filename for given date"""
        return f"plots/person_counts_{date_str}.png"
        
    def ensure_csv_file_open(self):
        """Ensure CSV file is open for current date"""
        current_date = self.get_current_date()
        
        # Check if we need to open a new file (new day or first time)
        if self.current_date != current_date:
            # Close existing file if open
            if self.csv_file:
                self.csv_file.close()
                # Generate plot for the previous day
                if self.current_date:
                    try:
                        self.generate_plot(self.current_date)
                        print(f"Generated plot for {self.current_date}")
                    except Exception as e:
                        print(f"Error generating plot for {self.current_date}: {e}")
            
            # Open new file for current date
            self.current_date = current_date
            csv_filename = self.get_csv_filename(current_date)
            
            # Check if file exists to determine if we need headers
            file_exists = Path(csv_filename).exists()
            
            self.csv_file = open(csv_filename, 'a', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header if new file
            if not file_exists:
                self.csv_writer.writerow(['timestamp', 'person_count'])
                print(f"Created new CSV file: {csv_filename}")
            else:
                print(f"Appending to existing CSV file: {csv_filename}")
                
    def log_person_count(self, person_count):
        """Log person count to CSV file"""
        self.ensure_csv_file_open()
        
        # Get current timestamp in ISO-8601 format with timezone
        timestamp = datetime.now(timezone.utc).astimezone().isoformat()
        
        # Write to CSV
        self.csv_writer.writerow([timestamp, person_count])
        self.csv_file.flush()  # Ensure data is written
        
    def should_log_frame(self):
        """Check if we should log this frame (1 second interval)"""
        current_time = time.time()
        if current_time - self.last_log_time >= 1.0:
            self.last_log_time = current_time
            return True
        return False
        
    def should_generate_plot(self):
        """Check if we should generate plot (5 minute interval)"""
        current_time = time.time()
        if current_time - self.last_plot_time >= 300.0:  # 5 minutes
            self.last_plot_time = current_time
            return True
        return False
        
    def generate_plot(self, date_str):
        """Generate plot for given date"""
        csv_filename = self.get_csv_filename(date_str)
        plot_filename = self.get_plot_filename(date_str)
        
        if not Path(csv_filename).exists():
            return
            
        try:
            # Read CSV data
            df = pd.read_csv(csv_filename)
            
            if df.empty:
                return
                
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['timestamp'], df['person_count'], 'b-', linewidth=1.5, alpha=0.8)
            ax.fill_between(df['timestamp'], df['person_count'], alpha=0.3)
            
            # Format plot
            ax.set_xlabel('Time')
            ax.set_ylabel('Person Count')
            ax.set_title(f'Person Detection Count - {date_str}')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            if len(df) > 1:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                plt.xticks(rotation=45)
            
            # Set y-axis to start from 0
            ax.set_ylim(bottom=0)
            
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Error generating plot: {e}")
            
    def get_camera_info(self):
        return "RPi Camera (imx219)"
        
    def get_color_for_class(self, class_name):
        """Get color for a specific class"""
        return CLASS_COLORS.get(class_name, CLASS_COLORS['default'])
        
    def update_fps(self):
        """Calculate and update FPS"""
        self.fps_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        
        if elapsed > 1.0:  # Update FPS every second
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_start_time = current_time
            self.fps_frame_count = 0

def yolov8_callback(pad, info, user_data):
    """
    Callback function for YOLOv8 detection on RPi camera stream
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
    
    # Print camera info once
    if not user_data.camera_info_printed and format is not None:
        print(f"\n=== YOLOv8 Detection Started ===")
        print(f"Camera: {user_data.get_camera_info()}")
        print(f"Format: {format}")
        print(f"Resolution: {width}x{height}")
        print(f"Model: {user_data.args.model}")
        print(f"Confidence threshold: {user_data.args.conf_threshold}")
        print(f"GUI: {'Disabled' if user_data.args.no_gui else 'Enabled'}")
        print(f"Person counting: Enabled (1 sample/sec)")
        print(f"Plotting: Every 5 minutes")
        print(f"================================\n")
        user_data.camera_info_printed = True
    
    # Get video frame if enabled
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)
    
    # Get detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # Reset current frame stats
    for class_name in user_data.detection_stats:
        user_data.detection_stats[class_name] = 0
    
    # Process detections
    frame_detections = []
    detections_by_class = {}
    person_count = 0
    
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        # Skip low confidence detections
        if confidence < user_data.args.conf_threshold:
            continue
            
        # Handle different label formats
        display_label = label
        
        # If label is a number, map to COCO class
        if label.isdigit():
            idx = int(label)
            if 0 <= idx < len(COCO_CLASSES):
                display_label = COCO_CLASSES[idx]
        # Handle plural labels
        elif label == "persons":
            display_label = "person"
        elif label.endswith('s') and label[:-1] in COCO_CLASSES:
            display_label = label[:-1]
        
        # Count persons for logging
        if display_label == "person" or label == "0":  # YOLO class id 0 is person
            person_count += 1
        
        # Track unique labels for debugging
        if label not in user_data.detected_labels:
            user_data.detected_labels.add(label)
            print(f"[NEW LABEL]: '{label}' -> '{display_label}'")
        
        # Count detections by class
        if display_label not in detections_by_class:
            detections_by_class[display_label] = []
        
        # Get track ID if available
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
        
        detections_by_class[display_label].append({
            'bbox': bbox,
            'confidence': confidence,
            'track_id': track_id
        })
        
        frame_detections.append({
            'label': display_label,
            'bbox': bbox,
            'confidence': confidence,
            'track_id': track_id
        })
    
    # Update statistics
    user_data.total_detections = len(frame_detections)
    user_data.current_person_count = person_count
    for class_name, class_dets in detections_by_class.items():
        if class_name in user_data.detection_stats:
            user_data.detection_stats[class_name] = len(class_dets)
    
    # Log person count every second
    if user_data.should_log_frame():
        user_data.log_person_count(person_count)
        
        # Generate plot every 5 minutes
        if user_data.should_generate_plot():
            try:
                user_data.generate_plot(user_data.current_date)
                print(f"Generated plot update for {user_data.current_date}")
            except Exception as e:
                print(f"Error generating plot: {e}")
    
    # Draw on frame if enabled
    if user_data.use_frame and frame is not None:
        # Draw detections
        for det in frame_detections:
            label = det['label']
            bbox = det['bbox']
            confidence = det['confidence']
            track_id = det['track_id']
            
            # Get color for this class
            color = user_data.get_color_for_class(label)
            
            # Convert normalized coordinates to pixel coordinates
            x1 = int(bbox.xmin() * width)
            y1 = int(bbox.ymin() * height)
            x2 = int(bbox.xmax() * width)
            y2 = int(bbox.ymax() * height)
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_text = f"{label}: {confidence:.2f}"
            if track_id > 0:
                label_text += f" ID:{track_id}"
            
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
        draw_stats_overlay(frame, user_data, width, height)
        
        # Convert frame to BGR for display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)
    
    # Print summary periodically
    if user_data.args.verbose or (user_data.frame_count % 30 == 0 and user_data.total_detections > 0):
        print(f"\rFrame {user_data.frame_count}: {user_data.total_detections} detections | "
              f"FPS: {user_data.current_fps:.1f} | Persons: {person_count}", end="")
        
        if user_data.args.verbose and user_data.total_detections > 0:
            print()  # New line
            for class_name, count in detections_by_class.items():
                print(f"  - {class_name}: {count}")
    
    return Gst.PadProbeReturn.OK

def draw_stats_overlay(frame, user_data, width, height):
    """Draw statistics overlay on the frame"""
    # Create semi-transparent overlay for main stats
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw title
    cv2.putText(frame, "YOLOv8 Detection - RPi Camera", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw camera info
    cv2.putText(frame, f"Camera: {user_data.get_camera_info()}", (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw FPS
    cv2.putText(frame, f"FPS: {user_data.current_fps:.1f}", (20, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # Draw total detections
    cv2.putText(frame, f"Detections: {user_data.total_detections}", (20, 110),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    # Draw person count (highlighted)
    cv2.putText(frame, f"Persons: {user_data.current_person_count}", (20, 135),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw frame count
    cv2.putText(frame, f"Frame: {user_data.frame_count}", (20, 155),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # If there are active detections, show them on the right side
    if user_data.total_detections > 0:
        y_offset = 35
        x_offset = width - 250
        
        # Background for detection list
        active_classes = sum(1 for count in user_data.detection_stats.values() if count > 0)
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (x_offset - 10, 10), 
                     (width - 10, min(10 + 30 * active_classes + 20, height - 10)), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
        
        # Title for detection list
        cv2.putText(frame, "Detected Objects:", (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
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
    parser = argparse.ArgumentParser(description='YOLOv8 object detection with RPi Camera')
    parser.add_argument('--model', '-m', type=str, default='yolov8s',
                       help='Model name (default: yolov8s)')
    parser.add_argument('--hef', type=str, default='hailo_model_zoo/yolov8s_h8l.hef',
                       help='Path to HEF file')
    parser.add_argument('--conf-threshold', '-c', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--input', '-i', type=str, default='rpi',
                       help='Input source (default: rpi for RPi camera)')
    parser.add_argument('--use-frame', action='store_true', default=True,
                       help='Enable frame visualization (default: True)')
    parser.add_argument('--no-gui', action='store_true',
                       help='Disable GUI display for headless operation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Save output video to file')
    
    args = parser.parse_args()
    
    # Disable frame visualization if no-gui is set
    if args.no_gui:
        args.use_frame = False
    
    # Set HEF file path if provided
    if args.hef and os.path.exists(args.hef):
        os.environ['HAILO_HEF_PATH'] = args.hef
        print(f"Using HEF file: {args.hef}")
    
    # Print startup info
    print("=" * 60)
    print("YOLOv8 Object Detection with RPi Camera + Person Counting")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"HEF file: {args.hef}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"Input source: {args.input} (RPi Camera)")
    print(f"Frame visualization: {'Enabled' if args.use_frame else 'Disabled'}")
    print(f"GUI mode: {'Disabled (headless)' if args.no_gui else 'Enabled'}")
    print(f"Person counting: Enabled (logging to CSV)")
    print(f"Plot generation: Every 5 minutes + on shutdown")
    print("=" * 60)
    print("\nStarting detection pipeline...")
    print("Press Ctrl+C to stop\n")
    
    # Create user data instance
    user_data = YOLOv8CameraApp(args)
    user_data.use_frame = args.use_frame
    
    try:
        # Create and run the GStreamer app
        # The app will handle the camera input based on args.input
        app = GStreamerDetectionApp(yolov8_callback, user_data)
        app.run()
    except KeyboardInterrupt:
        print("\n\nStopping detection...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nDetection stopped.")
        user_data.cleanup()
        if len(user_data.detected_labels) > 0:
            print(f"Unique labels detected: {sorted(user_data.detected_labels)}")

if __name__ == "__main__":
    main()
    print("=" * 60)
    print("\nStarting detection pipeline...")
    print("Press Ctrl+C to stop\n")
    
    # Create user data instance
    user_data = YOLOv8CameraApp(args)
    user_data.use_frame = args.use_frame
    
    try:
        # Create and run the GStreamer app
        # The app will handle the camera input based on args.input
        app = GStreamerDetectionApp(yolov8_callback, user_data)
        app.run()
    except KeyboardInterrupt:
        print("\n\nStopping detection...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nDetection stopped.")
        user_data.cleanup()
        if len(user_data.detected_labels) > 0:
            print(f"Unique labels detected: {sorted(user_data.detected_labels)}")

if __name__ == "__main__":
    main()