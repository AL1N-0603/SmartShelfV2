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

# Enhanced callback class with productivity tracking
class DanProductivityTracker(app_callback_class):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
        self.total_frames = 0
        self.frames_with_person = 0
        self.current_status = "Dan's goofing off"
        self.goofing_off_percentage = 100.0

# Callback function
def dan_productivity_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    
    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # Get video frame
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Track person detection
    person_detected = False
    detection_count = 0
    
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        if label == "person":
            person_detected = True
            detection_count += 1
            
            # Draw box if frame available
            if user_data.use_frame and frame is not None:
                # Get coordinates properly from HailoBBox object by calling the methods
                x1, y1 = bbox.xmin(), bbox.ymin()
                x2, y2 = bbox.xmax(), bbox.ymax()
                
                # Convert normalized coordinates to pixel coordinates
                x1, x2 = int(x1 * width), int(x2 * width)
                y1, y2 = int(y1 * height), int(y2 * height)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"Person: {confidence:.2f}"
                cv2.putText(frame, label_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update productivity stats
    user_data.total_frames += 1
    if person_detected:
        user_data.frames_with_person += 1
        user_data.current_status = "Dan's working"
    else:
        user_data.current_status = "Dan's goofing off"
    
    # Calculate percentage
    if user_data.total_frames > 0:
        user_data.goofing_off_percentage = 100.0 * (user_data.total_frames - user_data.frames_with_person) / user_data.total_frames
    
    # Add status overlay
    if user_data.use_frame and frame is not None:
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 100), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        
        # Status text
        status_color = (0, 255, 0) if person_detected else (0, 0, 255)
        cv2.putText(frame, user_data.current_status, (20, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        
        # Percentage
        cv2.putText(frame, f"Goofing: {user_data.goofing_off_percentage:.1f}%", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Convert for display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)
    
    # Console output
    print(f"\rFrames: {user_data.total_frames} | Status: {user_data.current_status} | Goofing: {user_data.goofing_off_percentage:.1f}%", end="")
    
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    # Create tracker and app
    print("[INFO] Starting Dan's Productivity Monitor...")
    tracker = DanProductivityTracker()
    tracker.use_frame = True  # Enable frame processing
    
    app = GStreamerDetectionApp(dan_productivity_callback, tracker)
    app.run()
