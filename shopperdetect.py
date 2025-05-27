import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example
        # Initialize face detection cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Keep state about facing/non-facing people
        self.facing_count = 0
        self.total_person_count = 0

    def new_function(self):  # New function example
        return "The meaning of life is: "
        
    def detect_face(self, frame, bbox):
        """Detect if a person is facing the camera by checking for face presence"""
        if frame is None:
            return False
            
        # Extract person ROI from the frame based on the bounding box
        x1, y1, x2, y2 = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
        # Ensure coordinates are within frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        # Get the region of interest (person area)
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            return False
            
        # Convert to grayscale for face detection
        gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_RGB2GRAY)
        
        # Detect faces in the ROI
        # Adjust parameters for better detection - smaller faces and fewer neighbors needed
        faces = self.face_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )
        
        # If any faces are detected in the person's bounding box, they're facing the camera
        return len(faces) > 0

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Reset counts for this frame
    user_data.facing_count = 0
    user_data.total_person_count = 0

    # Parse the detections
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        if label == "person":
            user_data.total_person_count += 1
            
            # Check if the person is facing the camera
            is_facing = False
            if user_data.use_frame and frame is not None:
                is_facing = user_data.detect_face(frame, bbox)
            
            # Only process and display people facing the camera
            if is_facing:
                user_data.facing_count += 1
                
                # Get track ID
                track_id = 0
                track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                if len(track) == 1:
                    track_id = track[0].get_id()
                
                string_to_print += (f"Window Shopper: ID: {track_id} Confidence: {confidence:.2f}\n")
                
                # Draw a green box and label for a person facing the camera
                if user_data.use_frame and frame is not None:
                    x1, y1, x2, y2 = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Shopper #{track_id}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # Optionally, draw a red box for people not facing the camera
                if user_data.use_frame and frame is not None:
                    x1, y1, x2, y2 = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Thinner red box

    if user_data.use_frame and frame is not None:
        # Add overall stats to the frame
        cv2.putText(frame, f"Window Shoppers: {user_data.facing_count}/{user_data.total_person_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Example of how to use the new_variable and new_function from the user_data
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert the frame to BGR for display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    # Enable frame processing
    user_data.use_frame = True
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
