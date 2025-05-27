# dan_at_work_cv2_debug.py
# Removed Streamlit, using direct OpenCV display for debugging

# Keep necessary imports
from picamera2 import Picamera2
import time
import numpy as np
import cv2
import os

# <<< ADAPT: Add necessary HailoRT imports from detection.py >>>
# Example: May include things like Device, HailoRTException, InferVStreams, ConfigureNetworkParams, FormatType, FormatOrder etc.
try:
    # <<< COPY/ADAPT Hailo specific imports from detection.py >>>
    # Example: from hailo_platform.runtime.pyhailort import Device, InferVStreams, ConfigureNetworkParams, FormatType, FormatOrder, HailoRTException
    HAILO_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] Failed to import Hailo libraries. Ensure SDK is installed: {e}")
    HAILO_AVAILABLE = False
    # exit() # Optional: Exit if Hailo is essential

# --- Configuration (Same as before) ---
HAILO_MODEL_WIDTH = 640
HAILO_MODEL_HEIGHT = 640
MAIN_FRAME_WIDTH = 1280
MAIN_FRAME_HEIGHT = 720

# <<< ADAPT: Get HEV path used in detection.py >>>
HAILO_HEV_FILE = "/path/to/your/model.hev" # <<<=== FIND AND SET THIS PATH!

# <<< ADAPT: Get relevant parameters from detection.py >>>
PERSON_CLASS_ID = 0 # IMPORTANT: Find the correct class ID for 'person'
DETECTION_THRESHOLD = 0.5 # Check threshold used

# --- Global Variables (Instead of st.session_state) ---
picam2 = None
network_group = None
# input_vstream_info = None # Optional: Store if needed
# output_vstream_info = None # Optional: Store if needed
start_time = None
total_time_at_work = 0.0
last_update_time = None
person_present_last_check = False

# --- Initialization Function ---
def initialize():
    global picam2, network_group, start_time, last_update_time # Declare globals we modify
    print("[INFO] Initializing...")
    try:
        # --- Camera Setup ---
        print("[INFO] Initializing Camera...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (MAIN_FRAME_WIDTH, MAIN_FRAME_HEIGHT), "format": "RGB888"},
            lores={"size": (HAILO_MODEL_WIDTH, HAILO_MODEL_HEIGHT), "format": "YUV420"}, # Check format needed
            controls={"FrameDurationLimits": (33333, 33333)}
        )
        picam2.configure(config)
        picam2.start()
        print("[INFO] Camera Initialized!")

        # --- Hailo Setup ---
        if HAILO_AVAILABLE:
            print("[INFO] Initializing Hailo AI Module...")
            # <<< ADAPT: Copy Hailo device scanning/selection logic from detection.py >>>
            # Example: devices = Device.scan(); device = Device(devices[0]) ...

            # <<< ADAPT: Copy HEV loading and network configuration logic from detection.py >>>
            # Example: params = ConfigureNetworkParams(...)
            # Example: network_group_local = device.create_configured_network_group(HAILO_HEV_FILE, params)
            # network_group = network_group_local # Assign to global variable

            # <<< Optional: Get input/output stream info if needed >>>
            # Example: input_vstream_info = network_group.get_input_vstream_infos()[0]
            # Example: output_vstream_info = network_group.get_output_vstream_infos()
            print("[INFO] Hailo Initialized!")
        else:
             print("[WARNING] Hailo library not found. Detection will not function.")

        start_time = time.monotonic()
        last_update_time = start_time
        return True

    except Exception as e:
        print(f"[ERROR] Initialization Failed: {e}")
        if picam2: picam2.stop()
        # <<< Add Hailo cleanup if necessary >>>
        return False

# --- Main Execution ---
if initialize():
    print("[INFO] Starting main loop...")
    while True:
        request = picam2.capture_request()
        main_frame = request.make_array("main") # For display (RGB888)
        lores_frame_yuv = request.make_array("lores") # For AI (YUV420 or configured format)
        request.release()

        if main_frame is None:
            print("[WARNING] Failed to capture main frame.")
            time.sleep(0.1)
            continue

        person_detected_this_frame = False
        processed_detections = [] # To store final detections

        # --- Hailo Inference Pipeline (Adapt from detection.py) ---
        if HAILO_AVAILABLE and network_group:
            try:
                # 1. <<< ADAPT: Preprocessing logic from detection.py >>>
                # Example: input_frame_rgb = cv2.cvtColor(lores_frame_yuv, cv2.COLOR_YUV420p_to_RGB)
                # input_data = preprocess_input(input_frame_rgb) # Apply normalization etc.

                # 2. <<< ADAPT: Inference call from detection.py >>>
                # Example: with network_group.infer_async(input_data) as infer_results:
                # Example:     raw_hailo_output = infer_results.get()

                # 3. <<< ADAPT: Postprocessing logic from detection.py >>>
                # Example function call: final_detections = postprocess_yolov8(raw_hailo_output, ...)
                # processed_detections = final_detections

                # 4. Check for 'person' in the final detections
                for detection in processed_detections:
                    # <<< ADAPT: Extract class ID and score based on detection format >>>
                    # Example: bbox, score, class_id = detection
                    if class_id == PERSON_CLASS_ID and score > DETECTION_THRESHOLD:
                        person_detected_this_frame = True
                        break # Found a person

            except Exception as e:
                 # print(f"Hailo Pipeline Error: {e}") # Log error for debugging
                 pass # Allow loop to continue

        # --- Timing Update ---
        current_time = time.monotonic()
        time_delta = current_time - last_update_time
        if person_present_last_check:
            total_time_at_work += time_delta
        total_elapsed_time = current_time - start_time
        work_percentage = (total_time_at_work / total_elapsed_time) * 100 if total_elapsed_time > 0 else 0.0
        last_update_time = current_time
        person_present_last_check = person_detected_this_frame

        # --- Console Output ---
        status_message = "✅ Dan at work" if person_detected_this_frame else "❌ Dan's goofing off"
        # Print status and percentage on the same line, overwriting previous output
        print(f"\rStatus: {status_message} | Work Percentage: {work_percentage:.1f}%   ", end="")


        # --- OpenCV Display ---
        # Optional: Draw bounding boxes on the main_frame before displaying
        display_frame = main_frame.copy() # Draw on a copy
        # <<< ADAPT: Drawing logic from detection.py (using cv2.rectangle/putText) >>>
        # Example: Iterate through 'processed_detections' again
        # for bbox, score, class_id in processed_detections:
        #    if class_id == PERSON_CLASS_ID:
        #        # Draw rectangle on display_frame using bbox coordinates
        #        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #        cv2.putText(display_frame, f"Person: {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Display the frame
        cv2.imshow("Dan@Work Debug Output", display_frame)

        # Check for 'q' key press to exit
        key = cv2.waitKey(1) & 0xFF # waitKey(1) is crucial for imshow to refresh
        if key == ord('q'):
            print("\n[INFO] 'q' pressed, exiting...")
            break

    # --- Cleanup ---
    print("\n[INFO] Cleaning up...")
    if picam2:
        picam2.stop()
        print("[INFO] Camera stopped.")
    cv2.destroyAllWindows()
    print("[INFO] OpenCV windows closed.")
    # <<< Add Hailo cleanup if necessary, e.g., device.release() >>>
    print("[INFO] Cleanup complete.")

else:
    print("[ERROR] Failed to initialize. Exiting.")
