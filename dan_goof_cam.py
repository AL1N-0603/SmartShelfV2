import streamlit as st
from picamera2 import Picamera2
import time
import numpy as np
import cv2
import os # For path handling if needed

# <<< ADAPT: Add necessary HailoRT imports from detection.py >>>
# Example: May include things like Device, HailoRTException, InferVStreams, ConfigureNetworkParams, FormatType, FormatOrder etc.
# from hailo_platform.runtime.pyhailort import * # Adjust based on actual imports in detection.py
try:
    # <<< COPY/ADAPT Hailo specific imports from detection.py >>>
    # Example: from hailo_platform.runtime.pyhailort import Device, InferVStreams, ConfigureNetworkParams, FormatType, FormatOrder, HailoRTException
    # <<< You might also need imports related to postprocessing used in detection.py >>>
    # Example: import numpy as np
    HAILO_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import Hailo libraries. Ensure SDK is installed: {e}")
    HAILO_AVAILABLE = False
    # st.stop() # Optional: Stop if Hailo is essential

# --- Configuration ---
HAILO_MODEL_WIDTH = 640 # Check if detection.py uses a different size
HAILO_MODEL_HEIGHT = 640
MAIN_FRAME_WIDTH = 1280
MAIN_FRAME_HEIGHT = 720

# <<< ADAPT: Get HEV path used in detection.py >>>
# This might be passed via args or hardcoded in detection.py
# Ensure this points to the correct .hev file (likely yolov8 related now)
HAILO_HEV_FILE = "/path/to/your/model.hev" # <<<=== FIND AND SET THIS PATH!

# <<< ADAPT: Get relevant parameters from detection.py >>>
PERSON_CLASS_ID = 0 # IMPORTANT: Find the correct class ID for 'person' in the YOLOv8 model output
DETECTION_THRESHOLD = 0.5 # Check threshold used in detection.py

# --- Streamlit App ---
st.set_page_config(page_title="Dan@Work Monitor", layout="centered")
st.title("Dan@Work Monitor")

# --- State Management ---
if 'initialized' not in st.session_state:
    # (Keep existing state variables: start_time, total_time_at_work, etc.)
    st.session_state.initialized = False
    st.session_state.start_time = None
    st.session_state.total_time_at_work = 0.0
    st.session_state.last_update_time = None
    st.session_state.person_present_last_check = False
    st.session_state.picam2 = None
    # Add state for Hailo objects
    st.session_state.hailo_network_group = None # To store the configured model object
    st.session_state.hailo_input_vstream_info = None # Optional: Store info if needed later
    st.session_state.hailo_output_vstream_info = None# Optional: Store info if needed later
    st.session_state.error = None

# --- Initialization (Camera & Hailo - Run Once) ---
if not st.session_state.initialized and st.session_state.error is None:
    try:
        # --- Camera Setup (Keep as is) ---
        st.write("Initializing Camera...")
        picam2 = Picamera2()
        # Check if detection.py requires a specific format (RGB? BGR?) for the lores stream
        config = picam2.create_preview_configuration(
            main={"size": (MAIN_FRAME_WIDTH, MAIN_FRAME_HEIGHT), "format": "RGB888"},
            lores={"size": (HAILO_MODEL_WIDTH, HAILO_MODEL_HEIGHT), "format": "YUV420"}, # YUV420 is common, check if model needs RGB/BGR
            controls={"FrameDurationLimits": (33333, 33333)} # ~30 FPS limit
        )
        picam2.configure(config)
        picam2.start()
        st.session_state.picam2 = picam2
        st.toast("Camera Initialized!", icon="✅")

        # --- Hailo Setup (Adapt from detection.py) ---
        if HAILO_AVAILABLE:
            st.write("Initializing Hailo AI Module...")
            # <<< ADAPT: Copy Hailo device scanning/selection logic from detection.py >>>
            # Example: devices = Device.scan(); device = Device(devices[0]) ...

            # <<< ADAPT: Copy HEV loading and network configuration logic from detection.py >>>
            # Example: params = ConfigureNetworkParams(...) # Check params used
            # Example: network_group = device.create_configured_network_group(HAILO_HEV_FILE, params)
            # Example: st.session_state.hailo_network_group = network_group

            # <<< Optional: Store input/output stream info if needed for preprocessing/postprocessing >>>
            # Example: st.session_state.hailo_input_vstream_info = network_group.get_input_vstream_infos()[0] # Assuming one input
            # Example: st.session_state.hailo_output_vstream_info = network_group.get_output_vstream_infos() # Get all outputs

            st.toast("Hailo Initialized!", icon="🧠")
        else:
             st.warning("Hailo library not found. Detection will not function.")

        st.session_state.start_time = time.monotonic()
        st.session_state.last_update_time = st.session_state.start_time
        st.session_state.initialized = True
        st.rerun() # Rerun to clear initialization messages and start the loop

    except Exception as e:
        st.session_state.error = f"Initialization Failed: {e}"
        st.error(st.session_state.error)
        if st.session_state.picam2: st.session_state.picam2.stop()
        # <<< Add Hailo cleanup if necessary, e.g., device.release() >>>
        st.stop()

# --- Main Application Loop ---
if st.session_state.initialized:
    picam2 = st.session_state.picam2
    network_group = st.session_state.hailo_network_group # Get Hailo model object

    # Placeholders for UI
    status_placeholder = st.empty()
    percentage_placeholder = st.empty()
    image_placeholder = st.empty()

    while True: # Main processing loop
        request = picam2.capture_request()
        main_frame = request.make_array("main") # For display (RGB888)
        lores_frame_yuv = request.make_array("lores") # For AI (YUV420 or configured format)
        request.release()

        person_detected_this_frame = False
        processed_detections = [] # To store final detections (e.g., [bbox, score, class_id])

        # --- Hailo Inference Pipeline (Adapt from detection.py) ---
        if HAILO_AVAILABLE and network_group:
            try:
                # 1. <<< ADAPT: Preprocessing logic from detection.py >>>
                # Convert lores_frame (likely YUV420) to the format expected by the model (RGB? BGR?)
                # Example: input_frame_rgb = cv2.cvtColor(lores_frame_yuv, cv2.COLOR_YUV420p_to_RGB) # Check correct YUV format
                # Apply any resizing, normalization, or data type changes used in detection.py
                # Ensure the final 'input_data' matches the model's input tensor requirements

                # 2. <<< ADAPT: Inference call from detection.py >>>
                # Example: Use network_group.infer_async() or network_group.infer()
                # Pass the preprocessed 'input_data'
                # Example: with network_group.infer_async(input_data) as infer_results:
                # Example:     raw_hailo_output = infer_results.get() # Or similar method

                # 3. <<< ADAPT: Postprocessing logic from detection.py >>>
                # This is often the most complex part. It takes raw_hailo_output (raw tensors)
                # and performs decoding, Non-Max Suppression (NMS), filtering etc.
                # It should produce a list of final detections.
                # Example function call: final_detections = postprocess_yolov8(raw_hailo_output, ...)
                # The result ('final_detections') might be a list of tuples/objects,
                # e.g., [(bbox, score, class_id), ...]
                # processed_detections = final_detections # Store for drawing later if needed

                # 4. Check for 'person' in the final detections
                for detection in processed_detections:
                    # <<< ADAPT: Extract class ID and score based on detection format >>>
                    # Example: bbox, score, class_id = detection # If detection is a tuple
                    # Example: class_id = detection['class_id']; score = detection['score'] # If dict
                    if class_id == PERSON_CLASS_ID and score > DETECTION_THRESHOLD:
                        person_detected_this_frame = True
                        break # Found a person, no need to check further

            except Exception as e:
                 # print(f"Hailo Pipeline Error: {e}") # Log error for debugging
                 pass # Allow loop to continue

        # --- Timing Update (Keep as is) ---
        current_time = time.monotonic()
        time_delta = current_time - st.session_state.last_update_time
        if st.session_state.person_present_last_check:
            st.session_state.total_time_at_work += time_delta
        total_elapsed_time = current_time - st.session_state.start_time
        work_percentage = (st.session_state.total_time_at_work / total_elapsed_time) * 100 if total_elapsed_time > 0 else 0.0
        st.session_state.last_update_time = current_time
        st.session_state.person_present_last_check = person_detected_this_frame

        # --- UI Update (Keep as is, optionally add drawing) ---
        if person_detected_this_frame:
            status_placeholder.markdown("## <span style='color:green'>✅ Dan at work</span>", unsafe_allow_html=True)
        else:
            status_placeholder.markdown("## <span style='color:red'>❌ Dan's goofing off</span>", unsafe_allow_html=True)
        percentage_placeholder.metric("Work Percentage Since Start", f"{work_percentage:.1f}%")

        # Optional: Draw bounding boxes on the main_frame before displaying
        # <<< ADAPT: Drawing logic from detection.py (using cv2.rectangle/putText) >>>
        # Example: Iterate through 'processed_detections' again
        # for bbox, score, class_id in processed_detections:
        #    if class_id == PERSON_CLASS_ID:
        #        # Draw rectangle on main_frame using bbox coordinates
        #        cv2.rectangle(main_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        image_placeholder.image(main_frame, caption="Live Feed", use_container_width=True) # Display frame

        time.sleep(0.01) # Shorter sleep might improve responsiveness

elif st.session_state.error:
    st.error(st.session_state.error)

