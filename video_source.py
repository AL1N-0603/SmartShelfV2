#!/usr/bin/env python3
"""
Check which video source is being used and test the camera
"""

import subprocess
import sys
import os

def check_camera_devices():
    """Check available camera devices"""
    print("Checking camera devices...")
    print("-" * 40)
    
    # Check for /dev/video* devices
    video_devices = []
    for i in range(10):
        device = f"/dev/video{i}"
        if os.path.exists(device):
            video_devices.append(device)
            print(f"Found: {device}")
    
    if not video_devices:
        print("No video devices found!")
    
    # Check if libcamera is available
    print("\nChecking libcamera...")
    try:
        result = subprocess.run(['libcamera-hello', '--list-cameras'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("libcamera devices:")
            print(result.stdout)
        else:
            print("libcamera-hello not found or no cameras detected")
    except:
        print("libcamera tools not installed")
    
    # Check v4l2 devices
    print("\nChecking v4l2 devices...")
    try:
        result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
    except:
        print("v4l2-ctl not installed")
    
    return video_devices

def test_gstreamer_pipelines():
    """Test different GStreamer pipeline configurations"""
    print("\n" + "=" * 60)
    print("Testing GStreamer pipelines...")
    print("=" * 60)
    
    # Test pipelines
    test_pipelines = [
        # RPi camera using libcamera
        ("RPi Camera (libcamera)", 
         "gst-launch-1.0 libcamerasrc ! video/x-raw,width=640,height=480 ! videoconvert ! autovideosink"),
        
        # V4L2 source (USB or RPi camera in V4L2 mode)
        ("V4L2 Camera", 
         "gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480 ! videoconvert ! autovideosink"),
        
        # Test pattern (should always work)
        ("Test Pattern", 
         "gst-launch-1.0 videotestsrc ! video/x-raw,width=640,height=480 ! videoconvert ! autovideosink")
    ]
    
    for name, pipeline in test_pipelines:
        print(f"\nTesting: {name}")
        print(f"Pipeline: {pipeline}")
        print("Press Ctrl+C to stop the test and continue to next...")
        
        try:
            subprocess.run(pipeline.split(), timeout=5)
            print(f"✓ {name} works!")
        except subprocess.TimeoutExpired:
            print(f"✓ {name} works! (stopped after 5 seconds)")
        except KeyboardInterrupt:
            print(f"✓ {name} interrupted by user")
        except Exception as e:
            print(f"✗ {name} failed: {e}")

def check_hailo_pipeline_source():
    """Check how hailo detection pipeline is configured"""
    print("\n" + "=" * 60)
    print("Checking Hailo pipeline configuration...")
    print("=" * 60)
    
    # Look for the detection pipeline source code
    try:
        import hailo_apps_infra.detection_pipeline as dp
        import inspect
        
        print("Detection pipeline module location:")
        print(f"  {dp.__file__}")
        
        # Try to find the source configuration
        if hasattr(dp, 'GStreamerDetectionApp'):
            print("\nGStreamerDetectionApp found")
            
            # Look for the run method
            if hasattr(dp.GStreamerDetectionApp, 'run'):
                source = inspect.getsource(dp.GStreamerDetectionApp.run)
                if 'libcamerasrc' in source:
                    print("  → Uses libcamerasrc for RPi camera")
                elif 'v4l2src' in source:
                    print("  → Uses v4l2src")
                else:
                    print("  → Camera source not found in run method")
    except Exception as e:
        print(f"Could not inspect detection pipeline: {e}")

def create_custom_pipeline_script():
    """Create a script with explicit camera source"""
    print("\n" + "=" * 60)
    print("Creating custom pipeline script...")
    print("=" * 60)
    
    custom_script = '''#!/usr/bin/env python3
"""
Custom detection script with explicit RPi camera source
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import sys

def create_pipeline():
    """Create a GStreamer pipeline with RPi camera source"""
    Gst.init(None)
    
    # Pipeline for RPi camera -> display
    # Modify this pipeline to add Hailo detection
    pipeline_str = """
        libcamerasrc ! 
        video/x-raw,width=640,height=480,framerate=30/1 ! 
        videoconvert ! 
        autovideosink
    """
    
    pipeline = Gst.parse_launch(pipeline_str)
    return pipeline

def main():
    print("Starting RPi camera test...")
    pipeline = create_pipeline()
    
    # Start playing
    pipeline.set_state(Gst.State.PLAYING)
    
    # Run main loop
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\\nStopping...")
    
    # Cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
'''
    
    with open('test_rpi_camera_pipeline.py', 'w') as f:
        f.write(custom_script)
    
    print("Created: test_rpi_camera_pipeline.py")
    print("Run it with: python test_rpi_camera_pipeline.py")

def main():
    print("Camera and Pipeline Diagnostic Tool")
    print("=" * 60)
    
    # Check camera devices
    devices = check_camera_devices()
    
    # Test GStreamer pipelines
    test_gstreamer_pipelines()
    
    # Check Hailo pipeline configuration
    check_hailo_pipeline_source()
    
    # Create custom pipeline script
    create_custom_pipeline_script()
    
    print("\n" + "=" * 60)
    print("Diagnostic complete!")
    print("\nTo ensure RPi camera is used:")
    print("1. The hailo detection app should use 'libcamerasrc' for RPi camera")
    print("2. Or use 'v4l2src device=/dev/video0' if camera is in V4L2 mode")
    print("3. Run the detection script with: --input rpi")
    print("=" * 60)

if __name__ == "__main__":
    main()
