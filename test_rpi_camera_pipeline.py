#!/usr/bin/env python3
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
        print("\nStopping...")
    
    # Cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
