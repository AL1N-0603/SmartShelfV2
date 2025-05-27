#!/usr/bin/env python3
"""
Simple test script to verify Hailo pipeline is working
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import hailo
import sys

def test_hailo_import():
    """Test if Hailo modules can be imported"""
    print("Testing Hailo imports...")
    
    try:
        from hailo_apps_infra.hailo_rpi_common import (
            get_caps_from_pad,
            get_numpy_from_buffer,
            app_callback_class,
        )
        print("✓ hailo_rpi_common imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import hailo_rpi_common: {e}")
        return False
        
    try:
        from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp
        print("✓ GStreamerDetectionApp imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import GStreamerDetectionApp: {e}")
        return False
        
    return True

def test_gstreamer():
    """Test GStreamer initialization"""
    print("\nTesting GStreamer...")
    
    try:
        Gst.init(None)
        print("✓ GStreamer initialized successfully")
        
        # Check for Hailo elements
        if Gst.ElementFactory.find("hailonet"):
            print("✓ hailonet element found")
        else:
            print("✗ hailonet element not found - check Hailo GStreamer plugins installation")
            
        if Gst.ElementFactory.find("hailofilter"):
            print("✓ hailofilter element found")
        else:
            print("✗ hailofilter element not found")
            
        return True
    except Exception as e:
        print(f"✗ GStreamer initialization failed: {e}")
        return False

def test_hailo_device():
    """Test Hailo device detection"""
    print("\nTesting Hailo device...")
    
    try:
        # Check if we can detect Hailo device through lspci
        import subprocess
        result = subprocess.run(['lspci'], capture_output=True, text=True)
        if 'Hailo' in result.stdout:
            print("✓ Hailo device detected via lspci")
        else:
            print("? Hailo device not found via lspci (might still work)")
            
        return True
    except Exception as e:
        print(f"! Could not run lspci: {e}")
        return True  # Not critical

def main():
    print("=" * 60)
    print("Hailo Pipeline Test")
    print("=" * 60)
    
    all_good = True
    
    # Test imports
    if not test_hailo_import():
        all_good = False
        print("\n⚠️  Some imports failed. You may need to:")
        print("  1. Install hailo-apps-infra:")
        print("     git clone https://github.com/hailo-ai/hailo-apps.git")
        print("     cd hailo-apps")
        print("     pip install -e .")
        print("  2. Make sure PYTHONPATH includes the hailo-apps directory")
        
    # Test GStreamer
    if not test_gstreamer():
        all_good = False
        print("\n⚠️  GStreamer issues detected. You may need to:")
        print("  1. Install GStreamer Hailo plugins")
        print("  2. Set GST_PLUGIN_PATH to include Hailo plugins")
        
    # Test device
    test_hailo_device()
    
    print("\n" + "=" * 60)
    if all_good:
        print("✅ All tests passed! You should be able to run the YOLOv8 detection script.")
    else:
        print("⚠️  Some tests failed. Please fix the issues above before running detection.")
    print("=" * 60)

if __name__ == "__main__":
    main()
