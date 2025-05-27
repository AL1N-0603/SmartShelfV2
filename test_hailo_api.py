#!/usr/bin/env python3
"""
Test script to check the correct Hailo API usage
"""

import hailo_platform as hailo
import sys

print(f"Hailo Platform version: {hailo.__version__ if hasattr(hailo, '__version__') else 'Unknown'}")
print("\nAvailable classes and functions:")

# List all available attributes in hailo module
attrs = dir(hailo)
for attr in sorted(attrs):
    if not attr.startswith('_'):
        print(f"  - {attr}")

# Try to find the correct class names
print("\n\nTrying to initialize Hailo device...")

try:
    # Try VDevice (new API)
    print("Trying VDevice...")
    device = hailo.VDevice()
    print("✓ VDevice works!")
    device.release()
except Exception as e:
    print(f"✗ VDevice failed: {e}")

try:
    # Try Device
    print("\nTrying Device...")
    device = hailo.Device()
    print("✓ Device works!")
    device.release()
except Exception as e:
    print(f"✗ Device failed: {e}")

# Check for HEF/Hef
print("\n\nChecking HEF class...")
if hasattr(hailo, 'HEF'):
    print("✓ HEF class exists")
elif hasattr(hailo, 'Hef'):
    print("✓ Hef class exists")
else:
    print("✗ Neither HEF nor Hef found")

# Check for ConfigureParams
print("\nChecking ConfigureParams...")
if hasattr(hailo, 'ConfigureParams'):
    print("✓ ConfigureParams exists")
else:
    print("✗ ConfigureParams not found")
    if hasattr(hailo, 'ConfigureNetworkParams'):
        print("  → Found ConfigureNetworkParams instead")

# Check for VStreams
print("\nChecking VStream classes...")
for vstream_type in ['InputVStreamParams', 'OutputVStreamParams', 'InferVStreams']:
    if hasattr(hailo, vstream_type):
        print(f"✓ {vstream_type} exists")
    else:
        print(f"✗ {vstream_type} not found")

# Check interface types
print("\nChecking interface types...")
if hasattr(hailo, 'HailoStreamInterface'):
    print("✓ HailoStreamInterface exists")
    if hasattr(hailo.HailoStreamInterface, 'PCIe'):
        print("  → PCIe interface available")
else:
    print("✗ HailoStreamInterface not found")
