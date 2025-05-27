#!/usr/bin/env python3
import hailo_platform as hailo
import numpy as np

# Initialize Hailo device
target = hailo.PcieDevice()

# Load the model
hef_path = "yolov8s.hef"
hef = hailo.Hef(hef_path)

# Configure the device
configure_params = hailo.ConfigureParams.create_from_hef(hef, interface=hailo.HailoStreamInterface.PCIe)
network_group = target.configure(hef, configure_params)[0]

# Get input/output info
input_vstreams_params = hailo.InputVStreamParams.make(network_group)
output_vstreams_params = hailo.OutputVStreamParams.make(network_group)

print("Hailo-8 initialized successfully!")
print(f"Model: {hef_path}")
print(f"Input shape: {input_vstreams_params[0].shape}")
