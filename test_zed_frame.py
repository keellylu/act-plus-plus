#!/usr/bin/env python3
"""Quick test to check ZED camera output format"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_pkg.zed import stream_zed_frames

print("Starting ZED camera test...")
try:
    for idx, (rgb_bgr, depth) in enumerate(stream_zed_frames()):
        print(f"\n[Frame {idx}]")
        print(f"  RGB BGR shape: {rgb_bgr.shape}")
        print(f"  RGB BGR dtype: {rgb_bgr.dtype}")
        print(f"  RGB BGR min/max: {rgb_bgr.min()}/{rgb_bgr.max()}")
        print(f"  Depth shape: {depth.shape}")
        print(f"  Depth dtype: {depth.dtype}")

        # Convert BGR to RGB like in the actual code
        import cv2
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        print(f"  RGB shape: {rgb.shape}")
        print(f"  RGB dtype: {rgb.dtype}")
        print(f"  RGB channels: {rgb.shape[2] if len(rgb.shape) == 3 else 'N/A'}")

        if idx >= 2:  # Just test first 3 frames
            break

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")
