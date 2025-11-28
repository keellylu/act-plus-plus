#!/usr/bin/env python3
"""Quick test to check Kiwi camera output format"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_pkg.kiwi import start_kiwi_server, stream_kiwi_frames

print("Starting Kiwi camera test...")
print("Waiting for iPhone connection on port 8888...")

try:
    conn, addr = start_kiwi_server(host='0.0.0.0', port=8888)
    print(f"Connected from {addr}")

    for idx, rgb in enumerate(stream_kiwi_frames(conn, print_interval=0)):
        print(f"\n[Frame {idx}]")
        print(f"  RGB shape: {rgb.shape}")
        print(f"  RGB dtype: {rgb.dtype}")
        print(f"  RGB channels: {rgb.shape[2] if len(rgb.shape) == 3 else 'N/A'}")
        print(f"  RGB min/max: {rgb.min()}/{rgb.max()}")

        if idx >= 2:  # Just test first 3 frames
            break

    conn.close()

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")
