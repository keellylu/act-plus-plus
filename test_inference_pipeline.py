#!/usr/bin/env python3
"""Test inference pipeline to find where single-channel image originates"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import torch
from torchvision import transforms

# Simulate the image processing pipeline from inference_server.py

def test_image_pipeline():
    """Simulate the exact image processing from inference_server.py"""

    # Create fake camera images (like what we got from the test)
    zed_rgb = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    kiwi_rgb = np.random.randint(0, 256, (720, 960, 3), dtype=np.uint8)

    images = {
        'zed_camera': zed_rgb,
        'arm_camera': kiwi_rgb
    }

    print("Initial images:")
    for cam_name, img in images.items():
        print(f"  {cam_name}: shape={img.shape}, dtype={img.dtype}, channels={img.shape[2] if len(img.shape) == 3 else 'N/A'}")

    # Now run through the exact pipeline from inference_server.py lines 256-284
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_tensors = []

    for cam_name, img in images.items():
        print(f"\n[{cam_name}]")
        print(f"  1. After retrieval: shape={img.shape}, channels={img.shape[2] if len(img.shape) == 3 else 'N/A'}")

        # Ensure image has 3 channels (RGB)
        if len(img.shape) == 2:
            print(f"  2. Converting 2D grayscale to RGB")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            print(f"  2. Converting 1-channel to RGB")
            img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            print(f"  2. Converting RGBA to RGB")
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            print(f"  2. Already 3 channels, no conversion needed")

        print(f"  3. After channel check: shape={img.shape}, channels={img.shape[2] if len(img.shape) == 3 else 'N/A'}")

        # Resize to training dimensions (480, 640)
        if img.shape[:2] != (480, 640):
            print(f"  4. Resizing from {img.shape[:2]} to (480, 640)")
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
        else:
            print(f"  4. Already correct size")

        print(f"  5. After resize: shape={img.shape}, channels={img.shape[2] if len(img.shape) == 3 else 'N/A'}")

        # Normalize to [0, 1]
        img_normalized = img.astype(np.float32) / 255.0
        print(f"  6. After normalization to [0,1]: shape={img_normalized.shape}")

        # Transpose to (C, H, W)
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float()
        print(f"  7. After transpose to (C,H,W): shape={img_tensor.shape}")

        # Apply ImageNet normalization
        img_tensor = normalize(img_tensor)
        print(f"  8. After ImageNet norm: shape={img_tensor.shape}")

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        print(f"  9. After unsqueeze(0): shape={img_tensor.shape}")

        image_tensors.append(img_tensor)

    # Concatenate all camera images
    print(f"\n[Concatenation]")
    print(f"  ZED tensor shape: {image_tensors[0].shape}")
    print(f"  Kiwi tensor shape: {image_tensors[1].shape}")

    image_tensor = torch.cat(image_tensors, dim=1)
    print(f"  Final concatenated tensor: shape={image_tensor.shape}")
    print(f"  Expected: (1, 6, 480, 640) for 2 cameras with 3 channels each")

    if image_tensor.shape == (1, 6, 480, 640):
        print("\n✅ SUCCESS! Image pipeline produces correct shape!")
    else:
        print(f"\n❌ FAILED! Expected (1, 6, 480, 640) but got {image_tensor.shape}")

if __name__ == '__main__':
    test_image_pipeline()
