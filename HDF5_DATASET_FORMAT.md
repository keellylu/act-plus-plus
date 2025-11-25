# HDF5 Dataset Format Guide

This guide explains how to format your custom HDF5 datasets for use with the ACT++ framework.

## Directory Structure

Store your datasets in the following directory structure:

```
dataset_dir/
├── episode_0.hdf5
├── episode_1.hdf5
├── episode_2.hdf5
└── ...
```

## HDF5 File Structure

Each episode HDF5 file should have the following structure:

```
episode_X.hdf5
├── attributes:
│   ├── sim (bool): Set to True for simulation data, False for real robot
│   └── compress (bool, optional): Set to True if images are JPEG compressed
├── observations/
│   ├── qpos (max_timesteps, 11): Robot state vectors
│   ├── qvel (max_timesteps, 11): Robot velocity vectors
│   └── images/
│       ├── arm_camera (max_timesteps, 480, 640, 3): First camera (uint8)
│       └── zed_camera (max_timesteps, 480, 640, 3): Second camera (uint8)
└── action (max_timesteps, 11): Action vectors
```

## Detailed Specifications

### Attributes

```python
import h5py

with h5py.File('episode_0.hdf5', 'w') as f:
    # Set metadata
    f.attrs['sim'] = True  # or False for real robot
    f.attrs['compress'] = False  # or True if using JPEG compression
```

### State Vectors (qpos and qvel)

**Shape**: `(max_timesteps, 11)` where 11 dimensions are:

1. **Indices 0-5**: Arm joint positions/velocities (6 joints)
2. **Index 6**: Gripper position/velocity (normalized 0-1, where 0=closed, 1=open)
3. **Index 7**: Body Z position/velocity
4. **Index 8**: Body velocity X
5. **Index 9**: Body velocity Y
6. **Index 10**: Body pitch angle/velocity

**Data Type**: `float64` (or `float32`)

**Example - qpos**:
```python
import numpy as np

max_timesteps = 400
qpos = np.zeros((max_timesteps, 11), dtype=np.float64)

# Timestep 0
qpos[0, :6] = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]  # arm joint positions
qpos[0, 6] = 0.5  # gripper position (0-1 normalized)
qpos[0, 7] = 0.5  # body z position
qpos[0, 8] = 0.0  # body vel x
qpos[0, 9] = 0.0  # body vel y
qpos[0, 10] = 0.0  # body pitch
```

### Actions

**Shape**: `(max_timesteps, 11)` - Same 11 dimensions as qpos

**Data Type**: `float64` (or `float32`)

**Notes**:
- Actions are the commanded/desired robot states
- Should include gripper actions in the 0-1 range (0=closed, 1=open)
- Body actions (indices 7-10) can be zeros if not applicable to your setup

### Images

**Shape**: `(max_timesteps, height, width, 3)` for each camera

**Data Type**: `uint8` (RGB image values 0-255)

**Default Resolution**: 480x640 (height x width)

**Cameras**:
- `arm_camera`: First external camera
- `zed_camera`: Second external camera (or replace with your camera names in constants.py)

**Example**:
```python
from PIL import Image
import numpy as np

max_timesteps = 400
arm_camera_images = np.zeros((max_timesteps, 480, 640, 3), dtype=np.uint8)
zed_camera_images = np.zeros((max_timesteps, 480, 640, 3), dtype=np.uint8)

# Load images
for t in range(max_timesteps):
    img = Image.open(f'path/to/arm_camera_frame_{t:04d}.jpg')
    arm_camera_images[t] = np.array(img)

    img = Image.open(f'path/to/zed_camera_frame_{t:04d}.jpg')
    zed_camera_images[t] = np.array(img)
```

## Creating an HDF5 File - Complete Example

```python
import h5py
import numpy as np
from PIL import Image

def create_episode_hdf5(output_path, max_timesteps=400, is_sim=True):
    """Create a complete HDF5 episode file"""

    with h5py.File(output_path, 'w') as root:
        # Set metadata
        root.attrs['sim'] = is_sim
        root.attrs['compress'] = False

        # Create observation group
        obs = root.create_group('observations')

        # Create qpos and qvel datasets
        qpos = obs.create_dataset('qpos', (max_timesteps, 11), dtype='float64')
        qvel = obs.create_dataset('qvel', (max_timesteps, 11), dtype='float64')

        # Create image group and datasets
        images = obs.create_group('images')
        arm_camera = images.create_dataset(
            'arm_camera',
            (max_timesteps, 480, 640, 3),
            dtype='uint8',
            chunks=(1, 480, 640, 3)  # Chunk by single frame for efficiency
        )
        zed_camera = images.create_dataset(
            'zed_camera',
            (max_timesteps, 480, 640, 3),
            dtype='uint8',
            chunks=(1, 480, 640, 3)
        )

        # Create action dataset
        action = root.create_dataset('action', (max_timesteps, 11), dtype='float64')

        # Now populate with your data
        # (This is where you load your actual data)

        # Example: Set some dummy data
        for t in range(max_timesteps):
            qpos[t] = np.random.randn(11)
            qvel[t] = np.random.randn(11)
            action[t] = np.random.randn(11)
            arm_camera[t] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            zed_camera[t] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Usage
create_episode_hdf5('episode_0.hdf5', max_timesteps=400, is_sim=True)
```

## Optional: Compressed Images

If you want to use JPEG compression to save disk space:

```python
import h5py
import cv2
import numpy as np

def create_compressed_hdf5(output_path, image_paths, qpos_data, qvel_data, action_data):
    """Create HDF5 with JPEG-compressed images"""

    max_timesteps = len(image_paths['arm_camera'])

    # JPEG compression settings
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Quality 50

    # Compress all images first
    compressed_arm = []
    compressed_zed = []
    max_len = 0

    for t in range(max_timesteps):
        # Compress arm_camera
        result, buf = cv2.imencode('.jpg', image_paths['arm_camera'][t], encode_param)
        compressed_arm.append(buf)
        max_len = max(max_len, len(buf))

        # Compress zed_camera
        result, buf = cv2.imencode('.jpg', image_paths['zed_camera'][t], encode_param)
        compressed_zed.append(buf)
        max_len = max(max_len, len(buf))

    # Pad to same length
    padded_arm = []
    padded_zed = []
    compress_len_arm = []
    compress_len_zed = []

    for buf in compressed_arm:
        padded = np.zeros(max_len, dtype='uint8')
        padded[:len(buf)] = buf
        padded_arm.append(padded)
        compress_len_arm.append(len(buf))

    for buf in compressed_zed:
        padded = np.zeros(max_len, dtype='uint8')
        padded[:len(buf)] = buf
        padded_zed.append(padded)
        compress_len_zed.append(len(buf))

    # Write to HDF5
    with h5py.File(output_path, 'w') as root:
        root.attrs['sim'] = True
        root.attrs['compress'] = True

        obs = root.create_group('observations')
        obs.create_dataset('qpos', data=qpos_data)
        obs.create_dataset('qvel', data=qvel_data)

        images = obs.create_group('images')
        images.create_dataset('arm_camera', data=padded_arm, dtype='uint8')
        images.create_dataset('zed_camera', data=padded_zed, dtype='uint8')

        root.create_dataset('action', data=action_data)
        root.create_dataset('compress_len', data=np.array([compress_len_arm, compress_len_zed]))
```

## Data Validation Checklist

Before using your datasets, verify:

- [ ] All episodes named as `episode_X.hdf5` (where X is 0-indexed)
- [ ] Each file has `sim` attribute set appropriately
- [ ] qpos shape: `(max_timesteps, 11)`
- [ ] qvel shape: `(max_timesteps, 11)`
- [ ] action shape: `(max_timesteps, 11)`
- [ ] Images are `uint8` with shape `(max_timesteps, 480, 640, 3)`
- [ ] Camera names match your config: `['arm_camera', 'zed_camera']`
- [ ] All timesteps have valid data (no NaNs or Infs)
- [ ] Action values are reasonable (gripper in 0-1 range)

## Dataset Directory Configuration

Update your constants.py to point to your dataset directory:

```python
SIM_TASK_CONFIGS = {
    'my_task': {
        'dataset_dir': '/path/to/your/dataset',  # Directory containing episode_0.hdf5, episode_1.hdf5, etc.
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['arm_camera', 'zed_camera']
    }
}
```

Then train with:
```bash
python imitate_episodes.py \
    --task_name my_task \
    --batch_size 8 \
    --num_steps 100000 \
    --policy_class ACT \
    --chunk_size 32
```

## Tips

1. **Chunking**: Use `chunks=(1, 480, 640, 3)` when creating image datasets for efficient single-frame access
2. **Data Types**: Stick with float64 for state/action, uint8 for images
3. **Compression**: JPEG compression saves ~90% disk space with minimal quality loss
4. **Episode Length**: Can vary, but typically 400-500 timesteps
5. **Normalization**: The training pipeline automatically computes and applies normalization statistics

Good luck with your data collection!
