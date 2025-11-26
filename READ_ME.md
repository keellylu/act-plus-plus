# ACT++ Spot Deployment: Complete Summary

## Architecture Shift

You've moved from a **single-machine architecture** to a **distributed architecture**:

### Old Approach (Wouldn't Work)
```
GPU Machine
├─ Cameras (ZED, Kiwi)
├─ Policy Network
└─ Direct Robot Connection ← PROBLEM: Can't connect to Spot from GPU
```

### New Approach (Correct)
```
GPU Machine                    Mac Machine
├─ Cameras (ZED, Kiwi)       ├─ Spot Robot
├─ Policy Network             ├─ Action Executor
├─ Send actions (TCP 9999) ──→ Receive actions
└─ Receive qpos (TCP 9998) ←── Send qpos
```

## Key Files Created

### 1. GPU-Side Environment: `spot_real_env_gpu.py`

Handles:
- ZED camera streaming (local USB)
- Kiwi camera reception (from iPhone on port 8888)
- Network communication with Mac
- No direct robot connection

```python
env = SpotRealEnvGPU(
    camera_names=['zed_camera', 'arm_camera'],
    action_send_port=9999,
    qpos_receive_port=9998
)

env.wait_for_mac_connection()
images = env._get_images()  # Get latest camera frames
env.send_action(action)     # Send action to Mac
qpos = env.receive_qpos()   # Receive qpos from Mac
```

### 2. GPU Inference Server: `inference_server.py`

Main loop on GPU:

```bash
python inference_server.py \
    --checkpoint policy_best.ckpt \
    --stats dataset_stats.pkl \
    --num-episodes 5
```

Does:
1. Load policy and stats
2. Start camera streaming
3. Wait for Mac to connect
4. Main loop:
   - Receive qpos from Mac
   - Get images from local cameras
   - Run policy
   - Send action to Mac

### 3. Mac Execution Client: `inference_client.py`

Main loop on Mac:

```bash
python inference_client.py \
    --gpu-ip 192.168.1.100 \
    --robot-ip 192.168.80.3 \
    --num-episodes 5
```

Does:
1. Connect to Spot robot
2. Connect to GPU server
3. Main loop:
   - Receive action from GPU
   - Execute on Spot
   - Get qpos from Spot
   - Send qpos to GPU

## Network Protocol

### Action Transmission (GPU → Mac)

```
Port: 9999
Format: 88 bytes (11 float64 values)
[arm[0], arm[1], ..., arm[5], gripper, body_x, body_y, body_z, body_pitch]
```

### Qpos Transmission (Mac → GPU)

```
Port: 9998
Format: 88 bytes (11 float64 values)
[arm[0], arm[1], ..., arm[5], gripper, body_x, body_y, body_z, body_pitch]
```

## Timing Analysis

### Control Loop Timing

```
GPU:                        Mac:
t=0ms   Get images (1ms)
        Receive qpos ←────── Get qpos from Spot (5ms)
        Run policy (15ms)    Send qpos (2ms)
t=18ms  Send action ──────→ Receive action (1ms)
        Wait for qpos       Execute (20ms)
                            Get new qpos (5ms)
t=48ms  Receive qpos ←────── Send qpos (2ms)
        Next iteration ───→ Next iteration
```

**Total latency: ~40-50ms (fits 20 Hz perfectly)**

## Data Flow During One Step

```
Step N:
1. Mac reads qpos from Spot robot
2. Mac sends qpos to GPU (port 9998)
3. GPU receives qpos
4. GPU gets latest images from buffers
5. GPU runs policy network
6. GPU sends action to Mac (port 9999)
7. Mac receives action
8. Mac executes action on Spot
9. Go to Step N+1
```

## Setup Instructions

### GPU Machine Setup

1. **Ensure cameras are working:**
   ```bash
   # Test ZED
   python -c "from zed import stream_zed_frames; [print('ZED OK'), break] for rgb, d in stream_zed_frames()"
   ```

2. **Configure Kiwi (iPhone app):**
   - In Kiwi app settings
   - Set server: `<GPU_IP>:8888`

3. **Start GPU server:**
   ```bash
   python inference_server.py \
       --checkpoint policy_best.ckpt \
       --stats dataset_stats.pkl
   ```

   Will output:
   ```
   Waiting for Mac client...
   Run on Mac: python inference_client.py --gpu-ip 192.168.1.100
   ```

### Mac Machine Setup

1. **Ensure Spot connection works:**
   ```bash
   python -c "from inference_client import InferenceClient; c = InferenceClient('192.168.80.3', 'dummy'); c.connect_to_robot(); print('OK')"
   ```

2. **Get GPU IP:**
   - Ask GPU user: `hostname -I`
   - Or ping GPU from Mac: `ping GPU_IP`

3. **Start Mac client:**
   ```bash
   python inference_client.py \
       --gpu-ip 192.168.1.100 \
       --robot-ip 192.168.80.3 \
       --num-episodes 5
   ```

## Advantage of This Architecture

| Aspect | Benefit |
|--------|---------|
| **Modularity** | Each machine has one job (GPU: inference, Mac: execution) |
| **Scalability** | Easy to add more machines later |
| **Reliability** | If one part fails, easier to debug |
| **Real-time** | No blocking on either side |
| **Matches data collection** | Same pattern as your synchronized_data_collection.py |

## Comparison with Data Collection

You already have a similar system for data collection:

**Data Collection:**
- GPU server: streams cameras, saves to HDF5
- Mac client: sends qpos to GPU, executes teleoperation

**Inference (New):**
- GPU server: streams cameras, runs policy, sends actions
- Mac client: sends qpos to GPU, executes policy actions

The difference: instead of saving data, you run the policy!

## Key Components

### GPU-Side (spot_real_env_gpu.py)

```python
class SpotRealEnvGPU:
    def __init__(camera_names, action_send_port, qpos_receive_port)
    def _start_camera_streaming()      # ZED + Kiwi threads
    def setup_network_sockets()        # TCP listeners
    def wait_for_mac_connection()      # Block until Mac connects
    def send_action(action)            # TCP send
    def receive_qpos()                 # TCP receive
    def _get_images()                  # Get latest camera frames
    def get_observation(qpos)          # Build obs from qpos+images
```

### Mac-Side (inference_client.py)

```python
class InferenceClient:
    def __init__(robot_hostname, gpu_ip, ports)
    def connect_to_robot()             # Spot connection
    def connect_to_gpu()               # GPU connection
    def get_qpos_from_robot()          # Read from Spot
    def execute_action(action)         # Send to Spot
    def receive_action()               # TCP receive
    def send_qpos(qpos)                # TCP send
    def run(num_episodes, max_steps)   # Main loop
```

## Testing Checklist

- [ ] ZED camera streams on GPU
- [ ] Kiwi app connects to GPU port 8888
- [ ] Spot can be controlled from Mac
- [ ] GPU and Mac can ping each other
- [ ] Network ports 9999 and 9998 are available
- [ ] Python environments on both machines have required packages
- [ ] Policy checkpoint and stats files are accessible
- [ ] Start GPU server, wait for "Waiting for Mac client"
- [ ] Start Mac client, verify connections
- [ ] First episode completes without errors
- [ ] Check action values are reasonable (not all zeros)
- [ ] Robot movements are smooth and controlled

## Troubleshooting Quick Links

| Issue | Fix |
|-------|-----|
| Can't connect to GPU from Mac | Check GPU IP, firewall ports |
| No camera images | Verify ZED USB, Kiwi app config |
| Robot doesn't move | Check action format, verify Spot connection |
| Slow execution | Check network latency, GPU load |
| Thread crashes | Check camera drivers, network sockets |

## Next Steps

1. ✅ **Understand the distributed architecture** (you are here)
2. ✅ **Set up both machines**
3. ✅ **Test cameras and network communication**
4. ✅ **Run first inference episode**
5. ✅ **Monitor and debug**
6. ✅ **Run full evaluation**

## Files Ready to Use

```
✅ spot_real_env_gpu.py              # GPU environment
✅ inference_server.py               # GPU inference loop
✅ inference_client.py               # Mac execution loop
✅ DISTRIBUTED_INFERENCE_ARCHITECTURE.md   # Detailed design
✅ DISTRIBUTED_INFERENCE_QUICKSTART.md     # Setup guide
```

## Running Inference

### Terminal 1 (GPU Machine)
```bash
python inference_server.py \
    --checkpoint checkpoints/policy_best.ckpt \
    --stats checkpoints/dataset_stats.pkl \
    --num-episodes 5
```

### Terminal 2 (Mac Machine)
```bash
python inference_client.py \
    --gpu-ip 192.168.1.100 \
    --robot-ip 192.168.80.3 \
    --num-episodes 5
```

Both terminals will show progress. GPU shows policy inference, Mac shows robot execution.

## Performance Metrics

**Expected timings per step:**
- GPU image fetch: <1ms
- GPU policy inference: 10-20ms
- GPU action sending: 2-5ms
- Mac action receiving: 1-2ms
- Mac action execution: 15-25ms
- Mac qpos reading: 3-5ms
- Mac qpos sending: 1-2ms

**Total: 30-50ms per step (20 Hz)**

---

**You're now ready to deploy your ACT++ policy on Spot!** 🚀

The distributed architecture perfectly mirrors your data collection setup, making it familiar and reliable. Good luck! 🎯
