# yolo-mono-depth

Viam camera module that turns a mono RGB camera into a point-cloud “lidar” using
[Ultralytics YOLO26 depth](https://docs.ultralytics.com/tasks/depth)
(`yolo26n-depth` by default). Compatible with nav-stack SLAM when wired as
`lidar` with `scan_source: point_cloud`.

Model: `viam-labs:yolo-mono-depth:camera` (`rdk:component:camera`).

## Configure

```json
{
  "name": "mono-lidar",
  "api": "rdk:component:camera",
  "model": "viam-labs:yolo-mono-depth:camera",
  "attributes": {
    "source": "cam",
    "model": "yolo26n-depth.pt",
    "backend": "auto",
    "imgsz": 416,
    "produce_hz": 10,
    "stride": 4,
    "scale": 1.0,
    "shm_name": "/viam-pc-mono",
    "movement_sensor": "imu"
  }
}
```

| Attribute | Notes |
| --- | --- |
| `source` | Required RGB camera name |
| `model` | Ultralytics weights or NCNN export dir (default `yolo26n-depth.pt`) |
| `backend` | `auto` / `pt` / `ncnn` |
| `imgsz` | Inference size (default `416`) |
| `produce_hz` | Producer rate (default `10`; drop to `5`–`7` on a Pi) |
| `stride` | Point subsample step (default `4`) |
| `scale` | Static metric multiplier |
| `movement_sensor` | Optional; online scale from wheel `linear_velocity` vs forward scan-match |
| `shm_name` | Optional POSIX shm writer (nav-stack / `viam-shared-memory-test` layout) |
| `fx`/`fy`/`cx`/`cy` | Intrinsics; or omit and use `hfov_deg` (default `70`) |

On a Raspberry Pi, set `"backend": "ncnn"`. You can omit `model` (defaults to `yolo26n-depth.pt`): the module downloads the weights if needed and **auto-exports** to `yolo26n-depth_ncnn_model/` on first load (slow once). Or export on a desktop and set `"model"` to that directory.

`run.sh` installs **CPU** PyTorch on Linux `aarch64` before Ultralytics so pip does not pull CUDA/`nvidia-cudnn` wheels (useless on a Pi). If a previous install already dragged those in, remove `viam-env/` and `.installed` and let the module reinstall.

Manual Pi venv (same order):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Point cloud frame is ROS camera_link style (**X forward**, Y left, Z up). Configure the SLAM lidar `mount` accordingly.

`GetImages` returns two JPEGs kept in sync with the point cloud:
- `color` — source RGB
- `depth` — metric depth colorized (near → purple/blue, far → yellow/red; black = invalid)

Camera DoCommand returns `scale`, `last_infer_ms`, `last_grab_ms`, `measured_hz`, frame counts; `set_scale` overrides the multiplier.

## Develop

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

## License

Apache-2.0
