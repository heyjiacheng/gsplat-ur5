# gsplat-ur5

## Install:
```bash
git clone https://github.com/heyjiacheng/gsplat-ur5.git
```
```bash
pixi r build
```
sometimes may need reinstall opencv-python.

## Quick Start:
Get eye on base transformation from calibration
```bash
python transformation.py
```
get ground plane (first three numbers are normal vector, last one is distance)
```bash
python scripts/find_ground.py temp/ground_plane.json --extrinsics my_env/cameras_tf.json --visualize
```
gaussian splats the ground plane (no item on the ground plane)
```bash
python scripts/build_body_from_pointcloud.py objects/ground_body.json --extrinsics my_env/cameras_tf.json --points temp/ground_plane.npy --visualize
```
gausssian splats the item
```bash
python scripts/build_simple_body.py objects/tblock.json     --extrinsics my_env/cameras_tf.json     --ground scripts/example_ground_plane.json     --visualize
```
visulization
```bash
python scripts/visualize_object.py objects/tblock.json
```
real time tracking
```bash
python scripts/real_time_tracking.py objects --extrinsics my_env/cameras_tf.json --ground temp/ground_plane.json --visualize
```

## Modification of Original Code
original project: [Embodied Gaussian](https://github.com/bdaiinstitute/embodied_gaussians)

1. add gravity in tracking.

2. deleted camera exposure from get_datapoints_from_live_cameras.

## Need to do

1. calibrate camera 1
