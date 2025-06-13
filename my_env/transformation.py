"""
生成相机外参文件 cameras_tf.json：
{
    "camera1": { "X_WT": [...] },   # Camera 1 (eye-on-base)
    "camera2": { "X_WT": [...] },   # Camera 2 (eye-on-base)
    "camera3": { "X_WT": [...] }    # Camera 3 (eye-on-base)
}
"""

import math
import json
import numpy as np
from datetime import datetime

OUTPUT_JSON = "cameras_tf.json"

# ------------- 公共数学工具函数 ------------- #
def quat_to_rotmat(qw, qx, qy, qz):
    """(qw,qx,qy,qz) → 3×3 旋转矩阵（ROS xyzw 顺序）"""
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w),   1-2*(x*x+y*y)],
    ])

def homogeneous_from_quat(qw, qx, qy, qz, x, y, z):
    H = np.eye(4)
    H[:3, :3] = quat_to_rotmat(qw, qx, qy, qz)
    H[:3,  3] = [x, y, z]
    return H

def format_matrix(mat, precision=12):
    """numpy 4×4 → Python list，保留 precision 位小数"""
    return [[round(float(v), precision) for v in row] for row in mat]

# Define the transformation matrix from OpenCV camera frame to Blender camera frame convention
# This matrix is used because the downstream scripts (like simple_body_builder.py)
# expect the input X_WC from cameras_tf.json to be in a "Blender camera" convention,
# such that their internal hardcoded transform (X_WC @ diag(1,-1,-1,1)) results in an OpenCV camera pose.
# So, if our current H is World->OpenCV_Cam, we need World->Blender_Cam.
# X_W_BlenderCam = X_W_OpenCVCam @ BlenderToOpenCV_Frame_Transfom_Inverse
# where BlenderToOpenCV_Frame_Transform is diag(1,-1,-1,1). This matrix is its own inverse.
OPENCV_CAM_TO_BLENDER_CAM_FRAME_TRANSFORM = np.array([
    [1.0,  0.0,  0.0,  0.0],
    [0.0,  -1.0,  0.0,  0.0],
    [0.0,  0.0,  -1.0,  0.0],
    [0.0,  0.0,  0.0,  1.0]
])

# ---------------- 处理三个 eye-on-base 相机 ---------------- #
# 直接把 easy_handeye/yaml 的四元数和平移抄进来
CAMERA_PARAMS = {
    "130322272869": {   # Camera 1
        "qw": 0.2753382259432706,
        "qx": -0.9507914491110444,
        "qy": 0.10163352731715625,
        "qz": -0.0992728953783751,
        "x": -0.3143201729315735,
        "y": -0.2818959803878928,
        "z": 0.4054716586599699,
    },
    "218622277783": {   # Camera 2
        "qw": 0.03720964521903238,
        "qx": -0.11719798735462308,
        "qy": 0.9587194572328736, 
        "qz": -0.2563924265375323,
        "x": -0.25474907532012264,
        "y": 0.2684911723287354,
        "z": 0.4021614244269471,
    },
    "819612070593": {   # Camera 3
        "qw": 0.15942332343944432,
        "qx": 0.6286470211203463,
        "qy": 0.7532848101985787,
        "qz": 0.10931203732493776,
        "x": -0.3254305537853347,
        "y": -0.03702118311658656,
        "z": 0.5819423743968877,
    },
}

def cameras_to_json_block():
    block = {}
    for camera_name, p in CAMERA_PARAMS.items():
        H = homogeneous_from_quat(**p)
        # Assuming H is World->OpenCVCamera, convert to World->BlenderCamera for script input
        H = H @ OPENCV_CAM_TO_BLENDER_CAM_FRAME_TRANSFORM
        # Inverse the final transformation matrix
        H = np.linalg.inv(H)
        block[camera_name] = {"X_WT": format_matrix(H)}
    return block

# ---------------- 主入口 ---------------- #
if __name__ == "__main__":
    result = cameras_to_json_block()  # 三台 eye-on-base 相机

    # 写入 json
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"[{datetime.now().isoformat(timespec='seconds')}] 已生成 {OUTPUT_JSON}")
