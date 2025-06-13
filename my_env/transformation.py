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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    "218622277783": {   # Camera 1  
        "qw": 0.2753382259432706,
        "qx": -0.9507914491110444,
        "qy": 0.10163352731715625,
        "qz": -0.0992728953783751,
        "x": -0.3143201729315735,
        "y": -0.2818959803878928,
        "z": 0.4054716586599699,
    },
    "130322272869": {   # Camera 2
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
    # Only apply transformation to camera1 and camera2
    cameras_to_transform = ["130322272869", "218622277783", "819612070593"]  # Camera 1 and Camera 2
    
    # 创建可视化
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 相机颜色和名称
    camera_colors = ['red', 'green', 'blue']
    camera_names = ['Camera 1 (218622277783)', 'Camera 2 (130322272869)', 'Camera 3 (819612070593)']
    
    for idx, (camera_name, p) in enumerate(CAMERA_PARAMS.items()):
        H = homogeneous_from_quat(**p)
        
        # Only apply OPENCV_CAM_TO_BLENDER_CAM_FRAME_TRANSFORM to specified cameras
        if camera_name in cameras_to_transform:
            # Assuming H is World->OpenCVCamera, convert to World->BlenderCamera for script input
            H = H @ OPENCV_CAM_TO_BLENDER_CAM_FRAME_TRANSFORM
            
        
        # 可视化部分
        pos = H[:3, 3]  # 相机位置
        
        # 相机坐标系的三个轴向量（旋转矩阵的列）
        x_axis = H[:3, 0] * 0.1  # X轴 (红色)
        y_axis = H[:3, 1] * 0.1  # Y轴 (绿色)
        z_axis = H[:3, 2] * 0.1  # Z轴 (蓝色)
        
        color = camera_colors[idx]
        
        # 绘制相机位置，标记是否应用了变换
        transform_status = " (Transformed)" if camera_name in cameras_to_transform else " (Original)"
        ax.scatter(pos[0], pos[1], pos[2], 
                  c=color, s=100, alpha=0.8, 
                  label=camera_names[idx] + transform_status)
        
        # 绘制坐标轴
        # X轴 (红色)
        ax.quiver(pos[0], pos[1], pos[2],
                 x_axis[0], x_axis[1], x_axis[2],
                 color='red', alpha=0.7, arrow_length_ratio=0.1)
        
        # Y轴 (绿色)
        ax.quiver(pos[0], pos[1], pos[2],
                 y_axis[0], y_axis[1], y_axis[2],
                 color='green', alpha=0.7, arrow_length_ratio=0.1)
        
        # Z轴 (蓝色)
        ax.quiver(pos[0], pos[1], pos[2],
                 z_axis[0], z_axis[1], z_axis[2],
                 color='blue', alpha=0.7, arrow_length_ratio=0.1)
        
        # 添加相机标签
        ax.text(pos[0], pos[1], pos[2] + 0.05, 
               f'Cam{idx+1}', fontsize=10, color=color)
        
        # H = np.linalg.inv(H)
        block[camera_name] = {"X_WT": format_matrix(H)}
    
    # 绘制世界坐标系原点
    ax.scatter(0, 0, 0, c='black', s=150, alpha=1.0, marker='o', label='World Origin')
    
    # 绘制世界坐标系轴
    axis_length = 0.15
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', alpha=0.9, arrow_length_ratio=0.1, linewidth=3)
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', alpha=0.9, arrow_length_ratio=0.1, linewidth=3)
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', alpha=0.9, arrow_length_ratio=0.1, linewidth=3)
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Camera Coordinate Systems Visualization\nRed=X-axis, Green=Y-axis, Blue=Z-axis')
    
    # 设置相等的轴比例
    max_range = 0.6
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range])
    
    # 添加图例
    ax.legend()
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return block



# ---------------- 主入口 ---------------- #
if __name__ == "__main__":
    result = cameras_to_json_block()  # 三台 eye-on-base 相机

    # 写入 json
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"[{datetime.now().isoformat(timespec='seconds')}] 已生成 {OUTPUT_JSON} (含可视化)")
