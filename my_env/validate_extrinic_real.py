#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_extrinic_real.py  ——  连接真实相机验证Eye-to-Hand外参
用法：python validate_extrinic_real.py
需要安装: pip install pyrealsense2 opencv-python numpy scipy
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import time
from typing import Dict, List, Tuple, Optional
import argparse

# 你的外参数据
CAMERA_PARAMS = {
    "218622277783": {   # Camera 1  
        "qw": 0.23186468050094158,
        "qx": -0.9594348501025816,
        "qy": 0.11126460595906879,
        "qz": -0.11551504579753447,
        "x": -0.32427694851559585,
        "y": -0.28084946597353533,
        "z": 0.41925774261130094,
    },
    "130322272869": {   # Camera 2
        "qw": 0.06665864612025105,
        "qx": -0.160152624284172,
        "qy": 0.9519265051134965,
        "qz": -0.25247512886364,
        "x": -0.26168610660054314,
        "y": 0.2712285181679946,
        "z": 0.4070830794034206,
    },
    "819612070593": {   # Camera 3
        "qw": 0.15619147328055644,
        "qx": 0.6925084506280336,
        "qy": 0.6959352798619778,
        "qz": 0.10821439703958144,
        "x": -0.39398859088913385,
        "y": 0.01688281794116203,
        "z": 0.7609657167075485,
    },
}

# ArUco 参数
ARUCO_DICT = cv2.aruco.DICT_ARUCO_ORIGINAL
ARUCO_SIZE = 0.1  # ArUco标记实际尺寸（米），请根据实际情况修改

class RealSenseCamera:
    """RealSense相机管理类"""
    def __init__(self, serial_number: str):
        self.serial_number = serial_number
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 配置相机
        self.config.enable_device(serial_number)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 启动相机
        try:
            self.profile = self.pipeline.start(self.config)
            
            # 获取相机内参
            color_stream = self.profile.get_stream(rs.stream.color)
            self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            # 转换为OpenCV格式
            self.camera_matrix = np.array([
                [self.intrinsics.fx, 0, self.intrinsics.ppx],
                [0, self.intrinsics.fy, self.intrinsics.ppy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            self.dist_coeffs = np.array(self.intrinsics.coeffs, dtype=np.float32)
            
            print(f"相机 {serial_number} 初始化成功")
            print(f"  分辨率: {self.intrinsics.width}x{self.intrinsics.height}")
            print(f"  焦距: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}")
            
        except Exception as e:
            print(f"相机 {serial_number} 初始化失败: {e}")
            raise
    
    def get_frame(self) -> Optional[np.ndarray]:
        """获取一帧图像"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)  # 增加超时时间
            color_frame = frames.get_color_frame()
            if color_frame:
                return np.asanyarray(color_frame.get_data())
        except Exception as e:
            # 不打印每次获取失败的错误，只在debug模式下显示
            pass
        return None
    
    def stop(self):
        """停止相机"""
        self.pipeline.stop()

class CameraExtrinsicsValidator:
    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # 初始化相机
        self.cameras = {}
        self.transform_matrices = {}
        
        print("正在初始化相机...")
        available_cameras = self._find_available_cameras()
        
        for serial in CAMERA_PARAMS.keys():
            if serial in available_cameras:
                try:
                    self.cameras[serial] = RealSenseCamera(serial)
                    self.transform_matrices[serial] = self._params_to_transform_matrix(CAMERA_PARAMS[serial])
                except Exception as e:
                    print(f"跳过相机 {serial}: {e}")
            else:
                print(f"未找到相机 {serial}")
        
        if not self.cameras:
            raise Exception("未找到任何可用的相机")
        
        print(f"成功初始化 {len(self.cameras)} 台相机")
    
    def _find_available_cameras(self) -> List[str]:
        """查找可用的RealSense相机"""
        ctx = rs.context()
        devices = ctx.query_devices()
        serials = []
        for dev in devices:
            serials.append(dev.get_info(rs.camera_info.serial_number))
        return serials
    
    def _params_to_transform_matrix(self, params: Dict) -> np.ndarray:
        """将四元数+平移参数转换为4x4变换矩阵"""
        quat = [params['qx'], params['qy'], params['qz'], params['qw']]
        rotation = R.from_quat(quat)
        rotation_matrix = rotation.as_matrix()
        
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = [params['x'], params['y'], params['z']]
        
        return transform
    
    def detect_aruco_pose(self, image: np.ndarray, camera: RealSenseCamera) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """检测ArUco标记并估计其在相机坐标系中的位姿"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            # 使用第一个检测到的标记
            # 在新版本OpenCV中，estimatePoseSingleMarkers移动到了cv2命名空间
            try:
                # 尝试新版本API
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, ARUCO_SIZE, 
                    camera.camera_matrix, 
                    camera.dist_coeffs
                )
            except AttributeError:
                # 如果新版API不可用，尝试旧版API位置
                try:
                    rvecs, tvecs, _ = cv2.estimatePoseSingleMarkers(
                        corners, ARUCO_SIZE, 
                        camera.camera_matrix, 
                        camera.dist_coeffs
                    )
                except AttributeError:
                    # 手动计算姿态 (fallback方法)
                    return self._estimate_pose_manual(corners[0], camera)
            
            # 在图像上绘制坐标轴（可选，用于可视化）
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            for i in range(len(rvecs)):
                cv2.drawFrameAxes(image, camera.camera_matrix, camera.dist_coeffs, 
                                rvecs[i], tvecs[i], 0.03)
            
            return rvecs[0][0], tvecs[0][0]
        return None
    
    def _estimate_pose_manual(self, corners: np.ndarray, camera: RealSenseCamera) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """手动估计ArUco姿态（备用方法）"""
        # 定义ArUco标记的3D点（标记中心为原点）
        half_size = ARUCO_SIZE / 2.0
        object_points = np.array([
            [-half_size, -half_size, 0],
            [half_size, -half_size, 0],
            [half_size, half_size, 0],
            [-half_size, half_size, 0]
        ], dtype=np.float32)
        
        # 使用solvePnP估计姿态
        success, rvec, tvec = cv2.solvePnP(
            object_points, 
            corners.reshape(-1, 2), 
            camera.camera_matrix, 
            camera.dist_coeffs
        )
        
        if success:
            return rvec.flatten(), tvec.flatten()
        return None
    
    def transform_to_base_frame(self, tvec: np.ndarray, rvec: np.ndarray, camera_serial: str) -> np.ndarray:
        """将相机坐标系中的位姿转换到机器人基座坐标系"""
        # 将rvec转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # 构建ArUco在相机坐标系中的4x4变换矩阵
        aruco_in_cam = np.eye(4)
        aruco_in_cam[:3, :3] = rotation_matrix
        aruco_in_cam[:3, 3] = tvec
        
        # 转换到基座坐标系
        cam_to_base = self.transform_matrices[camera_serial]
        aruco_in_base = cam_to_base @ aruco_in_cam
        
        return aruco_in_base[:3, 3]  # 只返回位置
    
    def capture_and_validate(self, num_samples: int = 30, show_images: bool = False) -> Dict:
        """捕获多帧图像并验证外参"""
        results = {cam_id: [] for cam_id in self.cameras.keys()}
        
        print(f"开始采集数据，将采集 {num_samples} 帧")
        print("请确保ArUco标记在所有相机视野中，按 'q' 键提前退出")
        
        for i in range(num_samples):
            print(f"\r采集进度: {i+1}/{num_samples}", end="", flush=True)
            
            all_images = {}
            valid_detections = 0
            
            # 从所有相机获取图像
            for cam_id, camera in self.cameras.items():
                image = camera.get_frame()
                if image is not None:
                    all_images[cam_id] = image.copy()
                    
                    # 检测ArUco
                    pose_result = self.detect_aruco_pose(image, camera)
                    if pose_result is not None:
                        rvec, tvec = pose_result
                        base_position = self.transform_to_base_frame(tvec, rvec, cam_id)
                        results[cam_id].append(base_position)
                        valid_detections += 1
            
            # 显示图像（可选）
            if show_images and all_images:
                combined_image = self._combine_images(all_images)
                cv2.imshow("Camera Views", combined_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n用户中断采集")
                    break
            
            time.sleep(0.1)  # 短暂延时
        
        print("\n采集完成")
        
        if show_images:
            cv2.destroyAllWindows()
        
        return results
    
    def _combine_images(self, images: Dict[str, np.ndarray]) -> np.ndarray:
        """将多个相机图像合并到一个窗口中显示"""
        if not images:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        image_list = list(images.values())
        if len(image_list) == 1:
            return image_list[0]
        elif len(image_list) == 2:
            return np.hstack(image_list)
        else:  # 3个或更多
            # 第一行放两个，第二行放剩余的
            row1 = np.hstack(image_list[:2])
            if len(image_list) == 3:
                # 第三个图像需要调整尺寸来匹配
                img3_resized = cv2.resize(image_list[2], (row1.shape[1], image_list[2].shape[0]))
                return np.vstack([row1, img3_resized])
            else:
                row2 = np.hstack(image_list[2:4])
                return np.vstack([row1, row2])
    
    def analyze_results(self, results: Dict) -> None:
        """分析结果，找出有问题的相机"""
        print("\n" + "="*60)
        print("外参验证结果分析")
        print("="*60)
        
        # 计算每个相机的统计信息
        avg_positions = {}
        valid_cameras = []
        
        for cam_id, positions in results.items():
            if len(positions) > 0:
                positions_array = np.array(positions)
                avg_pos = np.mean(positions_array, axis=0)
                std_pos = np.std(positions_array, axis=0)
                avg_positions[cam_id] = avg_pos
                valid_cameras.append(cam_id)
                
                print(f"\n相机 {cam_id}:")
                print(f"  有效检测次数: {len(positions)}")
                print(f"  平均位置 (x,y,z): ({avg_pos[0]:.4f}, {avg_pos[1]:.4f}, {avg_pos[2]:.4f}) 米")
                print(f"  位置标准差:      ({std_pos[0]:.4f}, {std_pos[1]:.4f}, {std_pos[2]:.4f}) 米")
                print(f"  位置稳定性:      {np.linalg.norm(std_pos)*1000:.2f} 毫米")
            else:
                print(f"\n相机 {cam_id}: 未检测到ArUco标记")
        
        if len(valid_cameras) < 2:
            print("\n❌ 错误: 有效相机数量不足，无法进行比较分析")
            print("   请检查:")
            print("   1. ArUco标记是否在相机视野中")
            print("   2. 光照是否充足")
            print("   3. ArUco标记尺寸设置是否正确")
            return
        
        # 计算相机间的位置差异
        print(f"\n相机间位置差异分析:")
        print("-" * 40)
        
        distances = {}
        for i, cam1 in enumerate(valid_cameras):
            for j, cam2 in enumerate(valid_cameras[i+1:], i+1):
                diff = avg_positions[cam1] - avg_positions[cam2]
                distance = np.linalg.norm(diff)
                distances[(cam1, cam2)] = distance
                
                print(f"相机 {cam1[:8]}... vs {cam2[:8]}...:")
                print(f"  位置差异: ({diff[0]:.4f}, {diff[1]:.4f}, {diff[2]:.4f}) 米")
                print(f"  空间距离: {distance:.4f} 米 ({distance*100:.2f} 厘米)")
                print()
        
        # 问题诊断
        print("问题诊断:")
        print("-" * 20)
        
        max_distance = max(distances.values()) if distances else 0
        
        if max_distance > 0.05:  # 5厘米阈值
            suspicious_pair = [k for k, v in distances.items() if v == max_distance][0]
            print(f"⚠️  发现异常: 相机差异过大")
            print(f"   最大差异: {max_distance*100:.2f} 厘米")
            print(f"   涉及相机: {suspicious_pair[0][:8]}... 和 {suspicious_pair[1][:8]}...")
            
            # 进一步分析哪个相机更可能有问题
            if len(valid_cameras) >= 3:
                cam_avg_distances = {}
                for cam in valid_cameras:
                    distances_for_cam = []
                    for pair, dist in distances.items():
                        if cam in pair:
                            distances_for_cam.append(dist)
                    cam_avg_distances[cam] = np.mean(distances_for_cam)
                
                most_suspicious = max(cam_avg_distances, key=cam_avg_distances.get)
                print(f"   最可疑的相机: {most_suspicious[:8]}...")
                print(f"   该相机的平均差异: {cam_avg_distances[most_suspicious]*100:.2f} 厘米")
        elif max_distance > 0.02:  # 2厘米
            print(f"⚠️  注意: 存在中等程度差异")
            print(f"   最大差异: {max_distance*100:.2f} 厘米")
            print(f"   在可接受范围内，但建议进一步检查")
        else:
            print("✅ 外参验证通过")
            print(f"   最大差异: {max_distance*100:.2f} 厘米")
            print(f"   所有相机外参质量良好")
        
        # 给出建议
        print(f"\n建议:")
        print("-" * 10)
        if max_distance > 0.05:
            print("• 🔴 需要重新标定外参，特别是差异最大的相机")
            print("• 检查相机安装是否稳固，避免振动或松动")
            print("• 确认ArUco标记尺寸设置是否正确")
            print("• 检查标定时的数据质量")
        elif max_distance > 0.02:
            print("• 🟡 建议检查外参标定质量")
            print("• 可以尝试增加标定数据量")
            print("• 当前精度可用于大部分应用")
        else:
            print("• 🟢 外参标定质量优秀")
            print("• 可以放心用于高精度应用")
            print("• 建议定期进行验证")
    
    def cleanup(self):
        """清理资源"""
        for camera in self.cameras.values():
            camera.stop()
        print("相机资源已释放")

def main():
    parser = argparse.ArgumentParser(description="Eye-to-Hand相机外参验证工具")
    parser.add_argument("--samples", type=int, default=30, help="采集样本数量 (默认: 30)")
    parser.add_argument("--show", action="store_true", help="显示相机画面")
    parser.add_argument("--aruco-size", type=float, default=0.05, help="ArUco标记尺寸(米) (默认: 0.05)")
    
    args = parser.parse_args()
    
    global ARUCO_SIZE
    ARUCO_SIZE = args.aruco_size
    
    print("Eye-to-Hand 相机外参验证工具")
    print("=" * 50)
    print(f"ArUco标记尺寸: {ARUCO_SIZE} 米")
    print(f"采集样本数量: {args.samples}")
    print(f"显示相机画面: {'是' if args.show else '否'}")
    print()
    
    try:
        validator = CameraExtrinsicsValidator()
        
        # 显示外参信息
        print("\n当前外参配置:")
        for cam_id, params in CAMERA_PARAMS.items():
            if cam_id in validator.cameras:
                print(f"相机 {cam_id[:8]}...:")
                print(f"  位置: ({params['x']:.4f}, {params['y']:.4f}, {params['z']:.4f}) 米")
                quat = [params['qx'], params['qy'], params['qz'], params['qw']]
                euler = R.from_quat(quat).as_euler('xyz', degrees=True)
                print(f"  旋转: ({euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°)")
        print()
        
        # 开始验证
        results = validator.capture_and_validate(num_samples=args.samples, show_images=args.show)
        validator.analyze_results(results)
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        if 'validator' in locals():
            validator.cleanup()

if __name__ == "__main__":
    main() 