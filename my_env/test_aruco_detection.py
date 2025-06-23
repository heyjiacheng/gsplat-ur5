#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_aruco_detection.py  ——  测试ArUco检测是否正常工作
用法：python test_aruco_detection.py
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import time
from typing import Optional

# ArUco 参数
ARUCO_DICT = cv2.aruco.DICT_ARUCO_ORIGINAL
ARUCO_SIZE = 0.05  # ArUco标记实际尺寸（米）

def test_camera_and_aruco():
    """测试相机连接和ArUco检测"""
    
    # 初始化ArUco检测器
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # 查找相机
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("❌ 未找到RealSense相机")
        return
    
    print(f"✅ 找到 {len(devices)} 台RealSense相机:")
    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        print(f"  {i+1}. {name} (序列号: {serial})")
    
    # 使用第一台相机进行测试
    first_device = devices[2]
    serial = first_device.get_info(rs.camera_info.serial_number)
    
    print(f"\n🔧 测试相机: {serial}")
    
    # 配置相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        # 启动相机
        profile = pipeline.start(config)
        
        # 获取相机内参
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float32)
        
        print(f"  📷 分辨率: {intrinsics.width}x{intrinsics.height}")
        print(f"  🔍 焦距: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
        
        print("\n🎯 开始ArUco检测测试...")
        print("请将ArUco标记放在相机前，按 'q' 退出, 按 's' 保存图像")
        
        frame_count = 0
        detection_count = 0
        
        while True:
            try:
                # 获取图像
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # 转换为numpy数组
                image = np.asanyarray(color_frame.get_data())
                frame_count += 1
                
                # 检测ArUco
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)
                
                # 绘制检测结果
                display_image = image.copy()
                
                if ids is not None and len(ids) > 0:
                    detection_count += 1
                    
                    # 绘制标记
                    cv2.aruco.drawDetectedMarkers(display_image, corners, ids)
                    
                    # 尝试估计姿态
                    try:
                        # 尝试不同的API
                        success = False
                        try:
                            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                                corners, ARUCO_SIZE, camera_matrix, dist_coeffs
                            )
                            success = True
                        except AttributeError:
                            try:
                                rvecs, tvecs, _ = cv2.estimatePoseSingleMarkers(
                                    corners, ARUCO_SIZE, camera_matrix, dist_coeffs
                                )
                                success = True
                            except AttributeError:
                                # 手动计算
                                half_size = ARUCO_SIZE / 2.0
                                object_points = np.array([
                                    [-half_size, -half_size, 0],
                                    [half_size, -half_size, 0],
                                    [half_size, half_size, 0],
                                    [-half_size, half_size, 0]
                                ], dtype=np.float32)
                                
                                success_pnp, rvec, tvec = cv2.solvePnP(
                                    object_points, 
                                    corners[0].reshape(-1, 2), 
                                    camera_matrix, 
                                    dist_coeffs
                                )
                                
                                if success_pnp:
                                    rvecs = [rvec.flatten()]
                                    tvecs = [tvec.flatten()]
                                    success = True
                        
                        if success:
                            # 绘制坐标轴
                            for i in range(len(rvecs)):
                                cv2.drawFrameAxes(display_image, camera_matrix, dist_coeffs, 
                                                rvecs[i], tvecs[i], 0.03)
                            
                            # 显示位置信息
                            pos_text = f"Pos: ({tvecs[0][0]:.3f}, {tvecs[0][1]:.3f}, {tvecs[0][2]:.3f})"
                            cv2.putText(display_image, pos_text, (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    except Exception as e:
                        print(f"姿态估计失败: {e}")
                    
                    # 显示检测到的标记ID
                    id_text = f"ID: {ids.flatten()}"
                    cv2.putText(display_image, id_text, (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # 显示统计信息
                stats_text = f"Frame: {frame_count}, Detected: {detection_count}"
                cv2.putText(display_image, stats_text, (10, display_image.shape[0] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 显示图像
                cv2.imshow(f"ArUco Test - Camera {serial[:8]}", display_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"aruco_test_{serial}_{int(time.time())}.jpg"
                    cv2.imwrite(filename, display_image)
                    print(f"保存图像: {filename}")
                
            except Exception as e:
                print(f"处理帧时出错: {e}")
                time.sleep(0.1)
        
        print(f"\n📊 测试结果:")
        print(f"  总帧数: {frame_count}")
        print(f"  检测成功: {detection_count}")
        if frame_count > 0:
            success_rate = (detection_count / frame_count) * 100
            print(f"  成功率: {success_rate:.1f}%")
        
    except Exception as e:
        print(f"❌ 相机测试失败: {e}")
    
    finally:
        try:
            pipeline.stop()
            cv2.destroyAllWindows()
        except:
            pass
        print("🔧 相机资源已释放")

if __name__ == "__main__":
    print("ArUco检测测试工具")
    print("=" * 30)
    test_camera_and_aruco() 