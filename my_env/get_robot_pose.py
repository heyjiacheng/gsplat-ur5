#!/usr/bin/env python3
"""
获取当前机器人末端执行机构的姿态
输出平移坐标 (x, y, z) 和旋转坐标 (Rx, Ry, Rz)
"""

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
except ImportError:
    raise ImportError("请先 pip install ur_rtde")

ROBOT_IP = "192.168.1.60"

def get_robot_pose(ip):
    """实时读取 Base→Tool 的 6D Pose [x,y,z,Rx,Ry,Rz]"""
    rtde_control = RTDEControlInterface(ip)
    rtde_receive = RTDEReceiveInterface(ip)
    try:
        raw_pose = rtde_receive.getActualTCPPose()
        # 不对旋转角度取负号，直接使用原始值
        task_pose = raw_pose  # [x, y, z, Rx, Ry, Rz]
        return task_pose
    finally:
        rtde_control.stopScript()

if __name__ == "__main__":
    try:
        pose = get_robot_pose(ROBOT_IP)
        
        print("当前末端执行机构姿态:")
        print("=" * 40)
        print(f"平移坐标 (m):")
        print(f"  X: {pose[0]:.6f}")
        print(f"  Y: {pose[1]:.6f}")
        print(f"  Z: {pose[2]:.6f}")
        print()
        print(f"旋转坐标 (rad):")
        print(f"  Rx: {pose[3]:.6f}")
        print(f"  Ry: {pose[4]:.6f}")
        print(f"  Rz: {pose[5]:.6f}")
        print()
        print(f"旋转坐标 (度):")
        print(f"  Rx: {pose[3] * 180 / 3.14159:.3f}°")
        print(f"  Ry: {pose[4] * 180 / 3.14159:.3f}°")
        print(f"  Rz: {pose[5] * 180 / 3.14159:.3f}°")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请确保:")
        print("1. 机器人IP地址正确 (当前设置: {})".format(ROBOT_IP))
        print("2. 机器人已连接并运行")
        print("3. ur_rtde 库已安装") 