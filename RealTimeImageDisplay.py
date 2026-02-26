import ctypes
import cv2
import numpy as np
import time
import os
from ctypes import cast, POINTER, c_ubyte, c_float, c_long, c_ulong

# 导入提供的模块
import GetSharedBufferInfo
from GetSharedBufferInfo import VSSBCT_RGB, VSSBCT_DEPTH

def main():
    # 连接到共享缓冲区
    api = GetSharedBufferInfo.api_ver2_contents
    handle = GetSharedBufferInfo.sbHandle
    if not handle:
        print("Error: Failed to connect to shared buffer.")
        return

    # 获取图像
    width = api.GetWidth(handle)
    height = api.GetHeight(handle)
    print(f"Connected to shared buffer. Image size: {width}x{height}")
    if width == 0 or height == 0:
        print("Error: Invalid image dimensions.")
        return

    # 显示图像
    print("Starting image capture loop. Press 'q' to exit.")
    try:
        while True:
            # 锁定缓冲区
            lock_status = api.Lock(handle, 100)
            
            # 检查是否锁定成功 (这里假设 Lock 返回非空指针表示成功，具体取决于 DLL 实现)
            # 如果 Lock 返回的是错误码指针，需要根据文档判断。
            
            try:
                # 获取 RGB 数据
                # pageindex 通常为 0
                pageindex = c_ulong(0)
                pixeldatatemp = api.GetData(handle, VSSBCT_RGB, pageindex)
                
                if pixeldatatemp:
                    # 将 ctypes 指针转换为 numpy 数组
                    # RGB 数据通常是 3 通道，每个通道 8 位 (ubyte)
                    # 数据大小 = width * height * 3
                    img_data = np.ctypeslib.as_array(cast(pixeldatatemp, POINTER(c_ubyte)), shape=(height, width, 3))
                    
                    # 转换颜色空间
                    frame = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                    # 图像倒置，垂直翻转
                    frame = cv2.flip(frame, 0)
                    
                    # 显示图像
                    cv2.imshow("CarSim Live View", frame)
                
                # 获取深度数据
                # depthdatatemp = api.GetData(handle, VSSBCT_DEPTH, pageindex)
                # if depthdatatemp:
                #     depth_data = np.ctypeslib.as_array(cast(depthdatatemp, POINTER(c_float)), shape=(height, width))
                #     # 归一化以便显示
                #     depth_display = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                #     cv2.imshow("CarSim Depth", depth_display)

            finally:
                api.UnLock(handle)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 帧率控制
            # time.sleep(0.01)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        # 断开连接
        # api.Disconnect(ctypes.byref(handle)) # 注意：GetSharedBufferInfo 中没有暴露 Disconnect 的直接调用方式，且 handle 是指针
        print("Exited.")

if __name__ == "__main__":
    main()
