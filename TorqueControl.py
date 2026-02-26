import argparse
from ctypes import cdll
import os
import struct
import platform
import numpy as np
import cv2
import math
import sys

import GetSharedBufferInfo
import Simulation_with_LiveAnimation
from GetSharedBufferInfo import VSSBCT_RGB, VSSBCT_DEPTH
from ctypes import cast, POINTER, c_ubyte, c_ulong

if __name__ == '__main__':

    if sys.version_info[0] < 3:
        error_occurred = "Python version must be 3.0 or greater."
    else:
        systemSize = (8 * struct.calcsize("P"))  # 32 or 64

        parser = argparse.ArgumentParser(
            description='Python 3.5 script that runs the simulation in simfile.sim in the current directory.')

        parser.add_argument('--simfile', '-s', dest='path_to_sim_file',
                        default=os.path.join(os.getcwd(), "simfile.sim"),
                        help="Path to simfile. For example D:\\Product_dev\\Image\\CarSim\\Core\\CarSim_Data\\simfile.sim")

        parser.add_argument('--runs', '-r', type=int, dest='number_of_runs',
                        default=1,
                        help="Number of runs to make per single load of DLL. This parameter exists to replicate how real-time system use the solver")

        args = parser.parse_args()
        path_to_sim_file = args.path_to_sim_file
        number_of_runs = args.number_of_runs
        if number_of_runs < 1:
            number_of_runs = 1

        vs = Simulation_with_LiveAnimation.VehicleSimulationWithLiveAnimation()
        path_to_vs_dll = r'E:\CarSim2024.1_Prog\Database\RL_VISION\RL_TC_VISION\vs_lv_ds_x64.dll'
        error_occurred = 1
        current_os = platform.system()
        if path_to_vs_dll is not None and os.path.exists(path_to_vs_dll):
            if current_os == "Linux":
              mc_type = platform.machine()
              if mc_type == 'x86_64':
                dllSize = 64
              else:
                dllSize = 32
            else:
              if "_x64" in path_to_vs_dll:  # change this from _64 to _x64 since the naming has x64
                dllSize = 64
              else:
                dllSize = 32
            
            if systemSize != dllSize:
                print("Python size (32/64) must match size of .dlls being loaded.")
                print("Python size:", systemSize, "DLL size:", dllSize)
            else:  # systems match, we can continue

                status = 0
                vs_dll = cdll.LoadLibrary(path_to_vs_dll)
                if vs_dll is not None:
                    if vs.get_api(vs_dll):

                        for i in range(0, number_of_runs):
                            print(os.linesep + "++++++++++++++++ Starting run number: " + str(i + 1) + " ++++++++++++++++" + os.linesep)
                            configuration = vs.ReadConfiguration(path_to_sim_file.replace('\\\\', '\\'))
                            t_current = configuration.get('t_start')
                            n_export = configuration.get('n_export')
                            print(f"Number of export variables expected by CarSim: {n_export}")
                            
                            # 初始化四轮扭矩控制输入
                            torque_l1 = 0.0
                            torque_r1 = 0.0
                            torque_l2 = 0.0
                            torque_r2 = 0.0
                            import_array = [torque_l1, torque_r1, torque_l2, torque_r2]
                            
                            export_array = [0.0] * n_export
                            
                            while status == 0:
                                j = 0
                                while status == 0:

                                    t_current = t_current + configuration.get('t_step')

                                    #步进更新
                                    status, export_array = vs.updateonestep(t_current, import_array, export_array)
                                    
                                    if j == 100: 
                                        
                                        # 读取图像
                                        if GetSharedBufferInfo.sbHandle:
                                            GetSharedBufferInfo.api_ver2_contents.Lock(GetSharedBufferInfo.sbHandle, 100)
                                            
                                            try:
                                                pageindex = c_ulong(0)
                                                pixeldatatemp = GetSharedBufferInfo.api_ver2_contents.GetData(GetSharedBufferInfo.sbHandle, VSSBCT_RGB, pageindex)
                                                
                                                if pixeldatatemp:
                                                    width = GetSharedBufferInfo.width
                                                    height = GetSharedBufferInfo.height
                                                    
                                                    # 获取图像数据
                                                    img_data = np.ctypeslib.as_array(cast(pixeldatatemp, POINTER(c_ubyte)), shape=(height, width, 3))
                                                    
                                                    # 转换颜色空间 RGB -> BGR
                                                    frame = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                                                    
                                                    # 垂直翻转图像 (解决倒置问题)
                                                    frame = cv2.flip(frame, 0)
                                                    
                                                    # 显示当前扭矩信息
                                                    info_text = f"Torque: L1={torque_l1:.1f}, R1={torque_r1:.1f}, L2={torque_l2:.1f}, R2={torque_r2:.1f}"
                                                    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                                    cv2.putText(frame, f"Time: {t_current:.2f} s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                                    # 显示图像
                                                    cv2.imshow('CarSim Live View', frame)
                                                    
                                                    if cv2.waitKey(1) & 0xFF == ord('q'):
                                                        status = 1
                                                        break
                                            finally:
                                                GetSharedBufferInfo.api_ver2_contents.UnLock(GetSharedBufferInfo.sbHandle)
                                        
                                        # 更新控制输入
                                        torque_l1 = min(10.0*t_current , 200.0)
                                        torque_r1 = min(10.0*t_current , 200.0)
                                        torque_l2 = min(10.0*t_current , 200.0)
                                        torque_r2 = min(10.0*t_current , 200.0)
                                        
                                        import_array = [torque_l1, torque_r1, torque_l2, torque_r2]
                                        
                                        j = 0
                                    j = j + 1

                            # Terminate solver
                            vs.terminate(t_current)
                            cv2.destroyAllWindows()

                            if error_occurred != 0:
                                break
                            print(os.linesep + "++++++++++++++++ Ending run number: " + str(i + 1) + " ++++++++++++++++" + os.linesep)

    sys.exit(error_occurred)
