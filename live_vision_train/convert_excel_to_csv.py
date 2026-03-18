#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Excel扭矩表转CSV工具
将 FLmap.xlsx, FRmap.xlsx, RLmap.xlsx, RRmap.xlsx 转换为 CSV格式

使用方式:
  python convert_excel_to_csv.py --input_dir "ASR_TorqueMap" --output_dir "ASR_TorqueMap"
"""
import os
import argparse

def convert_excel_to_csv(input_dir, output_dir=None):
    """
    将Excel文件转换为CSV格式
    """
    try:
        import pandas as pd
    except ImportError:
        print("[Error] 需要安装 pandas: pip install pandas openpyxl")
        return False
    
    if output_dir is None:
        output_dir = input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    wheel_files = [
        ('FLmap.xlsx', 'FLmap.csv'),
        ('FRmap.xlsx', 'FRmap.csv'),
        ('RLmap.xlsx', 'RLmap.csv'),
        ('RRmap.xlsx', 'RRmap.csv'),
    ]
    
    print(f"\n[Convert] Converting Excel files from: {input_dir}")
    print(f"[Convert] Output CSV to: {output_dir}\n")
    
    for xlsx_name, csv_name in wheel_files:
        xlsx_path = os.path.join(input_dir, xlsx_name)
        csv_path = os.path.join(output_dir, csv_name)
        
        if not os.path.exists(xlsx_path):
            print(f"[Skip] File not found: {xlsx_path}")
            continue
        
        try:
            # 读取Excel (假设数据在第一列)
            df = pd.read_excel(xlsx_path, header=None)
            
            # 保存为CSV (单列，无header)
            df.to_csv(csv_path, index=False, header=False)
            
            print(f"[OK] {xlsx_name} -> {csv_name} ({len(df)} rows)")
            
        except Exception as e:
            print(f"[Error] Failed to convert {xlsx_name}: {e}")
    
    print(f"\n[Done] Conversion complete!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Excel torque maps to CSV')
    parser.add_argument('--input_dir', type=str, default="ASR_TorqueMap",
                        help='Directory containing Excel files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for CSV output (default: same as input)')
    
    args = parser.parse_args()
    
    convert_excel_to_csv(args.input_dir, args.output_dir)
