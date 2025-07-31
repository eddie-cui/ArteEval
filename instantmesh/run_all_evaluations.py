import os
import json
import subprocess
import argparse
import numpy as np
from tqdm import tqdm

def run_evaluation(command):
    """执行评估命令并返回结果"""
    print(f"执行命令: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"命令执行失败: {result.stderr}")
        return None
    
    print(result.stdout)
    
    # 查找结果文件路径
    output_file = None
    for arg in command:
        if arg.startswith("--output") and "=" in arg:
            output_file = arg.split("=")[1]
        elif arg == "--output":
            output_file = command[command.index(arg) + 1]
    
    if not output_file:
        print("未找到结果文件路径")
        return None
    
    try:
        with open(output_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取结果文件失败: {str(e)}")
        return None

def calculate_overall_average(results_list):
    """计算所有结果的整体平均值"""
    all_samples = {}
    all_per_view_metrics = []
    
    # 收集所有样本结果
    for result in results_list:
        if 'per_sample' in result:
            all_samples.update(result['per_sample'])
        
        if 'view_average' in result and 'error' not in result['view_average']:
            # 如果视图平均值有效，则添加到列表中
            all_per_view_metrics.append(result['view_average'])
    
    # 计算样本平均值
    if all_samples:
        valid_samples = [s for s in all_samples.values() if 'average' in s and 'error' not in s]
        if valid_samples:
            metrics_keys = valid_samples[0]['average'].keys()
            
            # 计算每个样本的平均指标
            sample_avg = {
                k: np.mean([s['average'][k] for s in valid_samples]) 
                for k in metrics_keys
            }
            
            # 计算所有视角的平均指标
            if all_per_view_metrics:
                view_avg = {
                    k: np.mean([v[k] for v in all_per_view_metrics]) 
                    for k in metrics_keys
                }
            else:
                view_avg = {"error": "无有效视角"}
        else:
            sample_avg = {"error": "无有效样本"}
            view_avg = {"error": "无有效样本"}
    else:
        sample_avg = {"error": "无有效样本"}
        view_avg = {"error": "无有效样本"}
    
    return {
        'overall_sample_average': sample_avg,
        'overall_view_average': view_avg,
        'all_samples': all_samples
    }

def main():
    # 定义所有评估命令
    commands = [
        [
            "python", "evaluator.py", "--mode", "specific_models",
            "--gen_root", r"E:\yl\1\InstantMesh\InstantMesh\newdataset_output_mesh",
            "--gt_root", r"E:\potterylike_dataset\tmp",
            "--gt_rotate_y", "90",
            "--gen_rotate_x", "90",
            "--gen_rotate_y", "180",
            "--model_ids", "25,41",
            "--output", "results_1.json"
        ],
        [
            "python", "evaluator.py", "--mode", "specific_models",
            "--gen_root", r"E:\yl\1\InstantMesh\InstantMesh\newdataset_output_mesh",
            "--gt_root", r"E:\potterylike_dataset\tmp",
            "--gt_rotate_x", "-90",
            "--gen_rotate_x", "90",
            "--gen_rotate_y", "90",
            "--model_ids", "1,2",
            "--output", "results_2.json"
        ],
        [
            "python", "evaluator.py", "--mode", "specific_models",
            "--gen_root", r"E:\yl\1\InstantMesh\InstantMesh\newdataset_output_mesh",
            "--gt_root", r"E:\potterylike_dataset\tmp",
            "--gt_rotate_y", "90",
            "--gen_rotate_x", "90",
            "--gen_rotate_y", "90",
            "--model_ids", "27,30,39,42,46",
            "--output", "results_3.json"
        ],
        [
            "python", "evaluator.py", "--mode", "specific_models",
            "--gen_root", r"E:\yl\1\InstantMesh\InstantMesh\newdataset_output_mesh",
            "--gt_root", r"E:\potterylike_dataset\tmp",
            "--gt_rotate_y", "-90",
            "--gen_rotate_x", "90",
            "--gen_rotate_y", "-90",
            "--model_ids", "8,45",
            "--output", "results_4.json"
        ],
        [
            "python", "evaluator.py", "--mode", "specific_models",
            "--gen_root", r"E:\yl\1\InstantMesh\InstantMesh\newdataset_output_mesh",
            "--gt_root", r"E:\potterylike_dataset\tmp",
            "--gt_rotate_y", "90",
            "--gen_rotate_x", "90",
            "--gen_rotate_y", "-90",
            "--model_ids", "52",
            "--output", "results_5.json"
        ],
        [
            "python", "evaluator.py", "--mode", "specific_models",
            "--gen_root", r"E:\yl\1\InstantMesh\InstantMesh\newdataset_output_mesh",
            "--gt_root", r"E:\potterylike_dataset\tmp",
            "--gen_rotate_x", "90",
            "--gen_rotate_y", "-90",
            "--model_ids", "55",
            "--output", "results_6.json"
        ]
    ]
    
    # 执行所有评估命令
    all_results = []
    for i, cmd in enumerate(tqdm(commands, desc="执行评估任务")):
        result = run_evaluation(cmd)
        if result:
            all_results.append(result)
    
    # 计算整体平均值
    overall_results = calculate_overall_average(all_results)
    
    # 保存整体结果
    with open("overall_results.json", 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    print("\n整体评估结果:")
    print(f"样本平均指标: {overall_results['overall_sample_average']}")
    print(f"视角平均指标: {overall_results['overall_view_average']}")
    print(f"结果已保存到: overall_results.json")

if __name__ == "__main__":
    main()