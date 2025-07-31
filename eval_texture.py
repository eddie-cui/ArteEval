import os
import glob
import json
import numpy as np
import torch
from PIL import Image
import lpips
import clip
import torchvision.transforms as T
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

class TextureEvaluator:
    def __init__(self, device='cuda', lpips_net='alex', clip_model_name='ViT-B/32'):
        """
        初始化纹理质量评估器
        
        参数:
            device: 计算设备 ('cuda' 或 'cpu')
            lpips_net: LPIPS网络类型 ('alex', 'vgg' 或 'squeeze')
            clip_model_name: CLIP模型名称 ('ViT-B/32', 'RN50' 等)
        """
        self.device = torch.device(device)
        
        # 初始化LPIPS模型 (感知相似性度量)
        self.lpips_fn = lpips.LPIPS(net=lpips_net).to(self.device).eval()
        
        # 初始化CLIP模型
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        self.clip_model.eval()
        
        # 定义标准图像转换 (用于PSNR/SSIM计算)
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x * 255)  # 转换为0-255范围
        ])

    def compute_psnr(self, img1, img2):
        """计算PSNR (峰值信噪比)"""
        return peak_signal_noise_ratio(img1, img2, data_range=255)

    def compute_ssim(self, img1, img2):
        """计算SSIM (结构相似性)"""
        return structural_similarity(img1, img2, data_range=255, multichannel=True, channel_axis=-1)

    def compute_lpips(self, img1, img2):
        """计算LPIPS (感知相似性)"""
        with torch.no_grad():
            return self.lpips_fn(img1, img2).item()

    def compute_clip_similarity(self, img1, img2):
        """计算CLIP相似度"""
        img1 = self.clip_preprocess(Image.fromarray(img1)).unsqueeze(0).to(self.device)
        img2 = self.clip_preprocess(Image.fromarray(img2)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat1 = self.clip_model.encode_image(img1)
            feat2 = self.clip_model.encode_image(img2)
            sim = torch.nn.functional.cosine_similarity(feat1, feat2).item()
        return sim

    def process_view_pair(self, gen_path, gt_path):
        """处理单个视角的图像对"""
        # 加载图像
        gen_img = np.array(Image.open(gen_path).convert('RGB'))
        gt_img = np.array(Image.open(gt_path).convert('RGB'))
        
        # 转换用于深度学习的张量 (LPIPS需要)
        gen_tensor = self.to_tensor(gen_img).unsqueeze(0).to(self.device)
        gt_tensor = self.to_tensor(gt_img).unsqueeze(0).to(self.device)
        
        # 计算所有指标
        metrics = {
            'psnr': self.compute_psnr(gen_img, gt_img),
            'ssim': self.compute_ssim(gen_img, gt_img),
            'lpips': self.compute_lpips(gen_tensor, gt_tensor),
            'clip_sim': self.compute_clip_similarity(gen_img, gt_img)
        }
        return metrics

    def evaluate_mesh(self, gen_views_dir, gt_views_dir):
        """
        评估单个网格的纹理质量
        
        参数:
            gen_views_dir: 生成网格的渲染图像目录
            gt_views_dir: 真实网格的渲染图像目录
        
        返回:
            results: 包含每个视角指标和平均值的字典
        """
        # 获取匹配的视角文件
        gen_files = sorted(glob.glob(os.path.join(gen_views_dir, '*.png')))
        gt_files = sorted(glob.glob(os.path.join(gt_views_dir, '*.png')))
        assert len(gen_files) == len(gt_files), f"文件数量不匹配: {len(gen_files)} vs {len(gt_files)}"
        
        # 处理所有视角对
        all_metrics = []
        for gen_path, gt_path in zip(tqdm(gen_files, desc='处理视角'), gt_files):
            all_metrics.append(self.process_view_pair(gen_path, gt_path))
        
        # 计算平均指标
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        return {'per_view': all_metrics, 'average': avg_metrics}

    def evaluate_dataset(self, dataset_root, output_file='results.json'):
        """
        评估整个数据集
        
        参数:
            dataset_root: 数据集根目录，应包含多个样本的子目录
            output_file: 结果输出文件路径
        """
        results = {}
        
        # 获取所有样本目录 (假设每个样本包含gen和gt子目录)
        samples = sorted(os.listdir(dataset_root))
        for sample in tqdm(samples, desc='评估样本'):
            if not sample.startswith('.'):  # 忽略隐藏文件夹
                gen_dir = os.path.join(dataset_root, sample, 'gen')
                gt_dir = os.path.join(dataset_root, sample, 'gt')
                
                if os.path.exists(gen_dir) and os.path.exists(gt_dir):
                    results[sample] = self.evaluate_mesh(gen_dir, gt_dir)
        
        # 计算数据集平均值
        all_psnr = [r['average']['psnr'] for r in results.values()]
        all_ssim = [r['average']['ssim'] for r in results.values()]
        all_lpips = [r['average']['lpips'] for r in results.values()]
        all_clip = [r['average']['clip_sim'] for r in results.values()]
        
        dataset_avg = {
            'psnr': np.mean(all_psnr),
            'ssim': np.mean(all_ssim),
            'lpips': np.mean(all_lpips),
            'clip_sim': np.mean(all_clip)
        }
        
        # 保存结果
        final_results = {
            'per_sample': results,
            'dataset_average': dataset_avg
        }
        
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"评估完成! 结果已保存到: {output_file}")
        return final_results

# 使用示例
if __name__ == "__main__":
    # 初始化评估器
    evaluator = TextureEvaluator(device='cuda')
    
    # 评估单个网格
    gen_dir = 'path/to/generated_mesh/renders'
    gt_dir = 'path/to/groundtruth_mesh/renders'
    mesh_results = evaluator.evaluate_mesh(gen_dir, gt_dir)
    
    print("单个网格评估结果:")
    print(json.dumps(mesh_results['average'], indent=2))
    
    # 评估整个数据集 (GSO数据集)
    dataset_root = 'path/to/gso_dataset'
    # evaluator.evaluate_dataset(dataset_root, 'gso_texture_results.json')