import os
import json
import argparse
import numpy as np
from PIL import Image
import cv2
import lpips
import clip
import torch
import torchvision.transforms as T
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import shutil
import trimesh
import glob
import logging
import pyrender
from pyrender import (PerspectiveCamera, OffscreenRenderer,
                      DirectionalLight, SpotLight, Mesh, Node)
import math

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TextureEvaluator")

class PyRenderer:
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height
        self.renderer = OffscreenRenderer(width, height)
        
        # 灯光参数 - 减小光强
        self.light_intensity = 3.0
        self.light_color = np.array([1.0, 1.0, 1.0])
        
        # 相机距离
        self.camera_distance = 1.5
        
        # 背景颜色
        self.bg_color = np.array([0.0, 0.0, 0.0, 0.0])
    
    def create_scene(self):
        """创建新的场景"""
        scene = pyrender.Scene(bg_color=self.bg_color, ambient_light=np.array([0.6, 0.6, 0.6, 1.0]))
        return scene
    
    def add_lights(self, scene):
        """向场景添加灯光 - 全打光效果"""
        # 主光源 - 上方
        light_top = DirectionalLight(color=self.light_color, intensity=self.light_intensity * 0.8)
        scene.add(light_top, pose=np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]))
        
        # 辅助光源 - 前方
        light_front = SpotLight(color=self.light_color, intensity=self.light_intensity * 0.6,
                         innerConeAngle=np.pi/6, outerConeAngle=np.pi/4)
        scene.add(light_front, pose=np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]))
        
        # 辅助光源 - 后方
        light_back = SpotLight(color=self.light_color, intensity=self.light_intensity * 0.5,
                         innerConeAngle=np.pi/6, outerConeAngle=np.pi/4)
        scene.add(light_back, pose=np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]))
        
        # 辅助光源 - 左侧
        light_left = SpotLight(color=self.light_color, intensity=self.light_intensity * 0.5,
                         innerConeAngle=np.pi/6, outerConeAngle=np.pi/4)
        scene.add(light_left, pose=np.array([
            [0.0, 0.0, -1.0, -1.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]))
        
        # 辅助光源 - 右侧
        light_right = SpotLight(color=self.light_color, intensity=self.light_intensity * 0.5,
                         innerConeAngle=np.pi/6, outerConeAngle=np.pi/4)
        scene.add(light_right, pose=np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]))
        
        return scene

    def rotate_mesh(self, mesh, rotation):
        """旋转网格"""
        if rotation is not None:
            # 创建旋转矩阵
            rotation_matrix = trimesh.transformations.euler_matrix(
                math.radians(rotation[0]),
                math.radians(rotation[1]),
                math.radians(rotation[2])
            )
            # 应用旋转
            mesh.apply_transform(rotation_matrix)
        return mesh

    def center_mesh(self, mesh):
        """将网格移动到场景中心并缩放以适应相机"""
        # 计算网格的边界框
        min_bound = mesh.bounds[0]
        max_bound = mesh.bounds[1]
        center = (min_bound + max_bound) / 2.0
        
        # 计算网格大小并缩放以适应相机
        size = max_bound - min_bound
        max_size = max(size)
        scale_factor = 1.0 / max_size
        
        # 创建变换矩阵：先缩放后平移
        transform = np.eye(4)
        transform[:3, :3] *= scale_factor  # 缩放
        transform[:3, 3] = -center * scale_factor  # 平移
        
        # 应用变换
        mesh.apply_transform(transform)
        
        return mesh

    def look_at(self, eye, target=np.array([0, 0, 0]), up=np.array([0, 1, 0])):
        """计算从eye位置看向target点的视图矩阵"""
        # 计算前向向量
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        
        # 计算右向量
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # 重新计算上向量
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # 构建视图矩阵
        view_matrix = np.eye(4)
        view_matrix[:3, 0] = right
        view_matrix[:3, 1] = up
        view_matrix[:3, 2] = -forward
        view_matrix[:3, 3] = eye
        
        return view_matrix

    def load_obj_with_materials(self, obj_path):
        """加载OBJ文件并正确处理材质和纹理"""
        try:
            # 使用trimesh加载OBJ文件
            mesh = trimesh.load(obj_path)
            
            # 如果加载的是场景，尝试提取第一个网格
            if isinstance(mesh, trimesh.Scene):
                logger.warning(f"文件 {obj_path} 是一个场景，尝试提取第一个网格")
                for node in mesh.graph.nodes_geometry:
                    mesh = mesh.geometry[node]
                    break
            
            # 确保加载的是网格
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"无法从 {obj_path} 加载有效网格")
            
            return mesh
        except Exception as e:
            logger.error(f"加载OBJ文件时出错: {str(e)}")
            # 返回一个简单的立方体作为占位符
            return trimesh.creation.box()

    def load_mesh_with_texture(self, mesh_path):
        """加载网格并保留纹理信息"""
        try:
            # 检查文件扩展名
            ext = os.path.splitext(mesh_path)[1].lower()
            
            # 如果是OBJ文件，使用特殊加载器处理材质
            if ext == '.obj':
                return self.load_obj_with_materials(mesh_path)
            
            # 对于其他格式，使用trimesh加载
            mesh = trimesh.load(mesh_path)
            
            # 如果加载的是场景，尝试提取第一个网格
            if isinstance(mesh, trimesh.Scene):
                logger.warning(f"文件 {mesh_path} 是一个场景，尝试提取第一个网格")
                for node in mesh.graph.nodes_geometry:
                    mesh = mesh.geometry[node]
                    break
            
            # 确保加载的是网格
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"无法从 {mesh_path} 加载有效网格")
            
            return mesh
        except Exception as e:
            logger.error(f"加载网格时出错: {str(e)}")
            # 返回一个简单的立方体作为占位符
            return trimesh.creation.box()

    def render_mesh(self, mesh_path, elevation, azimuth, rotation=None):
        """渲染指定视角的网格"""
        try:
            # 创建新场景
            scene = self.create_scene()
            scene = self.add_lights(scene)
            
            # 加载网格并保留纹理
            mesh_trimesh = self.load_mesh_with_texture(mesh_path)
            
            # 旋转网格
            mesh_trimesh = self.rotate_mesh(mesh_trimesh, rotation)
            
            # 将网格移动到场景中心并缩放
            mesh_trimesh = self.center_mesh(mesh_trimesh)
            
            # 创建PyRender网格
            mesh = Mesh.from_trimesh(mesh_trimesh, smooth=False)
            
            # 添加网格到场景
            scene.add(mesh)
            
            # 设置相机
            camera = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=self.width/self.height)
            
            # 计算相机位置 (球坐标)
            azimuth_rad = math.radians(azimuth)
            elevation_rad = math.radians(elevation)
            
            # 计算相机位置
            x = self.camera_distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
            y = self.camera_distance * math.sin(elevation_rad)
            z = self.camera_distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
            eye = np.array([x, y, z])
            
            # 计算相机姿态矩阵
            camera_pose = self.look_at(eye)
            
            # 添加相机
            camera_node = scene.add(camera, pose=camera_pose)
            
            # 渲染
            color, depth = self.renderer.render(scene)
            
            # 转换为RGB格式
            color = cv2.cvtColor(color, cv2.COLOR_RGBA2RGB)
            
            return color
        except Exception as e:
            logger.error(f"渲染网格时出错: {str(e)}")
            # 返回黑色图像作为占位符
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def render_all_views(self, mesh_path, output_dir, rotation=None):
        """渲染网格的所有视角并保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        elevation_angles = [0, 15, 30]
        azimuth_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        
        images = []
        
        for elev in elevation_angles:
            for azim in azimuth_angles:
                img = self.render_mesh(mesh_path, elev, azim, rotation)
                
                # 保存渲染结果
                img_path = os.path.join(output_dir, f"elev{elev}_azim{azim}.png")
                cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                logger.info(f"已保存渲染结果: {img_path}")
                
                images.append(img)
        
        return images


class TextureEvaluator:
    def __init__(self, device='cuda', lpips_net='alex', clip_model_name='ViT-B/32'):
        self.device = torch.device(device)
        self.lpips_fn = lpips.LPIPS(net=lpips_net).to(self.device).eval()
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        self.clip_model.eval()
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x * 255)
        ])

    def compute_psnr(self, img1, img2):
        """计算峰值信噪比"""
        return peak_signal_noise_ratio(img1, img2, data_range=255)

    def compute_ssim(self, img1, img2):
        """计算结构相似性"""
        return structural_similarity(img1, img2, data_range=255, channel_axis=-1)

    def compute_lpips(self, img1, img2):
        """计算感知相似性"""
        with torch.no_grad():
            return self.lpips_fn(img1, img2).item()

    def compute_clip_similarity(self, img1, img2):
        """计算CLIP图像相似度"""
        img1 = self.clip_preprocess(Image.fromarray(img1)).unsqueeze(0).to(self.device)
        img2 = self.clip_preprocess(Image.fromarray(img2)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat1 = self.clip_model.encode_image(img1)
            feat2 = self.clip_model.encode_image(img2)
            sim = torch.nn.functional.cosine_similarity(feat1, feat2).item()
        return sim

    def process_view_pair(self, gen_img, gt_img):
        # 确保图像为uint8格式
        gen_img = gen_img.astype(np.uint8)
        gt_img = gt_img.astype(np.uint8)
        
        # 转换图像为张量
        gen_tensor = self.to_tensor(gen_img).unsqueeze(0).to(self.device) / 255.0
        gt_tensor = self.to_tensor(gt_img).unsqueeze(0).to(self.device) / 255.0
        
        # 计算所有指标
        metrics = {
            'psnr': self.compute_psnr(gen_img, gt_img),
            'ssim': self.compute_ssim(gen_img, gt_img),
            'lpips': self.compute_lpips(gen_tensor, gt_tensor),
            'clip_sim': self.compute_clip_similarity(gen_img, gt_img)
        }
        return metrics


class TextureEvaluationSystem:
    def __init__(self, render_resolution=512, device='cuda', save_renders=True):
        self.device = device
        self.render_resolution = render_resolution
        self.renderer = PyRenderer(width=render_resolution, height=render_resolution)
        self.evaluator = TextureEvaluator(device=device)
        self.save_renders = save_renders
        
        # 临时目录用于存储渲染图像
        self.temp_dir = "temp_renders"
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(f"初始化渲染器: {render_resolution}x{render_resolution}")
        logger.info(f"保存渲染结果: {'是' if self.save_renders else '否'}")

    def render_and_evaluate(self, gen_mesh_path, gt_mesh_path, gen_rotation=None, gt_rotation=None):
        """
        渲染并评估网格对
        
        参数:
            gen_mesh_path: 生成网格路径
            gt_mesh_path: 真实网格路径
            gen_rotation: 生成网格旋转角度 (x, y, z) 或 None
            gt_rotation: 真实网格旋转角度 (x, y, z) 或 None
            
        返回:
            评估果字典
        """
        logger.info(f"开始评估: 生成网格={gen_mesh_path}, 真实网格={gt_mesh_path}")
        if gen_rotation:
            logger.info(f"生成网格旋转: X={gen_rotation[0]}°, Y={gen_rotation[1]}°, Z={gen_rotation[2]}°")
        if gt_rotation:
            logger.info(f"真实网格旋转: X={gt_rotation[0]}°, Y={gt_rotation[1]}°, Z={gt_rotation[2]}°")
        
        try:
            # 获取模型ID（从生成网格路径中提取）
            # model_id = os.path.basename(os.path.dirname(gen_mesh_path))
            model_id = os.path.dirname(gen_mesh_path).split("\\")[-1]
            
            # 创建基于模型ID的父文件夹
            parent_dir = os.path.join(self.temp_dir, model_id)
            os.makedirs(parent_dir, exist_ok=True)
            
            # 创建gen和gt子文件夹
            gen_dir = os.path.join(parent_dir, "gen")
            gt_dir = os.path.join(parent_dir, "gt")
            os.makedirs(gen_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)
            
            # 渲染生成网格
            logger.info("渲染生成网格...")
            gen_images = self.renderer.render_all_views(gen_mesh_path, gen_dir, gen_rotation)
            
            # 渲染真实网格
            logger.info("渲染真实网格...")
            gt_images = self.renderer.render_all_views(gt_mesh_path, gt_dir, gt_rotation)
            
            # 确保渲染图像数量一致
            if len(gen_images) != len(gt_images):
                logger.warning(f"渲染视角数量不一致: {len(gen_images)} vs {len(gt_images)}")
                logger.warning(f"使用最小视角数: {min(len(gen_images), len(gt_images))}")
                min_views = min(len(gen_images), len(gt_images))
                gen_images = gen_images[:min_views]
                gt_images = gt_images[:min_views]
            
            # 评估每个视角对
            results = {'per_view': [], 'average': {}}
            for i in tqdm(range(len(gen_images)), desc="评估纹理质量"):
                metrics = self.evaluator.process_view_pair(gen_images[i], gt_images[i])
                results['per_view'].append(metrics)
            
            # 计算平均指标
            metrics_keys = results['per_view'][0].keys()
            results['average'] = {
                k: np.mean([m[k] for m in results['per_view']]) 
                for k in metrics_keys
            }
            
            # 保存渲染目录路径
            if self.save_renders:
                results['render_dirs'] = {
                    'gen': os.path.abspath(gen_dir),
                    'gt': os.path.abspath(gt_dir),
                    'parent': os.path.abspath(parent_dir)
                }
            
            logger.info(f"评估完成: PSNR={results['average']['psnr']:.2f}, SSIM={results['average']['ssim']:.4f}")
            
            return results
        except Exception as e:
            logger.error(f"评估过程中出错: {str(e)}")
            return {'error': str(e)}

    def evaluate_specific_models(self, gen_root, gt_root, model_ids, output_file='results.json', gen_rotation=None, gt_rotation=None):
        """
        评估指定的模型ID
        
        参数:
            gen_root: 生成网格根目录
            gt_root: 真实网格根目录
            model_ids: 要评估的模型ID列表
            output_file: 结果输出文件
            gen_rotation: 生成网格旋转角度 (x, y, z) 或 None
            gt_rotation: 真实网格旋转角度 (x, y, z) 或 None
            
        返回:
            完整评估结果
        """
        # 支持的网格格式
        mesh_exts = ['.obj', '.glb', '.ply', '.stl', '.fbx']
        
        # 将模型ID转换为字符串
        model_ids_str = [str(id) for id in model_ids]
        logger.info(f"评估指定的模型ID: {model_ids_str}")
        
        results = {}
        all_per_view_metrics = []  # 存储所有视角的指标
        
        # 处理每个指定的模型ID
        for model_id in tqdm(model_ids_str, desc='评估指定模型'):
            # 在生成目录中查找模型文件
            gen_files = []
            for ext in mesh_exts:
                gen_files.extend(glob.glob(os.path.join(gen_root, model_id, f'*{ext}')))
            
            if not gen_files:
                logger.warning(f"在 {os.path.join(gen_root, model_id)} 中未找到生成网格")
                continue
                
            # 使用第一个找到的生成网格文件
            gen_path = gen_files[0]
            logger.info(f"使用生成网格: {gen_path}")
            
            # 在真实目录中查找对应网格
            gt_model_dir = os.path.join(gt_root, model_id)
            
            if not os.path.exists(gt_model_dir):
                logger.warning(f"真实目录不存在: {gt_model_dir}")
                continue
            
            # 尝试在模型ID目录下直接查找网格文件
            gt_path = None
            for ext in mesh_exts:
                candidate_files = glob.glob(os.path.join(gt_model_dir, f'*{ext}'))
                if candidate_files:
                    gt_path = candidate_files[0]
                    break
            
            # 如果直接目录下没有找到，再查找子目录
            if not gt_path:
                # 查找真实模型子目录
                gt_model_dirs = []
                if os.path.exists(gt_model_dir):
                    gt_model_dirs = [d for d in os.listdir(gt_model_dir) 
                                     if os.path.isdir(os.path.join(gt_model_dir, d))]
                
                # 尝试在子目录中找到匹配的模型
                for model_dir in gt_model_dirs:
                    model_dir_path = os.path.join(gt_model_dir, model_dir)
                    
                    # 尝试多种可能的路径模式
                    candidate_paths = [
                        os.path.join(model_dir_path, "meshes", "model.obj"),
                        os.path.join(model_dir_path, "Scan", "Scan.obj"),
                        os.path.join(model_dir_path, "model.obj"),
                        os.path.join(model_dir_path, "Scan.obj"),
                        os.path.join(model_dir_path, "source", "model.obj"),
                        os.path.join(model_dir_path, "source", "Scan.obj"),
                        os.path.join(model_dir_path, "source", "model", "model.obj"),
                        os.path.join(model_dir_path, "source", "s313", "s313.obj"),
                        os.path.join(model_dir_path, "model")
                    ]
                    
                    for candidate in candidate_paths:
                        # 如果候选路径是目录，尝试在该目录下查找网格文件
                        if os.path.isdir(candidate):
                            for ext in mesh_exts:
                                files_in_dir = glob.glob(os.path.join(candidate, f'*{ext}'))
                                if files_in_dir:
                                    gt_path = files_in_dir[0]
                                    break
                            if gt_path:
                                break
                        # 如果候选路径是文件，直接使用
                        elif os.path.isfile(candidate):
                            gt_path = candidate
                            break
                    
                    if gt_path:
                        break
                
                # 如果仍然没有找到，尝试在子目录中查找任何网格文件
                if not gt_path:
                    for model_dir in gt_model_dirs:
                        model_dir_path = os.path.join(gt_model_dir, model_dir)
                        for ext in mesh_exts:
                            candidate_files = glob.glob(os.path.join(model_dir_path, f'*{ext}'))
                            if candidate_files:
                                gt_path = candidate_files[0]
                                break
                        if gt_path:
                            break
            
            if not gt_path:
                logger.warning(f"未找到真实网格对应: {gen_path}")
                continue
            
            try:
                # 确保临时目录存在
                os.makedirs(self.temp_dir, exist_ok=True)
                
                # 评估网格对
                sample_results = self.render_and_evaluate(gen_path, gt_path, gen_rotation, gt_rotation)
                
                # 保存结果
                results[model_id] = sample_results
                
                # 收集所有视角的指标
                if 'per_view' in sample_results:
                    all_per_view_metrics.extend(sample_results['per_view'])
            except Exception as e:
                logger.error(f"处理 {gen_path} 时出错: {str(e)}")
                results[model_id] = {'error': str(e)}
        
        # 计算数据集平均值
        if results:
            valid_results = [res for res in results.values() if 'average' in res and 'error' not in res]
            if valid_results:
                metrics_keys = valid_results[0]['average'].keys()
                
                # 计算每个样本的平均指标
                dataset_avg = {
                    k: np.mean([r['average'][k] for r in valid_results]) 
                    for k in metrics_keys
                }
                
                # 计算所有视角的平均指标
                if all_per_view_metrics:
                    view_avg = {
                        k: np.mean([m[k] for m in all_per_view_metrics]) 
                        for k in metrics_keys
                    }
                else:
                    view_avg = {"error": "无有效视角"}
                
                # 打印数据集平均指标
                avg_str = ", ".join([f"{k.upper()}={dataset_avg[k]:.4f}" for k in metrics_keys])
                logger.info(f"数据集平均指标: {avg_str}")
                
                # 打印视角平均指标
                view_str = ", ".join([f"{k.upper()}={view_avg[k]:.4f}" for k in metrics_keys])
                logger.info(f"视角平均指标: {view_str}")
            else:
                dataset_avg = {"error": "无有效样本"}
                view_avg = {"error": "无有效样本"}
        else:
            dataset_avg = {"error": "无有效样本"}
            view_avg = {"error": "无有效样本"}
        
        final_results = {
            'per_sample': results,
            'dataset_average': dataset_avg,
            'view_average': view_avg
        }
        
        # 保存结果
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"评估完成! 结果已保存到: {output_file}")
        return final_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="增强版3D网格纹理质量评估系统")
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['single', 'dataset', 'specific_models'], 
                        help="评估模式: 'single' 单样本评估 | 'dataset' 数据集评估 | 'specific_models' 评估指定模型")
    
    # 单样本模式参数
    parser.add_argument('--gen_mesh', type=str, help="生成网格路径")
    parser.add_argument('--gt_mesh', type=str, help="真实网格路径")
    
    # 数据集模式参数
    parser.add_argument('--gen_root', type=str, help="生成网格根目录")
    parser.add_argument('--gt_root', type=str, help="真实网格根目录")
    
    # 通用参数
    parser.add_argument('--output', type=str, default='results.json', help="结果输出文件路径")
    parser.add_argument('--resolution', type=int, default=512, help="渲染分辨率")
    parser.add_argument('--save_renders', action='store_true', help="保存渲染结果")
    parser.add_argument('--light_intensity', type=float, default=3.0, help="灯光强度")
    parser.add_argument('--camera_distance', type=float, default=1.5, help="相机距离")
    
    # 旋转参数 - 分别用于生成网格和真实网格
    parser.add_argument('--gen_rotate_x', type=float, default=0.0, help="生成网格绕X轴旋转角度（度）")
    parser.add_argument('--gen_rotate_y', type=float, default=0.0, help="生成网格绕Y轴旋转角度（度）")
    parser.add_argument('--gen_rotate_z', type=float, default=0.0, help="生成网格绕Z轴旋转角度（度）")
    
    parser.add_argument('--gt_rotate_x', type=float, default=0.0, help="真实网格绕X轴旋转角度（度）")
    parser.add_argument('--gt_rotate_y', type=float, default=0.0, help="真实网格绕Y轴旋转角度（度）")
    parser.add_argument('--gt_rotate_z', type=float, default=0.0, help="真实网格绕Z轴旋转角度（度）")
    
    # 模型ID参数
    parser.add_argument('--model_ids', type=str, required=True, help="要评估的模型ID列表，用逗号分隔，例如: '1,2,8,25,27,30,39,41,42,45,46,52,55'")
    args = parser.parse_args()

    # 初始化评估系统
    evaluator = TextureEvaluationSystem(
        render_resolution=args.resolution, 
        device='cuda',
        save_renders=args.save_renders
    )
    
    # 设置渲染参数
    evaluator.renderer.light_intensity = args.light_intensity
    evaluator.renderer.camera_distance = args.camera_distance
    
    # 创建旋转元组
    gen_rotation = (args.gen_rotate_x, args.gen_rotate_y, args.gen_rotate_z)
    gt_rotation = (args.gt_rotate_x, args.gt_rotate_y, args.gt_rotate_z)
    
    # 解析模型ID列表
    try:
        # 先去除字符串两端的引号
        model_ids_str = args.model_ids.strip().strip("'").strip('"')
        # 然后分割字符串
        model_ids = [int(id.strip()) for id in model_ids_str.split(',')]
        logger.info(f"评估指定的模型ID: {model_ids}")
    except Exception as e:
        logger.error(f"解析模型ID时出错: {str(e)}")
        model_ids = None
    
    if not model_ids:
        logger.error("未提供有效的模型ID列表")
        exit(1)
    
    if args.mode == 'single':
        if not all([args.gen_mesh, args.gt_mesh]):
            parser.error("单样本模式需要提供 --gen_mesh 和 --gt_mesh")
            
        # 检查文件是否存在
        if not os.path.exists(args.gen_mesh):
            parser.error(f"生成网格不存在: {args.gen_mesh}")
        if not os.path.exists(args.gt_mesh):
            parser.error(f"真实网格不存在: {args.gt_mesh}")
            
        # 执行评估
        results = evaluator.render_and_evaluate(args.gen_mesh, args.gt_mesh, gen_rotation, gt_rotation)
        
        # 打印结果
        if 'error' in results:
            print(f"评估失败: {results['error']}")
        else:
            print("\n评估结果:")
            print(f"PSNR: {results['average']['psnr']:.4f}")
            print(f"SSIM: {results['average']['ssim']:.4f}")
            print(f"LPIPS: {results['average']['lpips']:.4f}")
            print(f"CLIP相似度: {results['average']['clip_sim']:.4f}")
            
            # 如果保存了渲染结果，显示路径
            if 'render_dirs' in results:
                print("\n渲染结果已保存到:")
                if 'gen' in results['render_dirs']:
                    print(f"生成网格渲染: {results['render_dirs']['gen']}")
                if 'gt' in results['render_dirs']:
                    print(f"真实网格渲染: {results['render_dirs']['gt']}")
                if 'parent' in results['render_dirs']:
                    print(f"父文件夹: {results['render_dirs']['parent']}")
            
            # 保存结果
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
                print(f"评估结果已保存到: {args.output}")
        
    elif args.mode == 'dataset':
        if not all([args.gen_root, args.gt_root]):
            parser.error("数据集模式需要提供 --gen_root 和 --gt_root")
            
        # 检查目录是否存在
        if not os.path.isdir(args.gen_root):
            parser.error(f"生成目录不存在: {args.gen_root}")
        if not os.path.isdir(args.gt_root):
            parser.error(f"真实目录不存在: {args.gt_root}")
            
        # 执行数据集评估
        final_results = evaluator.evaluate_specific_models(
            args.gen_root, 
            args.gt_root, 
            model_ids, 
            args.output, 
            gen_rotation, 
            gt_rotation
        )
        
        # 打印总平均指标
        if 'dataset_average' in final_results and 'error' not in final_results['dataset_average']:
            print("\n数据集总平均指标:")
            for metric, value in final_results['dataset_average'].items():
                print(f"{metric.upper()}: {value:.4f}")
        
        if 'view_average' in final_results and 'error' not in final_results['view_average']:
            print("\n视角总平均指标:")
            for metric, value in final_results['view_average'].items():
                print(f"{metric.upper()}: {value:.4f}")
        
        print(f"评估完成! 结果已保存到: {args.output}")
        if args.save_renders:
            print(f"渲染结果已保存到: {os.path.abspath(evaluator.temp_dir)}")
    
    elif args.mode == 'specific_models':
        if not all([args.gen_root, args.gt_root]):
            parser.error("指定模型模式需要提供 --gen_root 和 --gt_root")
            
        # 检查目录是否存在
        if not os.path.isdir(args.gen_root):
            parser.error(f"生成目录不存在: {args.gen_root}")
        if not os.path.isdir(args.gt_root):
            parser.error(f"真实目录不存在: {args.gt_root}")
            
        # 执行指定模型评估
        final_results = evaluator.evaluate_specific_models(
            args.gen_root, 
            args.gt_root, 
            model_ids, 
            args.output, 
            gen_rotation, 
            gt_rotation
        )
        
        # 打印总平均指标
        if 'dataset_average' in final_results and 'error' not in final_results['dataset_average']:
            print("\n数据集总平均指标:")
            for metric, value in final_results['dataset_average'].items():
                print(f"{metric.upper()}: {value:.4f}")
        
        if 'view_average' in final_results and 'error' not in final_results['view_average']:
            print("\n视角总平均指标:")
            for metric, value in final_results['view_average'].items():
                print(f"{metric.upper()}: {value:.4f}")
        
        print(f"评估完成! 结果已保存到: {args.output}")
        if args.save_renders:
            print(f"渲染结果已保存到: {os.path.abspath(evaluator.temp_dir)}")