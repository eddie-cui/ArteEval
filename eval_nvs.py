import os

from PIL import Image
import cv2
import numpy as np
from argparse import ArgumentParser

import torch
from skimage.io import imread
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure
from sam_utils import sam_init, sam_out_nosave
from util import pred_bbox, image_preprocess_nosave
import lpips
import torchvision.transforms as transforms

def compute_psnr_float(img_gt, img_pr):
    img_gt = img_gt.reshape([-1, 3]).astype(np.float32)
    img_pr = img_pr.reshape([-1, 3]).astype(np.float32)
    mse = np.mean((img_gt - img_pr) ** 2, 0)
    mse = np.mean(mse) + 1e-8
    psnr = 10 * np.log10(1 / mse)
    return psnr

def color_map_forward(rgb):
    rgb = np.array(rgb, dtype=np.float32)
    # print(rgb)
    dim = rgb.shape[-1]
    new_size = (256, 256)
    if dim==3:
        rgb=cv2.resize(rgb,new_size,interpolation=cv2.INTER_CUBIC)
        return np.array(rgb, dtype=np.float32) /255
    else:
        rgb = np.array(rgb, dtype=np.float32) /255
        rgb, alpha = rgb[:,:,:3], rgb[:,:,3:]
        rgb = rgb * alpha + (1-alpha)
        rgb=cv2.resize(rgb,new_size,interpolation=cv2.INTER_CUBIC)
        return rgb

def preprocess_image(models, img_path, GT = False):
    # preprocess image
    # print(img_path)
    # print(os.path.exists(img_path))
    img = Image.open(img_path)
    # print('old input_im:', img.size)
    # img = np.array(img, dtype=np.float32) 
    
    if not GT:
        if not img.mode == 'RGBA':
            img.thumbnail([512, 512], Image.Resampling.LANCZOS)
            img = sam_out_nosave(models['sam'], img.convert("RGB"), pred_bbox(img))
            torch.cuda.empty_cache()
    img = image_preprocess_nosave(img, lower_contrast=False, rescale=True)
    return color_map_forward(img)
 
def Rect(image):
    
    return color_map_forward(image)

def compute_PPLC(images_path, perceptual_model, n_images):
    phi = 2 * 3.14159265 / n_images
    distances = []

    images = [imread(os.path.join(images_path, f'{k:03}.png')) for k in range(n_images)]
    for i in range(len(images) - 1):
        img1 = Rect(images[i])
        img2 = Rect(images[i+1])

        # 将图像转换为张量
        img1_tensor = transforms.ToTensor()(img1).unsqueeze(0).to('cuda')
        img2_tensor = transforms.ToTensor()(img2).unsqueeze(0).to('cuda')

        # 计算感知距离
        distance = perceptual_model(img1_tensor, img2_tensor).item()
        normalized_distance = (distance / (phi**2))
        
        distances.append(normalized_distance)
    
    # 计算平均感知路径长度一致性
    average_pplc = sum(distances) / len(distances)
    return average_pplc

# python eval_nvs.py --gt eval_examples/chicken-gt --pr eval_examples/chicken-pr  --name chicken
# python eval_nvs.py --gt D:\Free3D\zero123plus_out\out\Ecoforms_Plant_Container_12_Pot_Nova_gt --pr D:\Free3D\zero123plus_out\out\1  --name zero123++_Ecoforms_Plant_Container_12_Pot_Nova

def main():
    parser = ArgumentParser()
    parser.add_argument('--gt',type=str,default=r'D:\Free3D\zero123plus_out\out\Ecoforms_Plant_Container_12_Pot_Nova_gt')
    parser.add_argument('--pr',type=str,default=r'D:\Free3D\zero123plus_out\out\1')
    parser.add_argument('--num_images',type=int, default=6)
    parser.add_argument('--name',type=str, default=r'zero123++_Ecoforms_Plant_Container_12_Pot_Nova')
    args = parser.parse_args()

    gt_dir = args.gt
    pr_dir = args.pr
    num_images = args.num_images
    models = {}
    models['sam'] = sam_init(0, r'ckpts\sam_vit_h_4b8939.pth')
    models['lpips'] = lpips.LPIPS(net='vgg').cuda().eval()
    fid_score = 0
    # fid_score = fid.calculate_fid_given_paths([gt_dir, pr_dir], batch_size = 16, dims=64, device=torch.device('cuda:0'), num_workers=4)
    # fid_score = fid.compute_fid(gt_dir, pr_dir, mode='clean', dataset_res=256, batch_size=100, model_name="clip_vit_b_32", num_workers=1) # Free3D
    # perceptual_model = vgg16(pretrained=True).features
    # perceptual_model.eval()
    pplc = compute_PPLC(pr_dir,models['lpips'], num_images)

    psnrs, ssims, lpipss, l1losses = [], [], [], []
    for k in tqdm(range(num_images)):
        # img_gt_int = imread(os.path.join(gt_dir, f'{k:03}.png'))
        # img_pr_int = imread(os.path.join(pr_dir, f'{k:03}.png'))
        # img_gt = color_map_forward(img_gt_int)
        # img_pr = color_map_forward(img_pr_int)
        img_gt = preprocess_image(models,os.path.join(gt_dir, f'{k:03}.png'),True)
        img_pr = preprocess_image(models,os.path.join(pr_dir, f'{k:03}.png'),False)
        # print(img_gt.shape, img_pr.shape)
        # return
        save_path = os.path.join(pr_dir,'preprocessed')
        # pr_save_path = os.path.join(pr_dir,'preprocessed_pr',f'{k:03}.png')
        # os.makedirs(os.path.join(gt_dir, 'preprocessed_gt'), exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
        Image.fromarray((img_gt*255).astype(np.uint8)).save(os.path.join(save_path,f'gt_{k:03}.png'))
        Image.fromarray((img_pr*255).astype(np.uint8)).save(os.path.join(save_path,f'pr_{k:03}.png'))
        # return
        psnr = compute_psnr_float(img_gt, img_pr)

        with torch.no_grad():
            img_gt_tensor = torch.from_numpy(img_gt.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()
            img_pr_tensor = torch.from_numpy(img_pr.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()
            ssim = float(structural_similarity_index_measure(img_pr_tensor, img_gt_tensor).flatten()[0].cpu().numpy())
            gt_img_th, pr_img_th = img_gt_tensor*2-1, img_pr_tensor*2-1
            score = float(models['lpips'](gt_img_th, pr_img_th).flatten()[0].cpu().numpy())
            l1loss = np.mean(np.abs(img_gt, img_pr))

        ssims.append(ssim)
        lpipss.append(score)
        psnrs.append(psnr)
        l1losses.append(l1loss)
    
    msg=f'name\t psnrs\t ssims\t lpipss\t l1losses'
    print(msg)
    msg=f'{args.name.split("_")[0]}\t {np.mean(psnrs):.5f}\t {np.mean(ssims):.5f}\t {np.mean(lpipss):.5f}\t {np.mean(l1losses):.5f}\t'
    print(msg)
    with open('logs/metrics/nvs.log','a') as f:
        f.write(msg+'\n')
# 生成depth
# resolution of gt

if __name__=="__main__":
    main()