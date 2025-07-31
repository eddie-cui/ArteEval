import os
import shutil
import eval_nvs
import argparse
import cv2

view_ls = range(0,6)
log_dir = r"D:\wyh\InstantMesh\outputs\instant-mesh-large\images\Ecoforms_Plant_Container_FB6_Tur.png"
method_name = "instantmesh"
# view_ls = [11,1,12,2,13,3]
# log_dir = r"D:\Free3D\instantmesh"
# imgs_dir = os.path.join(log_dir, "images")
# mesh_dir = os.path.join(log_dir, "meshes")

# files = os.listdir(imgs_dir)
# for i, f in enumerate(files):
#     if not f.find("Ecoforms_Plant_Container_Quadra_Sand_QP6") == -1:
name = log_dir.split(".")[-1]
# os.makedirs(log_directory,exist_ok=True)
log_directory = os.path.join(r"D:\wyh\InstantMesh\outputs\instant-mesh-large\images",name)

src_dir = rf"D:\wyh\eval_I23\Ecoforms_Plant_Container_FB6_Tur\Ecoforms_Plant_Container_FB6_Tur-gt"
gt_mesh = rf"D:\wyh\eval_I23\Ecoforms_Plant_Container_FB6_Tur\Ecoforms_Plant_Container_FB6_Tur-mesh\meshes\model.obj"
pr_mesh = rf"D:\wyh\InstantMesh\outputs\instant-mesh-large\meshes\Ecoforms_Plant_Container_FB6_Tur.obj"
dstpath = os.path.join(log_directory,"_gt")
os.makedirs(dstpath, exist_ok=True)

# Load the predicted image
image = cv2.imread(os.path.join(log_dir))
# Split the image horizontally into 16 equal parts
height, width, _ = image.shape
split_width = width // 2
split_height = height // 3
split_images = [image[i//2*split_height:(i//2+1)*split_height, i%2*split_width:(i%2+1)*split_width] for i in range(6)]
for i, split_image in enumerate(split_images):
    # Save each split image with a new file name
    cv2.imwrite(os.path.join(log_directory, f'{i:03}.png'), split_image)
# Iterate over each file
for i, view in enumerate(view_ls):
    shutil.copy(os.path.join(src_dir, f'{view:03}.png'), os.path.join(dstpath, f'{i:03}.png')) 

# img_dir = log_directory
# obj_path = os.path.join(mesh_dir, name+".obj")
# testure_path = os.path.join(mesh_dir, name+".png")
# img_gt_dir = dstpath
# obj_gt_path = rf"D:\Free3D\google_views\{name}\mesh\meshes\model.obj"
# testure_gt_path = rf"D:\Free3D\google_views\{name}\mesh\meshes\texture.png"

cmd = 'python eval_nvs.py --gt {} --pr {}  --name {} --num_images {}'.format(dstpath, log_directory, method_name+"_nvs", 6)
print(cmd)
os.system(cmd)

cmd = 'python eval_mesh.py --pr_mesh {} --name {} --camera_info_dir {} --num_images 12  --gt_mesh {} --output logs'.format(pr_mesh, method_name+"_mesh", src_dir, gt_mesh)
print(cmd)
os.system(cmd)
