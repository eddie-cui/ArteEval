
# python evaluator.py --mode single --gen_mesh E:\yl\1\CRM\CRM\724_output_mesh\2\000.obj --gt_mesh E:\potterylike_dataset\tmp\2\Now_Designs_Bowl_Akita_Black\meshes\model.obj  --gen_rotate_x 90 --gt_rotate_x -90
# python evaluator.py --mode single --gen_mesh E:\yl\1\CRM\CRM\724_output_mesh\25\000.obj --gt_mesh E:\potterylike_dataset\tmp\2\Now_Designs_Bowl_Akita_Black\meshes\model.obj  --gen_rotate_x 90 --gt_rotate_x -90

python evaluator.py --mode dataset --gen_root E:\yl\1\CRM\CRM\724_output_mesh --gt_root E:\potterylike_dataset\tmp  --gen_rotate_x 90 --gt_rotate_x -90 --model_ids '25,27,30,39,41,42,45,46,52,55'
# python evaluator.py --mode dataset --gen_root E:\yl\1\CRM\CRM\724_output_mesh --gt_root E:\potterylike_dataset\tmp  --gen_rotate_x 90 --gt_rotate_x -90 --model_ids '1,2,8,25,27,30,39,41,42,45,46,52,55'
python evaluator.py --mode dataset --gen_root E:\yl\1\CRM\CRM\724_output_mesh --gt_root E:\potterylike_dataset\tmp --gt_rotate_y 90  --gen_rotate_x 90 --gen_rotate_y 90  --model_ids '27'
python evaluator.py --mode dataset --gen_root E:\yl\1\CRM\CRM\724_output_mesh --gt_root E:\potterylike_dataset\tmp --gt_rotate_y 90  --gen_rotate_x 90 --gen_rotate_y 270  --model_ids '27'

python evaluator.py --mode dataset --gen_root E:\yl\1\CRM\CRM\724_output_mesh --gt_root E:\potterylike_dataset\tmp --gt_rotate_y 90 --gen_rotate_y 180 --model_ids '41'

python evaluator.py --mode dataset --gen_root E:\yl\1\CRM\CRM\724_output_mesh --gt_root E:\potterylike_dataset\tmp --gt_rotate_x -90 --gen_rotate_x 180 --gen_rotate_z 180 --model_ids '2'

python evaluator.py --mode dataset --gen_root E:\yl\1\CRM\CRM\724_output_mesh --gt_root E:\potterylike_dataset\tmp --gt_rotate_y 90  --gen_rotate_y 180 --model_ids '27,30,39,42,46'

python evaluator.py --mode dataset --gen_root E:\yl\1\CRM\CRM\724_output_mesh --gt_root E:\potterylike_dataset\tmp --gt_rotate_y -90 --gen_rotate_y 90 --model_ids '45'

python evaluator.py --mode dataset --gen_root E:\yl\1\CRM\CRM\724_output_mesh --gt_root E:\potterylike_dataset\tmp --gt_rotate_y 90 --model_ids '52'

python evaluator.py --mode dataset --gen_root E:\yl\1\CRM\CRM\724_output_mesh --gt_root E:\potterylike_dataset\tmp --gen_rotate_y -90 --model_ids '55'

# 1 gt
# 2 gt
# 8 gt
# 25 gen
# 27 gen
# 30 gen
# 39 gen
# 42 gen
# 42 gt
# 45 gen
# 45 gt
# 46 gen
# 42 gt
# 52 gt

python evaluator.py --mode dataset --gen_root D:\chz\free3d_output\mesh --gt_root E:\potterylike_dataset\tmp --gen_rotate_y -90 --model_ids '46'

            python evaluator.py --mode specific_models  --gen_root D:\chz\free3d_output\mesh  --gt_root E:\potterylike_dataset\tmp  --gt_rotate_y -90  --gen_rotate_x -90 --gen_rotate_y -90 --model_ids 55  --output results_1.json