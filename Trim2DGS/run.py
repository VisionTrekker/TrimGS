import os

gpu = 2

# DTU数据集
# scene = 'scan24'
# factor = 1
# output_dir = "output/DTU_2DGS"
# tune_output_dir = "output/DTU_Trim2DGS"
# iteration = 7000
# position_lr_init = {"scan63": 0.0000016}
# print("--------------------------------")
# cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s /data2/liuzhi/Dataset_server/dataset_reality/gm_Museum/train -m {output_dir}/{scene} --quiet --test_iterations -1 --depth_ratio 1.0 -r {factor} --lambda_dist 1000"
# print(cmd)
# os.system(cmd)
#
# # cull the points that are not visible in the training set
# print("--------------------------------")
# cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python cull_pcd.py -m {output_dir}/{scene} --iteration 30000"
# print(cmd)
# os.system(cmd)

# print("--------------------------------")
# cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python tune.py -s data/dtu_dataset/DTU/{scene} -m {tune_output_dir}/{scene} --pretrained_ply {output_dir}/{scene}/point_cloud/iteration_30000/point_cloud_culled.ply --quiet --test_iterations -1 --depth_ratio 1.0 -r {factor} --lambda_dist 1000 --densification_interval 100 --contribution_prune_interval 100 --position_lr_init {position_lr_init.get(scene, 0.00016)} --split mix"
# print(cmd)
# os.system(cmd)
#
# print("--------------------------------")
# cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -s data/dtu_dataset/DTU/{scene} -m {tune_output_dir}/{scene} --quiet --skip_train --depth_ratio 1.0 --num_cluster 1 --iteration {iteration} --voxel_size 0.004 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0"
# print(cmd)
# os.system(cmd)

# print("--------------------------------")
# cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python scripts/eval_dtu/evaluate_single_scene.py --input_mesh {tune_output_dir}/{scene}/train/ours_{iteration}/fuse_post.ply --scan_id {scan_id} --output_dir {tune_output_dir}/{scene}/train/ours_{iteration} --mask_dir data/dtu_dataset/DTU --DTU data/dtu_dataset/Official_DTU_Dataset"
# print(cmd)
# os.system(cmd)
#
# print("--------------------------------")
# cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python scripts/eval_dtu_pcd/evaluate_single_scene.py --input_pcd {tune_output_dir}/{scene}/point_cloud/iteration_{iteration}/point_cloud.ply --scan_id {scan_id} --output_dir {tune_output_dir}/{scene}/train/ours_{iteration} --mask_dir data/dtu_dataset/DTU --DTU data/dtu_dataset/Official_DTU_Dataset"
# print(cmd)
# os.system(cmd)

# MipNeRF360数据集
# split = "scale"
#
# print("--------------------------------")
# cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s data/MipNeRF360/{scene} -m {output_dir}/{scene} --eval -i images_{factor} --test_iterations -1 --quiet"
# print(cmd)
# os.system(cmd)
#
# print("--------------------------------")
# cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python tune.py -s data/MipNeRF360/{scene} -m {tune_output_dir}/{scene} --eval -i images_{factor} --pretrained_ply {output_dir}/{scene}/point_cloud/iteration_30000/point_cloud.ply --test_iterations -1 --quiet --split {split} --max_screen_size 100"
# print(cmd)
# os.system(cmd)
#
# print("--------------------------------")
# cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py --iteration {iteration} -m {tune_output_dir}/{scene} --quiet --eval --skip_train"
# print(cmd)
# os.system(cmd)

# print("--------------------------------")
# cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {tune_output_dir}/{scene}"
# print(cmd)
# os.system(cmd)


# 自拍数据集
print("--------------------------------")
# 使用原始2DGS（默认增稠方式）训练30000代
cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s /data2/liuzhi/Dataset_server/dataset_reality/gm_Museum/train -m ./output/gm_Museum -r 1"
# print(cmd)
# os.system(cmd)

# 加载train训练好的高斯模型，增稠方式改为基于scale control的再训练7000代
print("--------------------------------")
cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python tune.py -s /data2/liuzhi/Dataset_server/dataset_reality/gm_Museum/train -m ./output/gm_Museum_trim --pretrained_ply ./output/gm_Museum/point_cloud/iteration_30000/point_cloud.ply -r 1 --split 'scale' --max_screen_size 100"
# print(cmd)
# os.system(cmd)

print("--------------------------------")
cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py --iteration 7000 -m ./output/gm_Museum_trim --skip_train --skip_test --skip_mesh --render_path"
print(cmd)
os.system(cmd)