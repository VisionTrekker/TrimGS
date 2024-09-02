import os

gpu = -

# MipNeRF360数据集
factors = [4, 2, 2, 4, 4, 4, 4, 2, 2]

normal_weights = [0.1, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01]

excluded_gpus = set([])

split = "scale"
output_dir = "output/MipNeRF360_3DGS"
tune_output_dir = f"output/MipNeRF360_Trim3DGS"
iteration = 7000

print("--------------------------")
cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s data/MipNeRF360/{scene} -m {output_dir}/{scene} --eval -i images_{factor} --test_iterations -1 --quiet"
print(cmd)
os.system(cmd)

print("--------------------------")
cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python tune.py -s data/MipNeRF360/{scene} -m {tune_output_dir}/{scene} --eval -i images_{factor} --pretrained_ply {output_dir}/{scene}/point_cloud/iteration_30000/point_cloud.ply --test_iterations -1 --quiet --split {split} --position_lr_init 0.0000016 --densification_interval 1000 --opacity_reset_interval 999999 --normal_regularity_param {weight} --contribution_prune_from_iter 0 --contribution_prune_interval 1000"
print(cmd)
os.system(cmd)

print("--------------------------")
cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py --iteration {iteration} -m {tune_output_dir}/{scene} --eval --skip_train --render_other"
print(cmd)
os.system(cmd)

print("--------------------------")
cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {tune_output_dir}/{scene}"
print(cmd)
os.system(cmd)

# print("--------------------------")
# cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python extract_mesh_tsdf.py -s data/MipNeRF360/{scene} -m {tune_output_dir}/{scene} --eval -i images_{factor} --iteration {iteration} --voxel_size 0.004 --sdf_trunc 0.04"
# print(cmd)
# os.system(cmd)