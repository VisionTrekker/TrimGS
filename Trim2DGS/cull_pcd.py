import torch
import torch.nn.functional as F
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import math

def cull_pcd(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        train_cameras = scene.getTrainCameras()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        with torch.no_grad():
            prune_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            # 遍历所有训练相机
            for view in train_cameras:
                # 高斯 世界坐标系 转换到 NDC
                p_homo = gaussians.get_xyz @ view.full_proj_transform[:3] + view.full_proj_transform[3] # 齐次坐标
                p_ndc = p_homo[:, :2] / (p_homo[:, 3:4] + 1e-6) # 归一化到NDC。(N,2)。范围[-1,1]

                view_mask = (p_ndc[:, 0] >= -1) & (p_ndc[:, 0] <= 1) & (p_ndc[:, 1] >= -1) & (p_ndc[:, 1] <= 1) # -1,1的立方体中
                # F.grid_sample：从输入张量中根据给定的采样网格进行采样。输入张量(N,C,H,W)，采样网格(N,H,W,2)。
                alpha_mask = F.grid_sample(view.gt_alpha_mask[None], p_ndc[view_mask][None, None], mode="nearest", align_corners=True).squeeze() < 0.5
                # 剪枝的mask
                prune_mask[view_mask.nonzero()] |= alpha_mask.unsqueeze(-1)
            gaussians.prune_points_without_optimizer(prune_mask)
        output = os.path.join(dataset.model_path, "point_cloud/iteration_{}".format(iteration), "point_cloud_culled.ply")
        gaussians.save_ply(output)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    cull_pcd(model.extract(args), args.iteration, pipeline.extract(args))