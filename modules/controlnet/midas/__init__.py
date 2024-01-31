import os

import cv2
import numpy as np
import torch
from einops import rearrange

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad
from .api import MiDaSInference

class MidasDetector:
    def __init__(self, model_type="dpt_hybrid"):
        filename = "dpt_hybrid-midas-501f0c75.pt"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)
        model = MiDaSInference(model_type=model_type, model_path=model_path)

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")
        self.model = model_patcher.ModelPatcher(model, load_device, offload_device)
    
    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1, depth_and_normal=False, colored=False):
        detected_map, remove_pad = image_pad(input_image)
        image_depth = detected_map

        model_loader.load_model_gpu(self.model)

        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float()
            image_depth = image_depth.to(self.model.current_device)
            image_depth = image_depth / 127.5 - 1.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model.model(image_depth)[0]

            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            depth_pt = depth_pt.cpu().numpy()
            depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

            if depth_and_normal:
                depth_np = depth.cpu().numpy()
                x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
                y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
                z = np.ones_like(x) * a
                x[depth_pt < bg_th] = 0
                y[depth_pt < bg_th] = 0
                normal = np.stack([x, y, z], axis=2)
                normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
                normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)[:, :, ::-1]
        
        depth_image = HWC3(depth_image)
        if depth_and_normal:
            normal_image = HWC3(normal_image)

        depth_image = remove_pad(depth_image)
        if depth_and_normal:
            normal_image = remove_pad(normal_image)

        if colored:
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_INFERNO)[:, :, ::-1]
        
        if depth_and_normal:
            return depth_image, normal_image
        else:
            return depth_image