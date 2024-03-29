import torch
import os
import numpy as np
from .depthfm import DepthFM

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad

class DepthFMDetector:
    def __init__(self):
        filename = "depthfm-v1.ckpt"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)

        model = DepthFM(model_path)
        model.eval()

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")

        self.model = model
        self.model_wrap = model_patcher.ModelPatcher(model.model, load_device, offload_device)


    def __call__(self, input_image, steps=2, ensemble_size=2):
        input_image, remove_pad = image_pad(input_image)
        device = self.model_wrap.load_device

        with torch.no_grad():
            model_loader.load_model_gpu(self.model_wrap)
            img_tensor = util.numpy_to_pytorch(input_image)
            img_tensor = image.permute(0, 3, 1, 2).to(device=device)
            depth = self.model.predict_depth(img_tensor, num_steps=steps, ensemble_size=ensemble_size)
            depth_map = depth.squeeze(0)
        
            if depth_map.dim() == 3:
                depth_map = depth_map.squeeze(0)

            depth_map = depth_map.unsqueeze(-1).repeat(1, 1, 3)
            depth_map = depth_map.unsqueeze(0)
            depth_map = 1.0 - depth_map

            detected_map = util.pytorch_to_numpy(depth_map)
        
        detected_map = remove_pad(HWC3(detected_map))
                
        return detected_map
