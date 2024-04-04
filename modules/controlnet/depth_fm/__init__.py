import torch
import os
import numpy as np
from .depthfm import DepthFM

import modules.paths
from modules import util
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad

class DepthFMDetector:
    def __init__(self):
        filename = "depthfm-v1.ckpt"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)

        load_device, offload_device, dtype, manual_cast_dtype = model_loader.get_device_and_dtype("annotator")
        model = DepthFM(model_path)
        model.to(dtype)

        self.model = model
        self.dtype = dtype
        self.model_wrap = model_patcher.ModelPatcher(model.model, load_device, offload_device)
        self.vae_wrap = model_patcher.ModelPatcher(model.vae, load_device, offload_device)


    def __call__(self, input_image, steps=4, ensemble_size=2):
        input_image, remove_pad = image_pad(input_image)
        device = self.model_wrap.load_device

        with torch.no_grad():
            model_loader.load_models_gpu([self.model_wrap, self.vae_wrap])
            img_tensor = util.numpy_to_pytorch(input_image)
            img_tensor = img_tensor.permute(0, 3, 1, 2).to(device=device, dtype=self.dtype)
            depth = self.model.predict_depth(img_tensor, num_steps=steps, ensemble_size=ensemble_size)
            depth_map = depth.squeeze(0)
        
            if depth_map.dim() == 3:
                depth_map = depth_map.squeeze(0)

            depth_map = depth_map.unsqueeze(-1).repeat(1, 1, 3)
            depth_map = depth_map.unsqueeze(0)
            depth_map = 1.0 - depth_map

            detected_map = util.pytorch_to_numpy(depth_map)[0]
        
        detected_map = remove_pad(HWC3(detected_map))
                
        return detected_map
