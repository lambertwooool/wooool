import os

import numpy as np
import torch
from einops import rearrange

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad
from .zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from .zoedepth.utils.config import get_config


class ZoeDetector:
    def __init__(self):
        filename="ZoeD_M12_N.pt"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)

        conf = get_config("zoedepth", "infer")
        model = ZoeDepth.build_from_config(conf)
        model.load_state_dict(model_helper.load_torch_file(model_path)['model'])
        model.eval()

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")

        self.model = model_patcher.ModelPatcher(model, load_device, offload_device)

    def to(self, device):
        self.model.to(device)
        return self
    
    def __call__(self, input_image):
        device = self.model.load_device
        input_image, remove_pad = image_pad(input_image)

        model_loader.load_model_gpu(self.model)

        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().to(device)
            image_depth = image_depth / 255.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model.model.infer(image_depth)

            depth = depth[0, 0].cpu().numpy()

            vmin = np.percentile(depth, 2)
            vmax = np.percentile(depth, 85)

            depth -= vmin
            depth /= vmax - vmin
            depth = 1.0 - depth
            depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = remove_pad(HWC3(depth_image))
            
        return detected_map
