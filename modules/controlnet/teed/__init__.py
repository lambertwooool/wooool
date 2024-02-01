import os
import cv2
import numpy as np

import torch
from einops import rearrange

from .ted import TED  # TEED architecture
import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad

def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y

class TEEDDector:
    """https://github.com/xavysp/TEED"""

    def __init__(self):
        filename="7_model.pth"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)

        state_dict = model_helper.load_torch_file(model_path)
        model = TED()
        model.load_state_dict(state_dict)
        model.eval()

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")

        self.model = model_patcher.ModelPatcher(model, load_device, offload_device)

    def __call__(self, input_image, safe_steps: int = 2):
        detected_map, remove_pad = image_pad(input_image)
        device = self.model.load_device
        model_loader.load_model_gpu(self.model)

        H, W, _ = detected_map.shape
        with torch.no_grad():
            image_teed = torch.from_numpy(detected_map).float().to(device)
            image_teed = rearrange(image_teed, 'h w c -> 1 c h w')
            edges = self.model.model(image_teed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if safe_steps != 0:
                edge = safe_step(edge, safe_steps)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
            
        detected_map = HWC3(edge)
        detected_map = remove_pad(detected_map)
            
        return detected_map