from .network import UNet
from .util import seg2img
import torch
import os
import numpy as np
from einops import rearrange

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad

class AnimeFaceSegmentor:
    def __init__(self):
        filename="UNet.pth"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)

        model = UNet()
        ckpt = model_helper.load_torch_file(model_path)
        model.load_state_dict(ckpt)
        model.eval()

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")

        self.model = model_patcher.ModelPatcher(model, load_device, offload_device)


    def __call__(self, input_image):
        input_image, remove_pad = image_pad(input_image)
        device = self.model.load_device

        with torch.no_grad():
            model_loader.load_model_gpu(self.model)

            image_feed = torch.from_numpy(input_image).float().to(device)
            image_feed = rearrange(image_feed, 'h w c -> 1 c h w')
            image_feed = image_feed / 255
            seg = self.model.model(image_feed).squeeze(dim=0)
            result = seg2img(seg.cpu().detach().numpy())
        
        detected_map = remove_pad(HWC3(result))
                
        return detected_map
