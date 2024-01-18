# MangaLineExtraction_PyTorch
# https://github.com/ljsabc/MangaLineExtraction_PyTorch

#NOTE: This preprocessor is designed to work with lineart_anime ControlNet so the result will be white lines on black canvas

import torch
import numpy as np
import os
import cv2
from einops import rearrange
from .model_torch import res_skip
from PIL import Image

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad

class LineartMangaDetector:
    def __init__(self):
        filename="erika.pth"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)

        net = res_skip()
        ckpt = model_helper.load_torch_file(model_path)
        for key in list(ckpt.keys()):
            if 'module.' in key:
                ckpt[key.replace('module.', '')] = ckpt[key]
                del ckpt[key]
        net.load_state_dict(ckpt)
        net.eval()

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")

        self.model = model_patcher.ModelPatcher(net, load_device, offload_device)

    def __call__(self, input_image):
        detected_map, remove_pad = image_pad(input_image)
        device = self.model.load_device
        model_loader.load_model_gpu(self.model)

        img = cv2.cvtColor(detected_map, cv2.COLOR_RGB2GRAY)
        with torch.no_grad():
            image_feed = torch.from_numpy(img).float().to(device)
            image_feed = rearrange(image_feed, 'h w -> 1 1 h w')

            line = self.model.model(image_feed)
            line = line.cpu().numpy()[0,0,:,:]
            line[line > 255] = 255
            line[line < 0] = 0
            
            line = 255 - line

            line = line.astype(np.uint8)
        
        detected_map = HWC3(line)
        detected_map = remove_pad(255 - detected_map)
            
        return detected_map
