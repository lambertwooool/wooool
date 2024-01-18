import torch
import os
import numpy as np
from .network import network

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad

class RemoveBackgroundDetector:
    def __init__(self):
        filename="isnetis.ckpt"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)

        model = network(model_path)
        model.net.eval()

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")

        self.model = model
        self.model_wrap = model_patcher.ModelPatcher(model.net, load_device, offload_device)


    def __call__(self, input_image):
        input_image, remove_pad = image_pad(input_image)
        device = self.model_wrap.load_device

        with torch.no_grad():
            model_loader.load_model_gpu(self.model_wrap)
            mask, detected_map = self.model(input_image, device, 0)
        
        detected_map = remove_pad(HWC3(detected_map))

        mask = remove_pad(mask)
        H, W, C = detected_map.shape
        tmp = np.zeros([H, W, C + 1])
        tmp[:,:,:C] = detected_map
        tmp[:,:,3:] = mask
        detected_map = tmp.astype(np.uint8)
                
        return detected_map
