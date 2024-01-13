from .network import UNet
from .util import seg2img
import torch
import os
import cv2
import numpy as np
from einops import rearrange
from .anime_segmentation import AnimeSegmentation

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad

class AnimeFaceSegmentor:
    def __init__(self, model, seg_model):
        filename="UNet.pth"
        seg_filename="isnetis.ckpt"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)
        seg_path = os.path.join(modules.paths.annotator_models_path, seg_filename)

        model = UNet()
        ckpt = model_helper.load_torch_file(model_path)
        model.load_state_dict(ckpt)
        model.eval()

        seg_model = AnimeSegmentation(seg_model_path)
        seg_model.net.eval()

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")

        self.model = model_patcher.ModelPatcher(model, load_device, offload_device)
        self.seg_model_wrap = model_patcher.ModelPatcher(seg_model.net, load_device, offload_device)


    def __call__(self, input_image, remove_background=True):
        input_image, remove_pad = image_pad(input_image)
        device = self.model.load_device

        with torch.no_grad():
            if remove_background:
                model_loader.load_model_gpu(self.seg_model_wrap)
                mask, input_image = self.seg_model.model(input_image, 0) #Don't resize image as it is resized
            image_feed = torch.from_numpy(input_image).float().to(device)
            image_feed = rearrange(image_feed, 'h w c -> 1 c h w')
            image_feed = image_feed / 255
            seg = self.model.model(image_feed).squeeze(dim=0)
            result = seg2img(seg.cpu().detach().numpy())
        
        detected_map = remove_pad(HWC3(hint_image))

        if remove_background:
            mask = remove_pad(mask)
            H, W, C = detected_map.shape
            tmp = np.zeros([H, W, C + 1])
            tmp[:,:,:C] = detected_map
            tmp[:,:,3:] = mask
            detected_map = tmp
                
        return detected_map
