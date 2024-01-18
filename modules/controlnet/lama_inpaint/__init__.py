# https://github.com/advimman/lama

import os
import warnings

import cv2
import numpy as np
import yaml
import torch
from omegaconf import OmegaConf
from einops import rearrange

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad
from modules import util
from .saicinpainting.training.trainers import make_training_model

class LamaInpaintdetector:
    def __init__(self):
        filename = "ControlNetLama.pth"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)
        sd = model_helper.load_torch_file(model_path)

        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
        config = yaml.safe_load(open(config_path, 'rt'))
        config = OmegaConf.create(config)
        config.training_model.predict_only = True
        config.visualizer.kind = 'noop'
        
        model = make_training_model(config).generator
        model.load_state_dict(sd)

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")
        self.model = model_patcher.ModelPatcher(model, load_device, offload_device)

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, input_image, mask=None):
        input_image, remove_pad = image_pad(input_image)
        mask, remove_pad_mask = image_pad(mask)
        input_mask = mask.copy()

        device = self.model.load_device
        model_loader.load_model_gpu(self.model)

        color = np.ascontiguousarray(input_image[:, :, 0:3]).astype(np.float32) / 255.0
        if mask is None and input_image.shape[3] == 4:
            mask = input_image[:, :, 3:4]
        else:
            mask = input_mask[:, :, :1]
        mask = np.ascontiguousarray(mask).astype(np.float32) / 255.0
        # color = np.ascontiguousarray(input_image[:, :, :3]).astype(np.float32) / 255.0
        # mask = np.ascontiguousarray(input_mask[:, :, :1]).astype(np.float32) / 255.0

        with torch.no_grad():
            color = torch.from_numpy(color).float().to(device)
            mask = torch.from_numpy(mask).float().to(device)
            mask = (mask > 0.5).float()
            color = color * (1 - mask)
            image_feed = torch.cat([color, mask], dim=2)
            image_feed = rearrange(image_feed, 'h w c -> 1 c h w')
            detected_map = self.model.model(image_feed)[0]
            detected_map = rearrange(detected_map, 'c h w -> h w c')
            detected_map = detected_map * mask + color * (1 - mask)
            detected_map *= 255.0
            detected_map = detected_map.detach().cpu().numpy().clip(0, 255).astype(np.uint8)
            detected_map_blur = util.blur(detected_map, 16)
            input_mask = input_mask.astype(np.float32) / 255
            detected_map = (input_image * (1 - input_mask) + detected_map_blur * input_mask).astype(np.uint8)

        detected_map = remove_pad(detected_map)

        return detected_map
