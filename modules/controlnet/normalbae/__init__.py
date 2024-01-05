import os
import types
import warnings

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from einops import rearrange

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad
from .nets.NNET import NNET


# load model
def load_checkpoint(fpath, model):
    ckpt = model_helper.load_torch_file(fpath)['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model

class NormalBaeDetector:
    def __init__(self):
        filename = "scannet.pt"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)

        args = types.SimpleNamespace()
        args.mode = 'client'
        args.architecture = 'BN'
        args.pretrained = 'scannet'
        args.sampling_ratio = 0.4
        args.importance_ratio = 0.7
        model = NNET(args)
        model = load_checkpoint(model_path, model)
        model.eval()

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")

        self.model = model_patcher.ModelPatcher(model, load_device, offload_device)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def to(self, device):
        self.model.to(device)
        return self


    def __call__(self, input_image):
        detected_map, remove_pad = image_pad(input_image)
        device = self.model.load_device
        model_loader.load_model_gpu(self.model)
        image_normal = detected_map
        with torch.no_grad():
            image_normal = torch.from_numpy(image_normal).float().to(device)
            image_normal = image_normal / 255.0
            image_normal = rearrange(image_normal, 'h w c -> 1 c h w')
            image_normal = self.norm(image_normal)

            normal = self.model.model(image_normal)
            normal = normal[0][-1][:, :3]
            # d = torch.sum(normal ** 2.0, dim=1, keepdim=True) ** 0.5
            # d = torch.maximum(d, torch.ones_like(d) * 1e-5)
            # normal /= d
            normal = ((normal + 1) * 0.5).clip(0, 1)

            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()
            normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = remove_pad(HWC3(normal_image))
            
        return detected_map
    