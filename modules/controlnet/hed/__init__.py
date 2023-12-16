# This is an improved version and model of HED edge detection with Apache License, Version 2.0.
# Please use this implementation in your products
# This implementation may produce slightly different results from Saining Xie's official implementations,
# but it generates smoother edges and is more suitable for ControlNet as well as other image-to-image translations.
# Different from official models and other implementations, this is an RGB-input model (rather than BGR)
# and in this way it works better for gradio's RGB protocol

import os
import warnings

import cv2
import numpy as np
import torch
from einops import rearrange

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import nms, safe_step

class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        for i in range(1, layer_number):
            self.convs.append(torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.projection = torch.nn.Conv2d(in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5

class HEDdetector:
    def __init__(self):
        filename = "ControlNetHED.pth"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)
        model = model_helper.load_torch_file(model_path)
        netNetwork = ControlNetHED_Apache2()
        netNetwork.load_state_dict(model)

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")
        self.netNetwork = model_patcher.ModelPatcher(netNetwork, load_device, offload_device)

        # self.netNetwork = netNetwork

    # @classmethod
    # def from_pretrained(cls, pretrained_model_or_path, filename=None, cache_dir=None):
    #     filename = filename or "ControlNetHED.pth"
    #     model_path = os.path.join(pretrained_model_or_path, filename)

    #     netNetwork = ControlNetHED_Apache2()
    #     netNetwork.load_state_dict(torch.load(model_path, map_location='cpu'))
    #     netNetwork.float().eval()

    #     return cls(netNetwork)
    
    def to(self, device):
        self.netNetwork.to(device)
        return self

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, input_image, safe, scribble):
        assert input_image.ndim == 3
        H, W, C = input_image.shape

        model_loader.load_model_gpu(self.model)

        with torch.no_grad():
            # device = next(iter(self.netNetwork.parameters())).device
            device = self.netNetwork.current_device
            image_hed = torch.from_numpy(input_image).float().to(device)
            image_hed = rearrange(image_hed, 'h w c -> 1 c h w')
            # edges = self.netNetwork(image_hed)
            edges = self.netNetwork.model(image_hed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if safe:
                edge = safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = edge
        
        if scribble:
            detected_map = nms(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0

        return detected_map
