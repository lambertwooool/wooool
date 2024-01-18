import os
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad

norm_layer = nn.InstanceNorm2d

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


class LineartDetector:
    def __init__(self):
        filename = "sk_model.pth"
        coarse_filename = "sk_model2.pth"

        model_path = os.path.join(modules.paths.annotator_models_path, filename)
        model_coarse_path = os.path.join(modules.paths.annotator_models_path, coarse_filename)

        model = Generator(3, 1, 3)
        model_sd = model_helper.load_torch_file(model_path)
        model.load_state_dict(model_sd)
        model.eval()

        model_coarse = Generator(3, 1, 3)
        model_coarse_sd = model_helper.load_torch_file(model_coarse_path)
        model_coarse.load_state_dict(model_coarse_sd)
        model_coarse.eval()

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")

        self.model = model_patcher.ModelPatcher(model, load_device, offload_device)
        self.model_coarse = model_patcher.ModelPatcher(model_coarse, load_device, offload_device)
        
    
    def __call__(self, input_image, coarse):
        assert input_image.ndim == 3

        device = self.model.load_device
        model = self.model_coarse if coarse else self.model
        model_loader.load_model_gpu(model)

        detected_map, remove_pad = image_pad(input_image)
        
        with torch.no_grad():
            image = torch.from_numpy(detected_map).float().to(device)
            image = image / 255.0
            image = rearrange(image, 'h w c -> 1 c h w')
            line = model.model(image)[0][0]

            line = line.cpu().numpy()
            line = (line * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = remove_pad(HWC3(line))
            
        return detected_map
