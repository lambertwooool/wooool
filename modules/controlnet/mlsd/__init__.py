import os
import warnings

import cv2
import numpy as np
import torch

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad
from .models.mbv2_mlsd_large import MobileV2_MLSD_Large
from .utils import pred_lines


class MLSDdetector:
    def __init__(self):
        filename = "mlsd_large_512_fp32.pth"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)

        model = MobileV2_MLSD_Large()
        model.load_state_dict(model_helper.load_torch_file(model_path), strict=True)
        model.eval()

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")

        self.model = model_patcher.ModelPatcher(model, load_device, offload_device)

    def to(self, device):
        self.model.to(device)
        return self
    
    def __call__(self, input_image, thr_v=0.1, thr_d=0.1):
        detected_map, remove_pad = image_pad(input_image)
        model_loader.load_model_gpu(self.model)
        img = detected_map
        img_output = np.zeros_like(img)
        try:
            with torch.no_grad():
                lines = pred_lines(img, self.model.model, [img.shape[0], img.shape[1]], thr_v, thr_d)
                for line in lines:
                    x_start, y_start, x_end, y_end = [int(val) for val in line]
                    cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
        except Exception as e:
            pass

        detected_map = remove_pad(HWC3(img_output[:, :, 0]))
            
        return detected_map
