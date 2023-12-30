import os

import cv2
import numpy as np
import torch

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad
from .leres.depthmap import estimateboost, estimateleres
from .leres.multi_depth_model_woauxi import RelDepthModel
from .leres.net_tools import strip_prefix_if_present
from .pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel
from .pix2pix.options.test_options import TestOptions


class LeresDetector:
    def __init__(self):
        filename = "res101.pth"
        filename_pix2pix = "latest_net_G.pth"

        model_path = os.path.join(modules.paths.annotator_models_path, filename)
        checkpoint = model_helper.load_torch_file(model_path)
        model = RelDepthModel(backbone='resnext101')
        model.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."), strict=True)

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")
        self.model = model_patcher.ModelPatcher(model, load_device, offload_device)

        # model_path_pix2pix = os.path.join(modules.paths.annotator_models_path, filename_pix2pix)
        # opt = TestOptions().parse()
        # if not torch.cuda.is_available():
        #     opt.gpu_ids = []  # cpu mode
        # pix2pixmodel = Pix2Pix4DepthModel(opt)
        # pix2pixmodel.save_dir = os.path.dirname(model_path_pix2pix)
        # pix2pixmodel.load_networks('latest')
        # pix2pixmodel.eval()

        # self.pix2pixmodel = pix2pixmodel

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, input_image, thr_a=0, thr_b=0, boost=False):
        detected_map, remove_pad = image_pad(input_image)

        model_loader.load_model_gpu(self.model)

        with torch.no_grad():
            # if boost:
            #     depth = estimateboost(detected_map, self.model.model, 0, self.pix2pixmodel, max(detected_map.shape[1], detected_map.shape[0]))
            # else:
            #     depth = estimateleres(detected_map, self.model.model, detected_map.shape[1], detected_map.shape[0])
            depth = estimateleres(detected_map, self.model.model, detected_map.shape[1], detected_map.shape[0])

            numbytes=2
            depth_min = depth.min()
            depth_max = depth.max()
            max_val = (2**(8*numbytes))-1

            # check output before normalizing and mapping to 16 bit
            if depth_max - depth_min > np.finfo("float").eps:
                out = max_val * (depth - depth_min) / (depth_max - depth_min)
            else:
                out = np.zeros(depth.shape)
            
            # single channel, 16 bit image
            depth_image = out.astype("uint16")

            # convert to uint8
            depth_image = cv2.convertScaleAbs(depth_image, alpha=(255.0/65535.0))

            # remove near
            if thr_a != 0:
                thr_a = ((thr_a/100)*255) 
                depth_image = cv2.threshold(depth_image, thr_a, 255, cv2.THRESH_TOZERO)[1]

            # invert image
            depth_image = cv2.bitwise_not(depth_image)

            # remove bg
            if thr_b != 0:
                thr_b = ((thr_b/100)*255)
                depth_image = cv2.threshold(depth_image, thr_b, 255, cv2.THRESH_TOZERO)[1]  

        detected_map = HWC3(remove_pad(depth_image))

        return detected_map