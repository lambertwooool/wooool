import os
import torchvision # Fix issue Unknown builtin op: torchvision::nms
import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad
from .densepose import DensePoseMaskedColormapResultsVisualizer, _extract_i_from_iuvarr, densepose_chart_predictor_output_to_result_with_confidences

N_PART_LABELS = 24

class DenseposeDetector:
    def __init__(self):
        filename="densepose_r50_fpn_dl.torchscript"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)

        model = model_helper.load_torch_file(model_path)

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")
        
        self.dense_pose_estimation = model
        self.dense_pose_estimation_wrap = model_patcher.ModelPatcher(model.model, load_device, offload_device)
        self.result_visualizer = DensePoseMaskedColormapResultsVisualizer(
            alpha=1, 
            data_extractor=_extract_i_from_iuvarr, 
            segm_extractor=_extract_i_from_iuvarr, 
            val_scale = 255.0 / N_PART_LABELS
        )

    def to(self, device):
        self.dense_pose_estimation.to(device)
        self.device = device
        return self
    
    def __call__(self, input_image, cmap="viridis"):
        input_image, remove_pad = image_pad(input_image)
        H, W  = input_image.shape[:2]

        hint_image_canvas = np.zeros([H, W], dtype=np.uint8)
        hint_image_canvas = np.tile(hint_image_canvas[:, :, np.newaxis], [1, 1, 3])

        device = self.dense_pose_estimation_wrap.load_device
        input_image = rearrange(torch.from_numpy(input_image).to(device), 'h w c -> c h w')

        model_loader.load_model_gpu(self.dense_pose_estimation_wrap)

        pred_boxes, corase_segm, fine_segm, u, v = self.dense_pose_estimation(input_image)

        extractor = densepose_chart_predictor_output_to_result_with_confidences
        densepose_results = [extractor(pred_boxes[i:i+1], corase_segm[i:i+1], fine_segm[i:i+1], u[i:i+1], v[i:i+1]) for i in range(len(pred_boxes))]

        if cmap=="viridis":
            self.result_visualizer.mask_visualizer.cmap = cv2.COLORMAP_VIRIDIS
            hint_image = self.result_visualizer.visualize(hint_image_canvas, densepose_results)
            hint_image = cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB)
            hint_image[:, :, 0][hint_image[:, :, 0] == 0] = 68
            hint_image[:, :, 1][hint_image[:, :, 1] == 0] = 1
            hint_image[:, :, 2][hint_image[:, :, 2] == 0] = 84
        else:
            self.result_visualizer.mask_visualizer.cmap = cv2.COLORMAP_PARULA
            hint_image = self.result_visualizer.visualize(hint_image_canvas, densepose_results)
            hint_image = cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB)

        detected_map = remove_pad(HWC3(hint_image))

        return detected_map
