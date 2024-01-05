# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Union

import numpy as np
from PIL import Image

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad
from .automatic_mask_generator import SamAutomaticMaskGenerator
from .build_sam import sam_model_registry


class SamDetector:
    def __init__(self):
        """
        Possible model_type : vit_h, vit_l, vit_b, vit_t
        download weights from https://github.com/facebookresearch/segment-anything
        """
        model_type="vit_t"
        filename="mobile_sam.pt"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)

        sam = sam_model_registry[model_type](checkpoint=model_path)
        mask_generator = SamAutomaticMaskGenerator(sam)

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")

        self.model = model_patcher.ModelPatcher(sam, load_device, offload_device)
        self.mask_generator = mask_generator
    

    def to(self, device):
        model = self.mask_generator.predictor.model.to(device)
        model.train(False) #Update attention_bias in https://github.com/Fannovel16/comfyui_controlnet_aux/blob/main/src/controlnet_aux/segment_anything/modeling/tiny_vit_sam.py#L251
        self.mask_generator = SamAutomaticMaskGenerator(model)
        return self


    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        h, w =  anns[0]['segmentation'].shape
        final_img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.empty((m.shape[0], m.shape[1], 3), dtype=np.uint8)
            for i in range(3):
                img[:,:,i] = np.random.randint(255, dtype=np.uint8)
            final_img.paste(Image.fromarray(img, mode="RGB"), (0, 0), Image.fromarray(np.uint8(m*255)))
        
        return np.array(final_img, dtype=np.uint8)

    def __call__(self, input_image):
        device = self.model.load_device
        input_image, remove_pad = image_pad(input_image)

        model_loader.load_model_gpu(self.model)

        # Generate Masks
        masks = self.mask_generator.generate(input_image)
        # Create map
        map = self.show_anns(masks)

        detected_map = HWC3(remove_pad(map))

        return detected_map
