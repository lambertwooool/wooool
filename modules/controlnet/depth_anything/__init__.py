import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dpt import DPT_DINOv2
from .util.transform import Resize, NormalizeImage, PrepareForNet

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad

transform = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)

class DepthAnythingDetector:
    """https://github.com/LiheYoung/Depth-Anything"""

    def __init__(self):
        filename = "depth_anything_vitl14.pth"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)
        state_dict = model_helper.load_torch_file(model_path)

        model = DPT_DINOv2()

        model.load_state_dict(state_dict)

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")

        self.model = model_patcher.ModelPatcher(model, load_device, offload_device)

    def __call__(self, input_image, colored: bool = False) -> np.ndarray:
        detected_map, remove_pad = image_pad(input_image)
        device = self.model.load_device
        model_loader.load_model_gpu(self.model)

        h, w = detected_map.shape[:2]

        image = cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        @torch.no_grad()
        def predict_depth(model, image):
            return model(image)
        depth = predict_depth(self.model.model, image)
        depth = F.interpolate(
            depth[None], (h, w), mode="bilinear", align_corners=False
        )[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)

        detected_map = HWC3(depth)
        detected_map = remove_pad(detected_map)

        if colored:
            return cv2.applyColorMap(detected_map, cv2.COLORMAP_INFERNO)[:, :, ::-1]
        else:
            return detected_map
