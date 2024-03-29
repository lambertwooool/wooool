import torch
import torch.nn.functional as F
import os
import numpy as np
from .briarmbg import BriaRMBG
from torchvision.transforms.functional import normalize

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad

def get_mask(rmbgmodel, input_image, device):
    h, w = input_image.shape[:2]
    im_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2,0,1)
    im_tensor = torch.unsqueeze(im_tensor,0)
    im_tensor = torch.divide(im_tensor,255.0)
    im_tensor = normalize(im_tensor,[0.5,0.5,0.5],[1.0,1.0,1.0])
    im_tensor.to(device)

    result = rmbgmodel(im_tensor)
    result = torch.squeeze(F.interpolate(result[0][0], size=(h, w), mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    mask = ((result-mi)/(ma-mi)).permute(1,2,0).numpy()
    np_img = (mask * input_image + 255 * (1 - mask)).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)
    
    return mask, np_img

class BriaRemoveBackgroundDetector:
    def __init__(self):
        filename = "rmbg14.safetensors"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)

        model = BriaRMBG()
        state_dict = model_helper.load_torch_file(model_path)
        model.load_state_dict(state_dict)

        load_device, offload_device, dtype, manual_cast_dtype = model_loader.get_device_and_dtype("annotator")

        self.model = model_patcher.ModelPatcher(model, load_device, offload_device)


    def __call__(self, input_image):
        input_image, remove_pad = image_pad(input_image)
        device = self.model.load_device

        with torch.no_grad():
            model_loader.load_model_gpu(self.model)
            mask, detected_map = get_mask(self.model.model, input_image, device)
        
        detected_map = remove_pad(HWC3(detected_map))

        mask = remove_pad(mask)
        H, W, C = detected_map.shape
        tmp = np.zeros([H, W, C + 1])
        tmp[:,:,:C] = detected_map
        tmp[:,:,3:] = mask
        detected_map = tmp.astype(np.uint8)
                
        return detected_map
