import os

from .api import make_detectron2_model, semantic_run
import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad

DEFAULT_CONFIGS = {
    "coco": {
        "name": "150_16_swin_l_oneformer_coco_100ep.pth",
        "config": os.path.join(os.path.dirname(__file__), 'configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml')
    },
    "ade20k": {
        "name": "250_16_swin_l_oneformer_ade20k_160k.pth",
        "config": os.path.join(os.path.dirname(__file__), 'configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml')
    }
}

class OneformerSegmentor:
    def __init__(self):
        filename = "250_16_swin_l_oneformer_ade20k_160k.pth"
        model_path = os.path.join(modules.paths.annotator_models_path, filename)
        config_path = DEFAULT_CONFIGS["ade20k" if "ade20k" in filename else "coco"]["config"]

        model, metadata = make_detectron2_model(config_path, model_path)

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")

        self.model = model
        self.model_wrap = model_patcher.ModelPatcher(model.model, load_device, offload_device)
        self.metadata = metadata

    def to(self, device):
        self.model.model.to(device)
        return self
    
    def __call__(self, input_image):
        detected_map, remove_pad = image_pad(input_image)
        model_loader.load_model_gpu(self.model_wrap)

        detected_map = semantic_run(input_image, self.model, self.metadata)
        detected_map = remove_pad(HWC3(detected_map))
            
        return detected_map
