import cv2
import numpy as np

import modules.paths
from modules.model import model_helper, model_loader, model_patcher
from modules.util import HWC3, image_pad

class MeshGraphormerDetector:
    def __init__(self):
        filename="graphormer_hand_state_dict.bin"
        hrnet_filename="hrnetv2_w64_imagenet_pretrained.pth"

        model_path = os.path.join(modules.paths.annotator_models_path, filename)
        hrnet_path = os.path.join(modules.paths.annotator_models_path, hrnet_filename)

        args.resume_checkpoint = model_helper.load_torch_file(model_path)
        args.hrnet_checkpoint = model_helper.load_torch_file(hrnet_path)
        args.seed = seed
        pipeline = MeshGraphormerMediapipe(args)

        load_device = model_loader.run_device("annotator")
        offload_device = model_loader.offload_device("annotator")
        self.pipeline = model_patcher.ModelPatcher(pipeline, load_device, offload_device)

    # @classmethod
    # def from_pretrained(cls, pretrained_model_or_path=MESH_GRAPHORMER_MODEL_NAME, filename="graphormer_hand_state_dict.bin", hrnet_filename="hrnetv2_w64_imagenet_pretrained.pth", seed=88):
    #     args.resume_checkpoint = custom_hf_download(pretrained_model_or_path, filename)
    #     args.hrnet_checkpoint = custom_hf_download(pretrained_model_or_path, hrnet_filename)
    #     args.seed = seed
    #     pipeline = MeshGraphormerMediapipe(args)
    #     return cls(pipeline)
    
    # def to(self, device):
    #     self.pipeline._model.to(device)
    #     self.pipeline.mano_model.to(device)
    #     self.pipeline.mano_model.layer.to(device)
    #     return self

    def __call__(self, input_image=None, mask_bbox_padding=30):
        device = self.model.load_device
        model_loader.load_model_gpu(self.model)

        depth_map, mask, info = self.pipeline.model.get_depth(input_image, mask_bbox_padding)
        if depth_map is None:
            depth_map = np.zeros_like(input_image)
            mask = np.zeros_like(input_image)

        #The hand is small
        mask = HWC3(mask)
        depth_map, remove_pad = image_pad(depth_map)
        depth_map = remove_pad(depth_map)
            
        return depth_map, mask, info
