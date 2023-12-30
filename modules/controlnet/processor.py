"""
This file contains a Processor that can be used to process images with controlnet processors
"""
import io
import logging
import os
import torch
from typing import Dict, Optional, Union

from PIL import Image

from .canny import CannyDetector
from .hed import HEDdetector
from .midas import MidasDetector
from .leres import LeresDetector
from .dwpose import DwposeDetector
from .shuffle import ContentShuffleDetector
from .ipadapter import IPAdapterDetector

import modules.paths
from modules.model import controlnet, model_loader, model_patcher

LOGGER = logging.getLogger(__name__)

MODELS = {
    # checkpoint models
    'canny': { 'class': CannyDetector },

    'scribble_hed': { 'class': HEDdetector },
    'softedge_hed': { 'class': HEDdetector },
    'scribble_hedsafe': { 'class': HEDdetector },
    'softedge_hedsafe': { 'class': HEDdetector },

    'depth_midas': { 'class': MidasDetector },
    # 'depth_zoe': { 'class': ZoeDetector }, 
    'depth_leres': { 'class': LeresDetector }, 
    'depth_leres++': { 'class': LeresDetector },

    'dwpose': { 'class': DwposeDetector },

    'shuffle': { 'class': ContentShuffleDetector },

    'ip_adapter': { 'class': IPAdapterDetector },
    'ip_adapter_face': { 'class': IPAdapterDetector },
}

MODEL_PARAMS = {
    'canny': { 'low_threshold': 64, 'high_threshold': 128 },

    'scribble_hed': { 'scribble': True, 'safe': False },
    'softedge_hed': { 'scribble': False, 'safe': False },
    'scribble_hedsafe': { 'scribble': True, 'safe': True },
    'softedge_hedsafe': { 'scribble': False, 'safe': True },

    'depth_midas': {},
    'depth_zoe': {},
    'depth_leres': { 'boost': False },
    'depth_leres++': { 'boost': True },

    'dwpose': { 'include_hand': True, 'include_face': True },

    'shuffle': { 'f': 512 },

    'ip_adapter': { },
    'ip_adapter_face': { },
}

class Processor:
    def __init__(self, processor_id: str, params: Optional[Dict] = None) -> None:
        """Processor that can be used to process images with controlnet aux processors

        Args:
            processor_id (str): processor name, options are 'hed, midas, mlsd, openpose,
                                pidinet, normalbae, lineart, lineart_coarse, lineart_anime,
                                canny, content_shuffle, zoe, mediapipe_face, tile'
            params (Optional[Dict]): parameters for the processor
        """
        LOGGER.info("Loading %s".format(processor_id))

        if processor_id not in MODELS:
            raise ValueError(f"{processor_id} is not a valid processor id. Please make sure to choose one of {', '.join(MODELS.keys())}")

        self.processor_id = processor_id
        self.processor = self.load_processor(self.processor_id)
        self.controlnet = None

        # load default params
        self.params = MODEL_PARAMS[self.processor_id]
        # update with user params
        if params:
            self.params.update(params)

    def model_keys():
        return MODELS.keys()

    def load_processor(self, processor_id: str) -> 'Processor':
        """Load controlnet processors

        Args:
            processor_id (str): processor name

        Returns:
            Processor: controlnet processor
        """

        processor = MODELS[processor_id]['class']()
        return processor

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, image: Union[Image.Image, bytes]) -> Image.Image:
        """processes an image with a controlnet aux processor

        Args:
            image (Union[Image.Image, bytes]): input image in bytes or PIL Image

        Returns:
            Image.Image: processed image
        """
        # check if bytes or PIL Image
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")

        processed_image = self.processor(image, **self.params)

        return processed_image
    
    def load_controlnet(self, filename: str):
        filename = os.path.join(modules.paths.controlnet_models_path, filename)
        controlnet_model = controlnet.load_controlnet(filename)
        self.controlnet = controlnet_model
        return self.controlnet
