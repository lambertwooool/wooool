"""
This file contains a Processor that can be used to process images with controlnet processors
"""
import io
import logging
import os
import torch
from typing import Any, Dict, Optional, Union

import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "custom_repo"))

from PIL import Image

from .invert import InvertDetector
from .binary import BinaryDetector
from .color import ColorDetector
from .canny import CannyDetector
from .hed import HEDdetector
from .midas import MidasDetector
from .leres import LeresDetector
from .zoe import ZoeDetector
from .mlsd import MLSDdetector
from .lineart import LineartDetector
from .lineart_anime import LineartAnimeDetector
from .manga_line import LineartMangaDetector
from .dwpose import DwposeDetector, AnimalposeDetector
from .densepose import DenseposeDetector
from .shuffle import ContentShuffleDetector
from .ipadapter import IPAdapterDetector
from .tile import TileDetector
from .normalbae import NormalBaeDetector
from .oneformer import OneformerSegmentor
from .sam import SamDetector
from .lama_inpaint import LamaInpaintdetector
from .anime_face_segment import AnimeFaceSegmentor
from .remove_bg import RemoveBackgroundDetector

import modules.paths
from modules.model import controlnet, model_loader, model_patcher

LOGGER = logging.getLogger(__name__)

class DefaultDetector:
    def __init__(self):
        pass

    def __call__(self, input_image):
        return input_image

MODELS = {
    # checkpoint models
    'default': { 'class': DefaultDetector },
    'invert': { 'class': InvertDetector },
    'binary': { 'class': BinaryDetector },
    'color': { 'class': ColorDetector },
    'canny': { 'class': CannyDetector },
    'scribble_hed': { 'class': HEDdetector },
    'softedge_hed': { 'class': HEDdetector },
    'scribble_hedsafe': { 'class': HEDdetector },
    'softedge_hedsafe': { 'class': HEDdetector },
    'normal_bae': { 'class': NormalBaeDetector },
    'lineart_coarse': { 'class': LineartDetector },
    'lineart_realistic': { 'class': LineartDetector },
    'lineart_anime': { 'class': LineartAnimeDetector },
    'lineart_anime_denoise': { 'class': LineartMangaDetector },
    'depth_midas': { 'class': MidasDetector },
    'depth_zoe': { 'class': ZoeDetector }, 
    'depth_leres': { 'class': LeresDetector }, 
    'depth_leres++': { 'class': LeresDetector },
    'mlsd': {'class': MLSDdetector },
    'dwpose': { 'class': DwposeDetector },
    'dwpose_face': { 'class': DwposeDetector },
    'densepose': { 'class': DenseposeDetector },
    'animal_pose': { 'class': AnimalposeDetector },
    'shuffle': { 'class': ContentShuffleDetector },
    'ip_adapter': { 'class': IPAdapterDetector },
    'ip_adapter_face': { 'class': IPAdapterDetector },
    'tile': { 'class': TileDetector },
    'oneformer': { 'class': OneformerSegmentor },
    'segment_anything': { 'class': SamDetector },
    'lama_inpaint': { 'class': LamaInpaintdetector },
    'anime_segmentation': { 'class': AnimeFaceSegmentor },
    'remove_bg': { 'class': RemoveBackgroundDetector },
}

MODEL_PARAMS = {
    'default': {},
    'invert': {},
    'binary': {},
    'color': {},
    'canny': { 'low_threshold': 64, 'high_threshold': 128 },
    'scribble_hed': { 'scribble': True, 'safe': False },
    'softedge_hed': { 'scribble': False, 'safe': False },
    'scribble_hedsafe': { 'scribble': True, 'safe': True },
    'softedge_hedsafe': { 'scribble': False, 'safe': True },
    'normal_bae': {},
    'lineart_realistic': { 'coarse': False },
    'lineart_coarse': { 'coarse': True },
    'lineart_anime': {},
    'lineart_anime_denoise': {},
    'depth_midas': {},
    'depth_zoe': {},
    'depth_leres': { 'boost': False },
    'depth_leres++': { 'boost': True },
    'mlsd': {},
    'dwpose': { 'include_body': True, 'include_hand': True, 'include_face': True },
    'dwpose_face': { 'include_body': False, 'include_hand': False, 'include_face': True },
    'densepose': {},
    'animal_pose': {},
    'shuffle': { 'f': 512 },
    'ip_adapter': {},
    'ip_adapter_face': {},
    'tile': {},
    'oneformer': {},
    'segment_anything': {},
    'lama_inpaint': {},
    'anime_segmentation': {},
    'remove_bg': {},
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
        self.params = MODEL_PARAMS.get(self.processor_id) or {}
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
    def __call__(self, image: Union[Image.Image, bytes], mask: Union[Image.Image, bytes]=None) -> Image.Image:
        """processes an image with a controlnet aux processor

        Args:
            image (Union[Image.Image, bytes]): input image in bytes or PIL Image

        Returns:
            Image.Image: processed image
        """
        # check if bytes or PIL Image
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")

        processed_image = self.processor(image, **self.params) if mask is None else self.processor(image, mask, **self.params)

        return processed_image
    
    def load_controlnet(self, filename: str):
        filename = os.path.join(modules.paths.controlnet_models_path, filename)
        controlnet_model = controlnet.load_controlnet(filename)
        self.controlnet = controlnet_model
        return self.controlnet
