import cv2
import io
import torch
from PIL import Image
from .image_text_base import Image2TextBase
from .moondream import MoondreamModelV1, MoondreamModelV2
from .qwen import QwenModel
from typing import Union

__all__ = [
    'MoondreamModelV1',
    'MoondreamModelV2',
    'QwenModel',
]

MODELS = {
    'moondream_v1': { 'class': MoondreamModelV1 },
    'moondream_v2': { 'class': MoondreamModelV2 },
    'qwen': { 'class': QwenModel },
}

class Image2TextProcessor:
    def __init__(self, processor_id: str) -> None:
        """Processor that can be used to process images with image to text processors

        Args:
            processor_id (str)
        """
        if processor_id not in MODELS:
            raise ValueError(f"{processor_id} is not a valid processor id. Please make sure to choose one of {', '.join(MODELS.keys())}")

        self.processor_id = processor_id
        self.processor: Image2TextBase = None

    def model_keys():
        return MODELS.keys()

    def load_processor(self):
        """Load processor

        """

        self.processor = MODELS[self.processor_id]['class']()
    
    @torch.inference_mode()
    def __call__(self, image: Union[Image.Image, bytes], question: str=None, max_new_tokens=256) -> str:
        """processes an image with a image to text processor

        Args:
            image (Union[Image.Image, bytes]): input image in bytes or PIL Image
            mask (Union[Image.Image, bytes]): mask in bytes or PIL Image

        Returns:
            Image.Image: processed image
        """
        # check if bytes or PIL Image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.processor is None:
            self.load_processor()
        
        question = [    "Describe this photograph.",
                        "What is this?",
                        "Please describe this image in detail."
                    ][0]
        
        answer = self.processor.answer_question(image, question, max_new_tokens=max_new_tokens, use_vision_cache=True)
        
        return answer