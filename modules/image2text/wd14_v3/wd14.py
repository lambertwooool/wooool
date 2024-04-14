import os
import re
import torch
import numpy as np
from huggingface_hub import snapshot_download
from ..image_text_base import Image2TextBase
from transformers.models.swinv2.modeling_swinv2 import Swinv2Config, Swinv2ForImageClassification
from .image_processing_tagger import WDTaggerImageProcessor

import modules.paths
from modules.model import model_helper, ops

class Wd14V3Model(Image2TextBase):
    def __init__(self, name="wd-swinv2-tagger-v3-hf", dtype=None):
        super().__init__(name, dtype=dtype)
        
    def load_model(self):
        model_path = os.path.join(modules.paths.image2text_path, self.name)
        state_dict_path = [os.path.join(model_path, x) for x in ["model.safetensors"]]
        if not all([os.path.exists(x) for x in state_dict_path]):
            snapshot_download(f"p1atdev/wd-swinv2-tagger-v3-hf", local_dir=model_path, allow_patterns=["*.safetensors"], local_dir_use_symlinks=False, resume_download=True)
        
        state_dict = model_helper.load_torch_file(state_dict_path[0])
        
        config_path = os.path.dirname(os.path.realpath(__file__))
        config = Swinv2Config.from_pretrained(config_path)
        
        with ops.auto_ops():
            text_model = Swinv2ForImageClassification(config)
        text_model.load_state_dict(state_dict)
        text_model.to(dtype=self.dtype).eval()
        
        processor = WDTaggerImageProcessor()
        
        return model_path, None, processor, text_model
    
    def vision_encoder(self, image: np.ndarray) -> torch.Tensor:
        inputs = self.vision_encoder_model.preprocess(image, return_tensors="pt")
        return inputs
    
    def image_embeds_to_text(self, image_embeds: torch.Tensor, threshold=0.3, threshold_character=0.8, exclude_tags=[], only_text=True, **kwargs) -> str:
        outputs = self.text_model(**image_embeds.to(self.device, self.dtype))
        logits = torch.sigmoid(outputs.logits[0]).cpu()
        
        # threshold = generate_config.get("threshold", 0.3)
        # threshold_character = generate_config.get("threshold_character", 0.8)
        # exclude_tags = generate_config.get("exclude_tags", [])
        # only_text = generate_config.get("only_text", True)
        
        ratings, characters, tags = {}, {}, {}
        rating_prefix = "rating:"
        character_prefix = "character:"
        for i, logit in enumerate(logits):
            tag = self.text_model.config.id2label[i]
            logit = logit.item()
            
            if tag.startswith(rating_prefix):
                ratings[tag[len(rating_prefix):]] = logit
            elif tag.startswith(character_prefix):
                if logit > threshold_character:
                    characters[tag[len(character_prefix):]] = logit
            else:
                if logit > threshold:
                    tags[tag] = logit
            
        results = sorted((characters | tags).items(), key=lambda item: item[1], reverse=True)
        results = [(re.sub(r"_", " ", tag[0]).capitalize(), tag[1]) for tag in results \
                    if tag[0].lower().strip() not in exclude_tags]
        
        output = ",".join([tag[0] for tag in results]) if only_text else results
        
        return output