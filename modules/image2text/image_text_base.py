import hashlib
import numpy as np
import torch
import torch.nn as nn
import os
import tqdm
import modules.paths
from transformers import CodeGenTokenizerFast as Tokenizer
from modules.model import model_patcher, model_loader

class Image2TextBase():
    def __init__(self,
                 model_name: str,
                 dtype: torch.dtype = None):
        self.name = model_name
        load_device, offload_device, dtype, manual_cast_dtype = model_loader.get_device_and_dtype("image2text", want_use_dtype=dtype)

        self.device = self.load_device = load_device
        self.offload_device = offload_device
        self.dtype = dtype
        self.manual_cast_dtype = manual_cast_dtype
        
        self.model_path, self.tokenizer, self.vision_encoder_model, self.text_model = self.load_model()
        self.vision_encoder_model.to(self.dtype).eval()
        self.text_model.to(self.dtype).eval()
        self.vision_encoder_wrap = model_patcher.ModelPatcher(self.vision_encoder_model, load_device=self.load_device, offload_device=self.offload_device)
        self.text_model_wrap = model_patcher.ModelPatcher(self.text_model, load_device=self.load_device, offload_device=self.offload_device)
    
    def load_model(self) -> tuple[str, Tokenizer, nn.Module, nn.Module]:
        raise NotImplementedError()
    
    def vision_encoder(self, image: np.ndarray) -> torch.Tensor:
        raise NotImplementedError()
    
    def image_embeds_to_text(self,
                             image_embeds: torch.Tensor,
                             question: str,
                             generate_config: dict = {} ) -> str:
        raise NotImplementedError()
    
    def answer_question(self,
                        image,
                        question: str,
                        max_new_tokens: int = 256,
                        use_vision_cache=False):
        
        generate_config = {
            "do_sample": False,
            "use_cache": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "max_new_tokens": max_new_tokens,
        }
        
        with torch.inference_mode():
            image_embeds = self._vision_encoder(image, use_vision_cache=use_vision_cache)
            counter_wrap = GenerateWrapper(self.text_model, max_new_tokens)
            
            try:                
                model_loader.load_model_gpu(self.text_model_wrap)
                output = self.image_embeds_to_text(image_embeds, question, generate_config)
            finally:
                counter_wrap.reset()

        model_loader.unload_all_models()
        return output
    
    def _vision_encoder(self, image, use_vision_cache=False) -> torch.Tensor:
        # Calculate checksum of the image
        np_image = image.numpy() if isinstance(image, torch.Tensor) else image
        image_hash = hashlib.sha256(str(self.name).encode() + np_image.tobytes()).hexdigest()[:10]

        # Check if `image_encoder_cache/{image_hash}.pt` exists, if so load and return it.
        # Otherwise, save the encoded image to `image_encoder_cache/{image_hash}.pt` and return it.
        cache_root = os.path.join(modules.paths.caches_path, "image_encoder_cache")
        cache_path = os.path.join(cache_root, f"{image_hash}.pt")
        if use_vision_cache and os.path.exists(cache_path):
            return torch.load(cache_path).to(device=self.device, dtype=self.dtype)
        else:
            model_loader.load_model_gpu(self.vision_encoder_wrap)
            image_vec = self.vision_encoder(image)
            if use_vision_cache:
                os.makedirs(cache_root, exist_ok=True)
                torch.save(image_vec, cache_path)
            return image_vec.to(device=self.device, dtype=self.dtype)

class GenerateWrapper(nn.Module):
    def __init__(self, model: nn.Module, max_new_tokens: int):
        super().__init__()
        self.text_counter = tqdm.tqdm(total=max_new_tokens, desc="Generating text")
        self.model = model
        self.original_forward = getattr(model, "forward")
        setattr(model, "forward", self.forward)
        
    # The forward signature must include the attention_mask parameter.
    def forward(self, *args, attention_mask=None, **kwargs):
        self.text_counter.update()
        kwargs["attention_mask"] = attention_mask
        return self.original_forward(*args, **kwargs)
    
    def reset(self):
        setattr(self.model, "forward", self.original_forward)