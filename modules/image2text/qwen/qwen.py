import os
import re
import torch
from numpy import ndarray
from huggingface_hub import snapshot_download
from ..image_text_base import Image2TextLLM
from .vision_encoder import VisionEncoder
from .configuration_uform_gen import VLMConfig
from .modeling_uform_gen import VLMForCausalLM, ImageFeaturesPooler
from .processing_uform_gen import VLMProcessor
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers import Qwen2TokenizerFast as Tokenizer

import modules.paths
from modules.model import model_patcher, model_loader, model_helper, ops


class QwenModel(Image2TextLLM):
    def __init__(self, name="uform-gen2-qwen-500m", dtype=None):
        super().__init__(name, dtype=dtype)
        
    def load_model(self):
        model_path = os.path.join(modules.paths.image2text_path, self.name)
        state_dict_path = [os.path.join(model_path, x) for x in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]]
        if not all([os.path.exists(x) for x in state_dict_path]):
            snapshot_download(f"unum-cloud/uform-gen2-qwen-500m", local_dir=model_path, ignore_patterns=["*.jpg", "*.pt", "*.bin", "*0000*", "*.py"], local_dir_use_symlinks=False)    
        tokenizer_path = os.path.join(model_path, "tokenizer")
        if not os.path.exists(tokenizer_path):
            snapshot_download("Qwen/Qwen1.5-0.5B-Chat", local_dir=tokenizer_path, allow_patterns=["*.json", "*.txt"], local_dir_use_symlinks=False)
        
        image_encoder_sd = {}
        image_pooler_sd = {}
        text_decoder_sd = {}
        
        for sd_path in state_dict_path:
            state_dict = model_helper.load_torch_file(sd_path)
            for k in [k for k in state_dict]:
                v = state_dict.pop(k)
                if k.startswith("image_encoder."):
                    image_encoder_sd[k[len("image_encoder."):]] = v
                elif k.startswith("text_decoder."):
                    text_decoder_sd[k[len("text_decoder."):]] = v
                elif k.startswith("image_pooler."):
                    image_pooler_sd[k[len("image_pooler."):]] = v

        config_path = os.path.dirname(os.path.realpath(__file__))
        vlm_config = VLMConfig.from_pretrained(config_path)
        text_config = Qwen2Config.from_pretrained(tokenizer_path)
        
        with ops.auto_ops():
            vision_encoder = VisionEncoder(
                vlm_config.image_encoder_hidden_size,
                vlm_config.image_encoder_patch_size,
                vlm_config.image_encoder_num_layers,
                vlm_config.image_encoder_num_heads,
            )
            text_decoder = Qwen2ForCausalLM(text_config)

        vision_encoder.load_state_dict(image_encoder_sd)
        
        tokenizer = Tokenizer.from_pretrained(tokenizer_path, additional_special_tokens=["<image>"])
        
        text_decoder.load_state_dict(text_decoder_sd)
        qwen_model = VLMForCausalLM(text_decoder, vlm_config, text_config)
        
        image_pooler = ImageFeaturesPooler(vlm_config, text_config)
        image_pooler.load_state_dict(image_pooler_sd)
        image_pooler.to(dtype=self.dtype).eval()
        image_pooler_wrap = model_patcher.ModelPatcher(image_pooler, load_device=self.load_device, offload_device=self.offload_device)
        
        processor = VLMProcessor(tokenizer, vlm_config)
        
        self.processor, self.image_pooler_wrap = processor, image_pooler_wrap
        return model_path, tokenizer, vision_encoder, qwen_model
    
    def vision_encoder(self, image: ndarray) -> torch.Tensor:
        image_features = self.processor.feature_extractor(image).unsqueeze(0)

        device_and_dtype = next(self.vision_encoder_wrap.model.parameters())
        image_features = self.vision_encoder_wrap.model(image_features.to(device_and_dtype))
        
        model_loader.load_model_gpu(self.image_pooler_wrap)
        image_features = self.image_pooler_wrap.model(image_features)
        
        return image_features
        
    def image_embeds_to_text(self,
                             image_embeds: torch.Tensor,
                             question: str = "",
                             generate_config: dict = {},
                             **kwargs) -> str:
        input_embeds = self.processor(text=[question], return_tensors="pt")
        input_embeds["images"] = image_embeds
        input_embeds = { name: tensor.to(device=self.device) for name, tensor in input_embeds.items() }
        
        text_model: VLMForCausalLM = self.text_model
        output = text_model.generate(
            **input_embeds,
            **generate_config,
        )
        
        prompt_len = input_embeds["input_ids"].shape[1]
        decoded_text = self.processor.batch_decode(output[:, prompt_len:])[0]
        decoded_text = decoded_text.replace("<|im_end|>","")
        re_duplicate = r"(.+?)\1+$"
        match_texts = []
        while True:
            match = re.search(re_duplicate, decoded_text)
            if match is not None:
                match_texts.insert(0, match.group(1))
                decoded_text = re.sub(re_duplicate, "", decoded_text)
            else:
                # decoded_text += "".join(match_texts)
                break
        
        return decoded_text