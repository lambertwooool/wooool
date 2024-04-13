import os
import torch
from huggingface_hub import snapshot_download
from ..image_text_base import Image2TextBase
from .vision_encoder import VisionEncoder
from .text_model import TextModel
from .modeling_phi import PhiForCausalLM
from .configuration_moondream import PhiConfig
from transformers import CodeGenTokenizerFast as Tokenizer

import modules.paths
from modules.model import model_helper, ops

class MoondreamModelV1(Image2TextBase):
    def __init__(self, name="moondream1", dtype=None):
        super().__init__(name, dtype=dtype)
        
    def load_model(self):
        model_path = os.path.join(modules.paths.image2text_path, self.name)
        state_dict_path = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(state_dict_path):
            snapshot_download(f"vikhyatk/{self.name}", local_dir=model_path, ignore_patterns=["*.jpg", "*.pt", "*.bin", "*0000*", "*.py"], local_dir_use_symlinks=False)
        
        model_path = {
            "moondream1": {
                "state_dict": state_dict_path,
                "tokenizer": f"{model_path}/tokenizer",
                "phi_config": f"{model_path}/text_model_cfg.json",
            },
            "moondream2": {
                "state_dict": state_dict_path,
                "tokenizer": f"{model_path}",
                "phi_config": None,
            }
        }.get(self.name)

        state_dict = model_helper.load_torch_file(model_path.get("state_dict"))
        vison_encoder_sd = {}
        text_model_sd = {}
        for k in [k for k in state_dict]:
            if k.startswith("vision_encoder."):
                v = state_dict.pop(k)
                vison_encoder_sd[k[len("vision_encoder."):]] = v
            elif k.startswith("text_model."):
                v = state_dict.pop(k)
                text_model_sd[k[len("text_model."):]] = v
        
        phi_config = PhiConfig.from_pretrained(model_path.get("phi_config")) if model_path.get("phi_config") is not None else PhiConfig()
        with ops.auto_ops():
            vision_encoder = VisionEncoder()
            phi_model = PhiForCausalLM(phi_config)
            
        vision_encoder.load_state_dict(vison_encoder_sd)
        vision_encoder.to(dtype=self.dtype).eval()
        
        tokenizer: Tokenizer = Tokenizer.from_pretrained(model_path.get("tokenizer"))
        
        phi_model.load_state_dict(text_model_sd)
        text_model = TextModel(tokenizer, phi_model).to(dtype=self.dtype)
        
        self.processor = text_model
        
        return model_path, tokenizer, vision_encoder, phi_model
        
    def vision_encoder(self, image):
        return self.vision_encoder_wrap.model(image)
    
    def image_embeds_to_text(self,
                             image_embeds: torch.Tensor,
                             question: str,
                             generate_config: dict = {} ):
        chat_history = ""
        prompt = f"<image>\n\n{chat_history}Question: {question}\n\nAnswer:"
        
        text_model: TextModel = self.text_model_wrap.model
        inputs_embeds = self.processor.input_embeds(prompt, image_embeds, self.tokenizer)
        output_ids = text_model.generate(
            inputs_embeds=inputs_embeds, **generate_config
        )
        
        answer = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        if self.name == "moondream1":
            answer = answer.split("<END>")[0]
        cleaned_answer = answer.strip()
        
        return cleaned_answer

class MoondreamModelV2(MoondreamModelV1):
    def __init__(self, name="moondream2", dtype=None):
        super().__init__(name, dtype=dtype)