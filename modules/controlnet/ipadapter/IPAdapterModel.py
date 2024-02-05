import torch
from .network import ImageProjModel, To_KV
from modules.model import model_patcher

class IPAdapterModel(torch.nn.Module):
    def __init__(self, state_dict, model_name, load_device=None, offload_device=None):
        super().__init__()

        self.model_name = model_name
        
        self.is_faceid = False
        self.is_faceid_v2 = False
        self.is_instantid = False

        cross_attention_dim = state_dict["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        clip_extra_context_tokens = 4

        is_full = "proj.3.weight" in state_dict['image_proj']
        is_plus = (
            is_full or
            "latents" in state_dict["image_proj"] or
            "perceiver_resampler.proj_in.weight" in state_dict["image_proj"]
        )

        self.is_full = is_full
        self.is_plus = is_plus
        self.sdxl = cross_attention_dim == 2048

        image_proj_model = self.init_ImageProjModel(
            state_dict,
            cross_attention_dim=cross_attention_dim,
            clip_extra_context_tokens=clip_extra_context_tokens
        )
        ip_layers = self.init_IPLayers(state_dict)

        self.image_proj_model = model_patcher.ModelPatcher(model=image_proj_model, load_device=load_device, offload_device=offload_device)  
        self.ip_layers = model_patcher.ModelPatcher(model=ip_layers, load_device=load_device, offload_device=offload_device)
    
    def init_ImageProjModel(self, state_dict, cross_attention_dim, clip_extra_context_tokens):
        clip_embeddings_dim = int(state_dict['image_proj']['proj.weight'].shape[1])
    
        image_proj_model = ImageProjModel(
            cross_attention_dim=cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
            clip_extra_context_tokens=clip_extra_context_tokens
        )

        image_proj_model.load_state_dict(state_dict["image_proj"])

        return image_proj_model

    def init_IPLayers(self, state_dict):
        ip_layers = To_KV(state_dict["ip_adapter"])

        return ip_layers
    
    def get_image_emb(self, image, clip_image, clip_vision):
        clip_image_embeds = clip_vision.model(pixel_values=clip_image)[2]
        uncond_clip_image_embeds = torch.zeros_like(clip_image_embeds)

        return clip_image_embeds, uncond_clip_image_embeds