import torch
from modules.model.clip_vision import clip_preprocess
from .IPAdapterModel import IPAdapterModel
from .network import MLPProjModel, Resampler

class IPAdapterPlusModel(IPAdapterModel):
    def init_ImageProjModel(self, state_dict, cross_attention_dim, clip_extra_context_tokens):
        clip_extra_context_tokens = 16
    
        if self.is_full:
            clip_embeddings_dim = int(state_dict["image_proj"]["proj.0.weight"].shape[1])
            image_proj_model = MLPProjModel(
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=clip_embeddings_dim
            )
        else:
            if self.sdxl:
                clip_embeddings_dim = int(state_dict["image_proj"]["latents"].shape[2])
            else:
                clip_embeddings_dim = int(state_dict['image_proj']['proj_in.weight'].shape[1])
        
            image_proj_model = Resampler(
                dim=1280 if self.sdxl else cross_attention_dim,
                depth=4,
                dim_head=64,
                heads=20 if self.sdxl else 12,
                num_queries=clip_extra_context_tokens,
                embedding_dim=clip_embeddings_dim,
                output_dim=cross_attention_dim,
                ff_mult=4,
                ops=self.ops
            )

        image_proj_model.load_state_dict(state_dict["image_proj"])

        return image_proj_model
    
    def get_image_emb(self, image, clip_image, clip_vision):
        clip_image_embeds = clip_vision.model(pixel_values=clip_image, intermediate_output=-2)[1]
        zero_clip_image = clip_preprocess(torch.zeros([1, 224, 224, 3])).float().to(clip_vision.load_device)
        uncond_clip_image_embeds = clip_vision.model(pixel_values=zero_clip_image, intermediate_output=-2)[1]

        return clip_image_embeds, uncond_clip_image_embeds