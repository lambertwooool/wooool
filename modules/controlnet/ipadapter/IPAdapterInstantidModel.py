import torch
from modules import insightface_model
from .IPAdapterModel import IPAdapterModel
from .network import Resampler

class IPAdapterInstantidModel(IPAdapterModel):
    def __init__(self, state_dict, model_name, load_device=None, offload_device=None):
        super().__init__(state_dict, model_name, load_device, offload_device)

        self.is_instantid = True

    def init_ImageProjModel(self, state_dict, cross_attention_dim, clip_extra_context_tokens):
        clip_embeddings_dim = 512

        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=clip_extra_context_tokens,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4
        )

        image_proj_model.load_state_dict(state_dict["image_proj"])

        return image_proj_model
    
    def get_image_emb(self, image, clip_image, clip_vision):
        analysis = insightface_model.Analysis(name="antelopev2")
        faces = analysis(image)
        face_embeds = faces[0].embedding
        # face_kps = util.draw_kps(img, faces[0].kps)

        """Get image embeds for instantid."""
        image_proj_model_in_features = 512
        if isinstance(face_embeds, torch.Tensor):
            face_embeds = face_embeds.clone().detach()
        else:
            face_embeds = torch.tensor(face_embeds)

        face_embeds = face_embeds.reshape([1, -1, image_proj_model_in_features])
        clip_image_embeds = face_embeds.to(clip_image_embeds)
        uncond_clip_image_embeds = torch.zeros_like(clip_image_embeds)

        return clip_image_embeds, uncond_clip_image_embeds