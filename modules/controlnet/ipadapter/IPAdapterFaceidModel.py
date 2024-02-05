import torch
from modules import insightface_model
from modules.model.clip_vision import clip_preprocess
from .IPAdapterModel import IPAdapterModel
from .network import ProjModelFaceIdPlus, MLPProjModelFaceId

class IPAdapterFaceidModel(IPAdapterModel):
    def __init__(self, state_dict, model_name, load_device=None, offload_device=None):
        super().__init__(state_dict, model_name, load_device, offload_device)

        self.is_faceid = True
        self.is_faceid_v2 = "v2" in model_name

    def init_ImageProjModel(self, state_dict, cross_attention_dim, clip_extra_context_tokens):
        if self.is_plus:
            clip_embeddings_dim = 1280
            image_proj_model = ProjModelFaceIdPlus(
                cross_attention_dim=cross_attention_dim,
                id_embeddings_dim=512,
                clip_embeddings_dim=clip_embeddings_dim,
                num_tokens=4,
            )
        else:
            image_proj_model = MLPProjModelFaceId(
                cross_attention_dim=cross_attention_dim,
                id_embeddings_dim=512,
                num_tokens=16,
            )

        image_proj_model.load_state_dict(state_dict["image_proj"])

        return image_proj_model
    
    def get_image_emb(self, image, clip_image, clip_vision):
        analysis = insightface_model.Analysis()
        faces = analysis(image)
        face_embeds = faces[0].normed_embedding
        face_embeds = torch.from_numpy(face_embeds).unsqueeze(0).to(clip_image)
    
        if self.is_plus:
            clip_image_embeds = clip_vision.model(pixel_values=clip_image, intermediate_output=-2)[1]
            zero_clip_image = clip_preprocess(torch.zeros([1, 224, 224, 3])).float().to(clip_vision.load_device)
            uncond_clip_image_embeds = clip_vision.model(pixel_values=zero_clip_image, intermediate_output=-2)[1]

            cond_params = { "face_embeds": face_embeds, "shortcut": self.is_faceid_v2, "scale": 1.0 }
            uncond_params = { "face_embeds": torch.zeros_like(face_embeds), "shortcut": self.is_faceid_v2, "scale": 1.0 }

            return clip_image_embeds, uncond_clip_image_embeds, cond_params, uncond_params
        else:
            clip_image_embeds = face_embeds
            uncond_clip_image_embeds = torch.zeros_like(face_embeds)

            return clip_image_embeds, uncond_clip_image_embeds