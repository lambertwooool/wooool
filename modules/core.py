from modules.patches import patch_all

import torch
from modules.model import sd, model_helper
from modules.paths import embeddings_path

patches = patch_all()

class StableDiffusionModel:
    def __init__(self, unet, vae, clip, clip_vision):
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision

@torch.no_grad()
@torch.inference_mode()
def load_model(ckpt_filename, output_clip=True, output_vae=True):
    unet, clip, vae, clip_vision = sd.load_checkpoint_guess_config(ckpt_filename, embedding_directory=embeddings_path, output_clip=output_clip, output_vae=output_vae)
    return StableDiffusionModel(unet=unet, clip=clip, vae=vae, clip_vision=clip_vision)

@torch.no_grad()
@torch.inference_mode()
def load_sd_lora(model, lora_filename, strength_model=1.0, strength_clip=1.0):
    if strength_model == 0 and strength_clip == 0:
        return model

    lora_model = model_helper.load_torch_file(lora_filename, safe_load=False)
    new_unet, new_clip = sd.load_lora_for_models(model.unet, model.clip, lora_model, strength_model, strength_clip)

    return StableDiffusionModel(unet=new_unet, clip=new_clip, vae=model.vae, clip_vision=model.clip_vision)