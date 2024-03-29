import modules.paths
from modules import model_helper, diffusers_convert

def apply(self, model, unet_name, strength=1.0):
    # sd = load_state_dict(folder_paths.get_full_path("unet", unet_name))
    model_path = os.path.join(modules.paths.unet_path, unet_name)
    unet_state_dict = model_helper.load_torch_file(model_path)
    sd = diffusers_convert.convert_unet_state_dict(unet_state_dict)

    model.add_patches(
        { f"diffusion_model.{k}": (v,) for k, v in sd.items() },
        strength_patch=strength,
        strength_model=min(1.0, max(0.0, 1.0 - strength)),
    )
    return model