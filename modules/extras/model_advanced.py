import torch
from modules.model import latent_formats, model_sampling

def Rescale_cfg(model, cfg_multiplier, cfg_scale_to):
    sigma_max = model.model.model_sampling.sigma_max
    latent_format_alpha = {
        latent_formats.SDXL_Playground_2_5: 3.0,
        latent_formats.SC_Prior: 1.0,
    }
    cond_alpha = 1.0
    for latent_format, alpha in latent_format_alpha.items():
        if isinstance(model.model.latent_format, latent_format):
            cond_alpha = alpha / 7.0
    cfg_scale_to = round(cfg_scale_to * cond_alpha, 1)

    def patch_cfg(args):
        cond = args["cond"]
        uncond = args["uncond"]
        cond_scale = round(args["cond_scale"] * cond_alpha, 1)
        sigma = args["sigma"]   
        sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
        x_orig = args["input"]

        #rescale cfg has to be done on v-pred model output
        x = x_orig / (sigma * sigma + 1.0)
        cond = ((x - (x_orig - cond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)
        uncond = ((x - (x_orig - uncond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)

        #rescalecfg
        cond_scale = cfg_scale_to + (cond_scale - cfg_scale_to) * sigma[0] / sigma_max
        x_cfg = uncond + cond_scale * (cond - uncond)
        ro_pos = torch.std(cond, dim=(1,2,3), keepdim=True)
        ro_cfg = torch.std(x_cfg, dim=(1,2,3), keepdim=True)

        x_rescaled = x_cfg * (ro_pos / ro_cfg)
        x_final = cfg_multiplier * x_rescaled + (1.0 - cfg_multiplier) * x_cfg

        return x_orig - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5)
    
    model.set_model_sampler_cfg_function(patch_cfg)

def Model_sampling(model, sampling, sigma_min=0.002, sigma_max=120.0, sigma_data=1.0):
    latent_format = None
    sigma_data = 1.0
    if sampling == model_sampling.EDM:
        sigma_data = 0.5
        latent_format = latent_formats.SDXL_Playground_2_5()
    
    class ModelSamplingAdvanced(model_sampling.ModelSamplingContinuousEDM, sampling):
        pass
    
    sampling_patch = ModelSamplingAdvanced(model.model.model_config)
    sampling_patch.set_parameters(sigma_min, sigma_max, sigma_data)
    model.add_object_patch("model_sampling", sampling_patch)
    if latent_format is not None:
        model.add_object_patch("latent_format", latent_format)