import torch
from modules.model import sample, samplers

def apply(model, empty_conditioning, neg_scale=1.0):
    nocond = sample.convert_cond(empty_conditioning)

    def cfg_function(args):
        model = args["model"]
        noise_pred_pos = args["cond_denoised"]
        noise_pred_neg = args["uncond_denoised"]
        cond_scale = args["cond_scale"]
        x = args["input"]
        sigma = args["sigma"]
        model_options = args["model_options"]
        nocond_processed = samplers.encode_model_conds(model.extra_conds, nocond, x, x.device, "negative")

        (noise_pred_nocond, _) = samplers.calc_cond_batch(model, [nocond_processed], x, sigma, model_options)

        pos = noise_pred_pos - noise_pred_nocond
        neg = noise_pred_neg - noise_pred_nocond
        perp = neg - ((torch.mul(neg, pos).sum())/(torch.norm(pos)**2)) * pos
        perp_neg = perp * neg_scale
        cfg_result = noise_pred_nocond + cond_scale*(pos - perp_neg)
        cfg_result = x - cfg_result
        return cfg_result

    model.set_model_sampler_cfg_function(cfg_function)
