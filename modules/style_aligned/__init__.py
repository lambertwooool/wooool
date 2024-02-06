import torch
import torch.nn as nn
from modules.model import model_patcher
from typing import Union
from .utils import adain, concat_first, expand_first, get_norm_layers

T = torch.Tensor

class StyleAlignedParams:
    def __init__(self, adain_queries=True, adain_keys=True, adain_values=True):
        self.adain_queries = adain_queries
        self.adain_keys = adain_keys
        self.adain_values = adain_values

    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = True

class SharedAttentionProcessor:
    def __init__(self, model, model_dtype, params: StyleAlignedParams, scale:float=1.0, start_at:float=0.0, end_at:float=1.0):
        self.params = params
        self.scale = scale

        self.sigma_start = model.model.model_sampling.percent_to_sigma(start_at)
        self.sigma_end = model.model.model_sampling.percent_to_sigma(end_at)
        if model_dtype == "sdxl":
            self.patch_blocks = [("input", id) for id in [1,2,4,5,7,8]] + \
                                [("output", id) for id in [3,4,5,6,7,8,9,10,11]] + \
                                [("middle", 0)]
        else:
            self.patch_blocks = None

    def __call__(self, q, k, v, extra_options):
        sigma = extra_options["sigmas"][0].item() if 'sigmas' in extra_options else 999999999.9
        block = extra_options["block"]
        if sigma > self.sigma_start or sigma <self.sigma_end or \
                (self.patch_blocks is not None and block not in self.patch_blocks):
            return q, k, v
        
        if self.params.adain_queries:
            q = adain(q)
        if self.params.adain_keys:
            k = adain(k)
        if self.params.adain_values:
            v = adain(v)
        if self.params.share_attention:
            k = concat_first(k, -2, scale=self.scale)
            v = concat_first(v, -2)
            # k = expand_first(k, scale=self.scale)
            # v = expand_first(v)

        return q, k, v

class StyleAlignedPatch:
    def register_shared_norm(
        self,
        model,
        share_group_norm: bool = True,
        share_layer_norm: bool = True,
    ):
        def register_norm_forward(
            norm_layer: Union[nn.GroupNorm, nn.LayerNorm],
        ) -> Union[nn.GroupNorm, nn.LayerNorm]:
            if not hasattr(norm_layer, "orig_forward"):
                setattr(norm_layer, "orig_forward", norm_layer.forward)
            orig_forward = norm_layer.orig_forward

            def forward_(hidden_states: T) -> T:
                n = hidden_states.shape[-2]
                hidden_states = concat_first(hidden_states, dim=-2)
                hidden_states = orig_forward(hidden_states)  # type: ignore
                return hidden_states[..., :n, :]

            norm_layer.forward = forward_  # type: ignore
            return norm_layer

        norm_layers = {"group": [], "layer": []}
        get_norm_layers(model, norm_layers, share_layer_norm, share_group_norm)
        print(
            f"Patching {len(norm_layers['group'])} group norms, {len(norm_layers['layer'])} layer norms."
        )
        
        [register_norm_forward(layer) for layer in norm_layers["group"]]
        [register_norm_forward(layer) for layer in norm_layers["layer"]]

    def patch(
        self,
        model: model_patcher.ModelPatcher,
        latent,
        noise,
        params: StyleAlignedParams,
        ref_positive: T,
        ref_latents: T,
        scale: float=1.0,
        share_group_norm=True,
        share_layer_norm=True,
    ):
        latent_t = latent["samples"]

        # Concat batch with style latent
        style_latent_tensor = ref_latents[0].unsqueeze(0)
        latent_t = torch.cat((style_latent_tensor, latent_t), dim=0)

        ref_noise = torch.zeros_like(noise[0]).unsqueeze(0)
        noise = torch.cat((ref_noise, noise), dim=0)

        # Register shared norms
        self.register_shared_norm(model.model, share_group_norm, share_layer_norm)

        # Patch cross attn
        model.set_model_attn1_patch(SharedAttentionProcessor(params, scale))

    def apply(
        self,
        positive: T,
        ref_positive: T,
        batch_size: int,
    ):
        # Add reference conditioning to batch 
        batched_condition = []
        for i, condition in enumerate(positive):
            additional = condition[1].copy()
            batch_with_reference = torch.cat([ref_positive[i][0], condition[0].repeat([batch_size] + [1] * len(condition[0].shape[1:]))], dim=0)
            if 'pooled_output' in additional and 'pooled_output' in ref_positive[i][1]:
                # combine pooled output
                pooled_output = torch.cat([ref_positive[i][1]['pooled_output'], additional['pooled_output'].repeat([batch_size] 
                    + [1] * len(additional['pooled_output'].shape[1:]))], dim=0)
                additional['pooled_output'] = pooled_output
            if 'control' in additional:
                if 'control' in ref_positive[i][1]:
                    # combine control conditioning
                    control_hint = torch.cat([ref_positive[i][1]['control'].cond_hint_original, additional['control'].cond_hint_original.repeat([batch_size] 
                        + [1] * len(additional['control'].cond_hint_original.shape[1:]))], dim=0)
                    cloned_controlnet = additional['control'].copy()
                    cloned_controlnet.set_cond_hint(control_hint, strength=additional['control'].strength, timestep_percent_range=additional['control'].timestep_percent_range)
                    additional['control'] = cloned_controlnet
                else:
                    # add zeros for first in batch
                    control_hint = torch.cat([torch.zeros_like(additional['control'].cond_hint_original), additional['control'].cond_hint_original.repeat([batch_size] 
                        + [1] * len(additional['control'].cond_hint_original.shape[1:]))], dim=0)
                    cloned_controlnet = additional['control'].copy()
                    cloned_controlnet.set_cond_hint(control_hint, strength=additional['control'].strength, timestep_percent_range=additional['control'].timestep_percent_range)
                    additional['control'] = cloned_controlnet
            batched_condition.append([batch_with_reference, additional])

        return batched_condition