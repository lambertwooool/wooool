import os
import torch
import contextlib
import modules.paths
import modules.model.clip_vision
from modules import devices, util
from modules.model import model_loader, model_helper, ops
from modules.model.clip_vision import clip_preprocess
from .IPAdapterModel import IPAdapterModel
from .IPAdapterPlusModel import IPAdapterPlusModel
from .IPAdapterFaceidModel import IPAdapterFaceidModel
from .IPAdapterInstantidModel import IPAdapterInstantidModel
from .attention import patch_unet_model

def tensorToNP(image):
    out = torch.clamp(255. * image.detach().cpu(), 0, 255).to(torch.uint8)
    out = out[..., [2, 1, 0]]
    out = out.numpy()

    return out

def NPToTensor(image):
    out = torch.from_numpy(image)
    out = torch.clamp(out.to(torch.float)/255., 0.0, 1.0)
    out = out[..., [2, 1, 0]]

    return out

class IPAdapterDetector:
    def __init__(self) -> None:
        self.cond = None
        self.uncond = None

    def load_state_dict(self, ip_adapter_path, dtype):
        ip_state_dict = model_helper.load_torch_file(ip_adapter_path, dtype=dtype)
        tmp_state_dict = { "image_proj": {}, "ip_adapter": {} }

        for key in ip_state_dict.keys():
            if key.startswith("image_proj."):
                tmp_state_dict["image_proj"][key.replace("image_proj.", "")] = ip_state_dict.get_tensor(key)
            elif key.startswith("ip_adapter."):
                tmp_state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = ip_state_dict.get_tensor(key)
            else:
                tmp_state_dict[key] = ip_state_dict.get(key)
        ip_state_dict = tmp_state_dict

        return ip_state_dict

    def load(self, cn_type, model_name, want_use_dtype=None):
        ip_adapter_path = os.path.join(modules.paths.controlnet_models_path, model_name)

        load_device, offload_device, dtype, manual_cast_dtype = model_loader.get_device_and_dtype("ipadapter", want_use_dtype=want_use_dtype)
        self.load_device = load_device
        self.offload_device = offload_device
        self.dtype = dtype
        self.manual_cast_dtype = manual_cast_dtype
        self.ops = ops.disable_weight_init if manual_cast_dtype is None else ops.manual_cast

        ip_state_dict = self.load_state_dict(ip_adapter_path, self.dtype)

        is_plus = (
            "proj.3.weight" in ip_state_dict['image_proj'] or
            "latents" in ip_state_dict["image_proj"] or
            "perceiver_resampler.proj_in.weight" in ip_state_dict["image_proj"]
        )
        # plus = "latents" in ip_state_dict["image_proj"]

        if "instant" in model_name:
            model = IPAdapterInstantidModel
        elif "faceid" in model_name: # "0.to_q_lora.down.weight" in ip_state_dict["ip_adapter"]
            model = IPAdapterFaceidModel
        elif is_plus:
            model = IPAdapterPlusModel
        else:
            model = IPAdapterModel
        
        self.ip_adapter = model(ip_state_dict, model_name, dtype, self.load_device, self.offload_device, self.ops)
        
    
    @torch.no_grad()
    @torch.inference_mode()
    def preprocess(self, img, clip_vision):
        image_pixel = img.copy()
        clip_image = clip_preprocess(NPToTensor(image_pixel).unsqueeze(0)).float().to(clip_vision.load_device)
        cond_params = {}
        uncond_params = {}
        
        device, dtype = self.load_device, self.manual_cast_dtype or self.dtype

        if clip_vision.dtype != torch.float32:
            precision_scope = torch.autocast
        else:
            precision_scope = lambda a, b: contextlib.nullcontext(a)

        with precision_scope(devices.get_autocast_device(clip_vision.load_device), torch.float32):
            model_loader.load_model_gpu(clip_vision.patcher)
            embeds = self.ip_adapter.get_image_emb(image_pixel, clip_image, clip_vision)

            if len(embeds) == 2:
                clip_image_embeds, uncond_clip_image_embeds = embeds
            else:
                clip_image_embeds, uncond_clip_image_embeds, cond_params, uncond_params = embeds

        clip_image_embeds = clip_image_embeds.to(device=device, dtype=dtype)
        uncond_clip_image_embeds = uncond_clip_image_embeds.to(device=device, dtype=dtype)
        cond_params = { k: v.to(device=device, dtype=dtype) if hasattr(v, "to") else v for k, v in cond_params.items() }
        uncond_params = { k: v.to(device=device, dtype=dtype) if hasattr(v, "to") else v for k, v in uncond_params.items() }
        # with precision_scope(devices.get_autocast_device(self.load_device), self.dtype):
        model_loader.load_model_gpu(self.ip_adapter.image_proj_model)
        cond = self.ip_adapter.image_proj_model.model(clip_image_embeds, **cond_params)
        uncond = self.ip_adapter.image_proj_model.model(uncond_clip_image_embeds, **uncond_params)
        self.cond = cond
        self.uncond = uncond

        model_loader.load_model_gpu(self.ip_adapter.ip_layers)

        image_emb = [m(cond).cpu() for m in self.ip_adapter.ip_layers.model.to_kvs]
        uncond_image_emb = [m(uncond).cpu() for m in self.ip_adapter.ip_layers.model.to_kvs]

        return image_emb, uncond_image_emb

    @torch.no_grad()
    @torch.inference_mode()
    def patch_model(self, model, image_emb, uncond_image_emb, weight, attn_mask=None, start_at=0.0, end_at=1.0):
        sigma_start = model.model.model_sampling.percent_to_sigma(start_at)
        sigma_end = model.model.model_sampling.percent_to_sigma(end_at)

        # '''
        # patch_name of sdv1-2: ("input" or "output" or "middle", block_id)
        # patch_name of sdxl: ("input" or "output" or "middle", block_id, transformer_index)
        # '''
        patch_kwargs = {
            "number": 0,
            "weight": weight,
            "dtype": self.dtype,
            "cond": image_emb,
            "uncond": uncond_image_emb,
            "attn_mask": attn_mask,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
            "sdxl": self.ip_adapter.sdxl,
        }

        new_model = patch_unet_model(model, patch_kwargs)

        return new_model

    def apply_conds(self, positive_cond, negative_cond):
        cond, uncond = self.cond, self.uncond
        positive_cond, negative_cond = self.ip_adapter.apply_conds(positive_cond, negative_cond, cond, uncond)
        return positive_cond, negative_cond
