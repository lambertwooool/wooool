import inspect
import os
from einops import rearrange
import torch
import torch.nn.functional as F
import contextlib
import modules.paths
import modules.model.clip_vision
import modules.model.ldm.modules.attention as attention
from modules import devices, shared
from modules.model import model_loader, model_helper
from modules.model.model_patcher import ModelPatcher
from .resampler import Resampler

SD_V12_CHANNELS = [320] * 4 + [640] * 4 + [1280] * 4 + [1280] * 6 + [640] * 6 + [320] * 6 + [1280] * 2
SD_XL_CHANNELS = [640] * 8 + [1280] * 40 + [1280] * 60 + [640] * 12 + [1280] * 20

def sdp(q, k, v, extra_options):
    return attention.optimized_attention(q, k, v, heads=extra_options["n_heads"], mask=None)


class ImageProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens,
                                                              self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class To_KV(torch.nn.Module):
    def __init__(self, cross_attention_dim):
        super().__init__()

        channels = SD_XL_CHANNELS if cross_attention_dim == 2048 else SD_V12_CHANNELS
        self.to_kvs = torch.nn.ModuleList(
            [torch.nn.Linear(cross_attention_dim, channel, bias=False) for channel in channels])

    def load_state_dict_ordered(self, sd):
        state_dict = []
        for i in range(4096):
            for k in ['k', 'v']:
                key = f'{i}.to_{k}_ip.weight'
                if key in sd:
                    state_dict.append(sd[key])
        for i, v in enumerate(state_dict):
            self.to_kvs[i].weight = torch.nn.Parameter(v, requires_grad=False)


class IPAdapterModel(torch.nn.Module):
    def __init__(self, state_dict, plus, cross_attention_dim=768, clip_embeddings_dim=1024, clip_extra_context_tokens=4,
                 sdxl_plus=False):
        super().__init__()
        self.plus = plus
        if self.plus:
            self.image_proj_model = Resampler(
                dim=1280 if sdxl_plus else cross_attention_dim,
                depth=4,
                dim_head=64,
                heads=20 if sdxl_plus else 12,
                num_queries=clip_extra_context_tokens,
                embedding_dim=clip_embeddings_dim,
                output_dim=cross_attention_dim,
                ff_mult=4
            )
        else:
            self.image_proj_model = ImageProjModel(
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=clip_embeddings_dim,
                clip_extra_context_tokens=clip_extra_context_tokens
            )

        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.ip_layers = To_KV(cross_attention_dim)
        self.ip_layers.load_state_dict_ordered(state_dict["ip_adapter"])

# ip_negative: torch.Tensor = None
ip_unconds = None

class IPAdapterDetector:
    def __init__(self) -> None:
        pass

    def load(self, cn_type, model_name):
        global ip_negative, ip_unconds

        # ip_negative_path = os.path.join(modules.paths.clip_vision_models_path, 'fooocus_ip_negative.safetensors')
        ip_adapter_path = os.path.join(modules.paths.controlnet_models_path, model_name)

        # if ip_negative is None:
        #     ip_negative = model_helper.load_torch_file(ip_negative_path)['data']
            # clip_vision_h_uc = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clip_vision_h_uc.data')
            # clip_vision_h_uc = torch.load(clip_vision_h_uc)['uc']
            # ip_negative_path = os.path.join(modules.paths.clip_vision_models_path, 'clip_vision_h_uc.data')
            # ip_negative = model_helper.load_torch_file(ip_negative_path)['uc']

        self.load_device = model_loader.run_device("controlnet")
        self.offload_device = model_loader.offload_device("controlnet")
        self.dtype = devices.dtype(self.load_device)

        def load_model():
            ip_state_dict = model_helper.load_torch_file(ip_adapter_path)
            plus = "latents" in ip_state_dict["image_proj"]
            cross_attention_dim = ip_state_dict["ip_adapter"]["1.to_k_ip.weight"].shape[1]
            sdxl = cross_attention_dim == 2048
            sdxl_plus = sdxl and plus

            if plus:
                clip_extra_context_tokens = ip_state_dict["image_proj"]["latents"].shape[1]
                # clip_embeddings_dim = ip_state_dict["image_proj"]["latents"].shape[2]
                clip_embeddings_dim = ip_state_dict['image_proj']['proj_in.weight'].shape[1]
            else:
                clip_extra_context_tokens = ip_state_dict["image_proj"]["proj.weight"].shape[0] // cross_attention_dim
                # clip_embeddings_dim = None
                clip_embeddings_dim = ip_state_dict['image_proj']['proj.weight'].shape[1]

            ipa = IPAdapterModel(
                ip_state_dict,
                plus=plus,
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=clip_embeddings_dim,
                clip_extra_context_tokens=clip_extra_context_tokens,
                sdxl_plus=sdxl_plus
            )
            ipa.sdxl = sdxl
            ipa.to(self.offload_device, dtype=self.dtype)

            return ipa
        
        self.ip_adapter = load_model()
        self.image_proj_model = ModelPatcher(model=self.ip_adapter.image_proj_model, load_device=self.load_device, offload_device=self.offload_device)
        self.ip_layers = ModelPatcher(model=self.ip_adapter.ip_layers, load_device=self.load_device, offload_device=self.offload_device)

        ip_unconds = None
        
        return
    
    @torch.no_grad()
    @torch.inference_mode()
    def preprocess(self, img, clip_vision):
        global ip_unconds

        inputs = clip_vision.processor(images=img, return_tensors="pt")
        model_loader.load_model_gpu(clip_vision.patcher)
        pixel_values = inputs['pixel_values'].to(clip_vision.load_device)

        if clip_vision.dtype != torch.float32:
            precision_scope = torch.autocast
        else:
            precision_scope = lambda a, b: contextlib.nullcontext(a)

        with precision_scope(devices.get_autocast_device(clip_vision.load_device), torch.float32):
            outputs = clip_vision.model(pixel_values=pixel_values, output_hidden_states=True)

        if self.ip_adapter.plus:
            cond = outputs.hidden_states[-2].to(self.dtype)
            with precision_scope(devices.get_autocast_device(clip_vision.load_device), torch.float32):
                uncond = clip_vision.model(torch.zeros_like(pixel_values), output_hidden_states=True).hidden_states[-2]
        else:
            cond = outputs.image_embeds.to(self.dtype)
            uncond = torch.zeros_like(cond)

        model_loader.load_model_gpu(self.image_proj_model)
        cond = self.image_proj_model.model(cond).to(device=self.load_device, dtype=self.dtype)
        uncond = self.image_proj_model.model(uncond.to(cond))

        model_loader.load_model_gpu(self.ip_layers)

        if ip_unconds is None:
            # uncond = ip_negative.to(device=self.load_device, dtype=self.dtype)
            ip_unconds = [m(uncond).cpu() for m in self.ip_layers.model.to_kvs]

        ip_conds = [m(cond).cpu() for m in self.ip_layers.model.to_kvs]

        # self.image_emb = cond
        self.image_emb = ip_conds if self.ip_adapter.sdxl else cond
        self.uncond_image_emb = ip_conds if self.ip_adapter.sdxl else uncond

        return ip_conds


    @torch.no_grad()
    @torch.inference_mode()
    def patch_model(self, model, weight):
        new_model = model.clone()

        def set_model_patch_replace(model, patch_kwargs, key):
            to = model.model_options["transformer_options"]
            if "patches_replace" not in to:
                to["patches_replace"] = {}
            if "attn2" not in to["patches_replace"]:
                to["patches_replace"]["attn2"] = {}
            if key not in to["patches_replace"]["attn2"]:
                patch = CrossAttentionPatch(**patch_kwargs)
                to["patches_replace"]["attn2"][key] = patch
            else:
                to["patches_replace"]["attn2"][key].set_new_condition(**patch_kwargs)

        # '''
        # patch_name of sdv1-2: ("input" or "output" or "middle", block_id)
        # patch_name of sdxl: ("input" or "output" or "middle", block_id, transformer_index)
        # '''
        patch_kwargs = {
            "number": 0,
            "weight": weight,
            "ipadapter": self.ip_adapter,
            "dtype": self.dtype,
            "cond": self.image_emb,
            "uncond": self.uncond_image_emb,
            "sdxl": self.ip_adapter.sdxl,
        }
        
        # From https://github.com/laksjdjf/IPAdapter-ComfyUI
        if not self.ip_adapter.sdxl:
            for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
                set_model_patch_replace(new_model, patch_kwargs, ("input", id))
                patch_kwargs["number"] += 1
            for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
                set_model_patch_replace(new_model, patch_kwargs, ("output", id))
                patch_kwargs["number"] += 1
            set_model_patch_replace(new_model, patch_kwargs, ("middle", 0))
        else:
            for id in [4,5,7,8]: # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, patch_kwargs, ("input", id, index))
                    patch_kwargs["number"] += 1
            for id in range(6): # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, patch_kwargs, ("output", id, index))
                    patch_kwargs["number"] += 1
            for index in range(10):
                set_model_patch_replace(new_model, patch_kwargs, ("middle", 0, index))
                patch_kwargs["number"] += 1
        
        return new_model
    
# def attention(q, k, v, extra_options):
#     if not hasattr(F, "multi_head_attention_forward"):
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=extra_options["n_heads"]), (q, k, v))
#         sim = torch.einsum('b i d, b j d -> b i j', q, k) * (extra_options["dim_head"] ** -0.5)
#         sim = F.softmax(sim, dim=-1)
#         out = torch.einsum('b i j, b j d -> b i d', sim, v)
#         out = rearrange(out, '(b h) n d -> b n (h d)', h=extra_options["n_heads"])
#     else:
#         b, _, _ = q.shape
#         q, k, v = map(
#             lambda t: t.view(b, -1, extra_options["n_heads"], extra_options["dim_head"]).transpose(1, 2),
#             (q, k, v),
#         )
#         out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
#         out = out.transpose(1, 2).reshape(b, -1, extra_options["n_heads"] * extra_options["dim_head"])
#     return out

class CrossAttentionPatch:
    # forward for patching
    def __init__(self, weight, ipadapter, dtype, number, cond, uncond, sdxl):
        self.weights = [weight]
        self.ipadapters = [ipadapter]
        self.conds = [cond]
        self.unconds = [uncond]
        self.dtype = dtype
        self.number = number
        self.sdxl = sdxl
    
    def set_new_condition(self, weight, ipadapter, cond, uncond, dtype, number, sdxl):
        self.weights.append(weight)
        self.ipadapters.append(ipadapter)
        self.conds.append(cond)
        self.unconds.append(uncond)
        self.dtype = dtype
        self.sdxl = sdxl


    def __call__(self, n, context_attn2, value_attn2, extra_options):
        org_dtype = n.dtype
        frame = inspect.currentframe()
        outer_frame = frame.f_back
        cond_or_uncond = outer_frame.f_locals["transformer_options"]["cond_or_uncond"]

        with torch.autocast("cuda", dtype=self.dtype):
            q = n
            k = [context_attn2]
            v = [value_attn2]
            b, _, _ = q.shape
            batch_prompt = b // len(cond_or_uncond)
            out = None

            for weight, cond, uncond, ipadapter in zip(self.weights, self.conds, self.unconds, self.ipadapters):
                if self.sdxl:
                    ip_k_c = cond[self.number * 2].to(q)
                    ip_v_c = cond[self.number * 2 + 1].to(q)
                    ip_k_uc = uncond[self.number * 2].to(q)
                    ip_v_uc = uncond[self.number * 2 + 1].to(q)

                    ip_k = torch.cat([(ip_k_c, ip_k_uc)[i] for i in cond_or_uncond], dim=0)
                    ip_v = torch.cat([(ip_v_c, ip_v_uc)[i] for i in cond_or_uncond], dim=0)
                    
                    ip_v_mean = torch.mean(ip_v, dim=1, keepdim=True)
                    ip_v_offset = ip_v - ip_v_mean

                    # B, F, C = ip_k.shape
                    # channel_penalty = float(C) / 1280.0
                    # weight = weight * channel_penalty

                    ip_k = ip_k * weight
                    ip_v = ip_v_offset + ip_v_mean * weight

                    k.append(ip_k)
                    v.append(ip_v)
                else:
                    uncond_cond = torch.cat([(cond.repeat(batch_prompt, 1, 1), uncond.repeat(batch_prompt, 1, 1))[i] for i in cond_or_uncond], dim=0)

                    # k, v for ip_adapter
                    ip_k = ipadapter.ip_layers.to_kvs.to(q)[self.number*2](uncond_cond)
                    ip_v = ipadapter.ip_layers.to_kvs.to(q)[self.number*2+1](uncond_cond)

                    # ip_out = attention(q, ip_k, ip_v, extra_options)
                    ip_out = sdp(q, ip_k, ip_v, extra_options)
                    # out = ip_out * weight if out is None else out + ip_out * weight
                    out = ip_out * weight

            if self.sdxl:
                k = torch.cat(k, dim=1)
                v = torch.cat(v, dim=1)
                out = sdp(q, k, v, extra_options)

        return out.to(dtype=org_dtype)