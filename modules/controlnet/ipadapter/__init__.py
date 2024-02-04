import inspect
import os
import cv2
from einops import rearrange
import torch
import torch.nn.functional as F
import contextlib
import modules.paths
import modules.model.clip_vision
import modules.model.ldm.modules.attention as attention
from modules import devices, shared, insightface_model, util
from modules.model import model_loader, model_helper
from modules.model.model_patcher import ModelPatcher
from .resampler import Resampler
from .faceid import ProjModelFaceIdPlus, MLPProjModelFaceId
from modules.model.clip_vision import clip_preprocess

class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

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

# Cross Attention to_k, to_v for IPAdapter
class To_KV(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        layers = []
        for i in range(4096):
            for k in ['k', 'v']:
                key = f'{i}.to_{k}_ip.weight'
                if key in state_dict:
                    value = state_dict[key]
                    layer = torch.nn.Linear(value.shape[1], value.shape[0], bias=False)
                    layer.weight = torch.nn.Parameter(value, requires_grad=False)
                    layers.append(layer)

        self.to_kvs = torch.nn.ModuleList(layers)

class IPAdapterModel(torch.nn.Module):
    def __init__(self, state_dict, cross_attention_dim=768, clip_embeddings_dim=1024,
                 is_plus=False, sdxl_plus=False, is_full=False, is_faceid=False, is_faceid_v2=False, is_instantid=False,
                 load_device=None, offload_device=None):
        super().__init__()
        self.plus = is_plus
        self.is_faceid = is_faceid
        self.is_faceid_v2 = is_faceid_v2
        self.is_instantid = is_instantid

        clip_extra_context_tokens = 16 if is_plus else 4

        if is_instantid:
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
        elif is_faceid:
            # self.image_proj_model = self.init_proj_faceid()
            if is_plus:
                self.image_proj_model = ProjModelFaceIdPlus(
                    cross_attention_dim=cross_attention_dim,
                    id_embeddings_dim=512,
                    clip_embeddings_dim=clip_embeddings_dim,
                    num_tokens=4,
                )
            else:
                self.image_proj_model = MLPProjModelFaceId(
                    cross_attention_dim=cross_attention_dim,
                    id_embeddings_dim=512,
                    num_tokens=16,
                )
        elif is_plus:
            if is_full:
                self.image_proj_model = MLPProjModel(
                    cross_attention_dim=cross_attention_dim,
                    clip_embeddings_dim=clip_embeddings_dim
                )
            else:
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
            clip_extra_context_tokens = state_dict["image_proj"]["proj.weight"].shape[0] // cross_attention_dim

            self.image_proj_model = ImageProjModel(
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=clip_embeddings_dim,
                clip_extra_context_tokens=clip_extra_context_tokens
            )

        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = To_KV(state_dict["ip_adapter"])
        self.ip_layers = ModelPatcher(model=ip_layers, load_device=load_device, offload_device=offload_device)

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
        pass

    def load(self, cn_type, model_name):
        global ip_negative

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
            tmp_state_dict = { "image_proj": {}, "ip_adapter": {} }

            for key in ip_state_dict.keys():
                if key.startswith("image_proj."):
                    tmp_state_dict["image_proj"][key.replace("image_proj.", "")] = ip_state_dict.get_tensor(key)
                elif key.startswith("ip_adapter."):
                    tmp_state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = ip_state_dict.get_tensor(key)
                else:
                    tmp_state_dict[key] = ip_state_dict.get(key)
            ip_state_dict = tmp_state_dict

            is_full = "proj.3.weight" in ip_state_dict['image_proj']
            # is_faceid = "0.to_q_lora.down.weight" in ip_state_dict["ip_adapter"]
            is_faceid = "faceid" in model_name
            is_faceid_v2 = "v2" in model_name if is_faceid else False
            is_instantid = "instant" in model_name
            is_plus = (
                is_full or
                "latents" in ip_state_dict["image_proj"] or
                "perceiver_resampler.proj_in.weight" in ip_state_dict["image_proj"]
            )
            # plus = "latents" in ip_state_dict["image_proj"]
            cross_attention_dim = ip_state_dict["ip_adapter"]["1.to_k_ip.weight"].shape[1]
            sdxl = cross_attention_dim == 2048
            sdxl_plus = sdxl and is_plus

            if is_instantid:
                clip_embeddings_dim = 512
            elif is_faceid:
                if is_plus:
                    clip_embeddings_dim = 1280
                else:
                    # Plain faceid does not use clip_embeddings_dim.
                    clip_embeddings_dim = None
            elif is_plus:
                if sdxl_plus:
                    clip_embeddings_dim = int(ip_state_dict["image_proj"]["latents"].shape[2])
                elif is_full:
                    clip_embeddings_dim = int(ip_state_dict["image_proj"]["proj.0.weight"].shape[1])
                else:
                    clip_embeddings_dim = int(ip_state_dict['image_proj']['proj_in.weight'].shape[1])
                
                # clip_extra_context_tokens = ip_state_dict["image_proj"]["latents"].shape[1]
                # # clip_embeddings_dim = ip_state_dict["image_proj"]["latents"].shape[2]
                # clip_embeddings_dim = ip_state_dict['image_proj']['proj_in.weight'].shape[1]
            else:
                clip_embeddings_dim = int(ip_state_dict['image_proj']['proj.weight'].shape[1])

                # clip_extra_context_tokens = ip_state_dict["image_proj"]["proj.weight"].shape[0] // cross_attention_dim
                # # clip_embeddings_dim = None
                # clip_embeddings_dim = ip_state_dict['image_proj']['proj.weight'].shape[1]

            ipa = IPAdapterModel(
                ip_state_dict,
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=clip_embeddings_dim,
                is_plus=is_plus,
                sdxl_plus=sdxl_plus,
                is_full=is_full,
                is_faceid=is_faceid,
                is_faceid_v2=is_faceid_v2,
                is_instantid=is_instantid,
                load_device=self.load_device,
                offload_device=self.offload_device,
            )
            ipa.sdxl = sdxl
            ipa.to(self.offload_device, dtype=self.dtype)

            return ipa
        
        self.ip_adapter = load_model()
        self.image_proj_model = ModelPatcher(model=self.ip_adapter.image_proj_model, load_device=self.load_device, offload_device=self.offload_device)
        # self.ip_layers = ModelPatcher(model=self.ip_adapter.ip_layers, load_device=self.load_device, offload_device=self.offload_device)
        
        return
    
    @torch.no_grad()
    @torch.inference_mode()
    def preprocess(self, img, clip_vision):
        image_pixel = img.copy()
        clip_image = clip_preprocess(NPToTensor(image_pixel).unsqueeze(0)).float().to(clip_vision.load_device)
        cond_params = {}
        uncond_params = {}

        if clip_vision.dtype != torch.float32:
            precision_scope = torch.autocast
        else:
            precision_scope = lambda a, b: contextlib.nullcontext(a)

        with precision_scope(devices.get_autocast_device(clip_vision.load_device), torch.float32):
            model_loader.load_model_gpu(clip_vision.patcher)
            if self.ip_adapter.plus:
                clip_image_embeds = clip_vision.model(pixel_values=clip_image, intermediate_output=-2)[1]
                zero_clip_image = clip_preprocess(torch.zeros([1, 224, 224, 3])).float().to(clip_vision.load_device)
                uncond_clip_image_embeds = clip_vision.model(pixel_values=zero_clip_image, intermediate_output=-2)[1]
            else:
                clip_image_embeds = clip_vision.model(pixel_values=clip_image)[0]
                uncond_clip_image_embeds = torch.zeros_like(clip_image_embeds)

        if self.ip_adapter.is_instantid:
            analysis = insightface_model.Analysis(name="antelopev2")
            faces = analysis(img)
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
        elif self.ip_adapter.is_faceid:
            analysis = insightface_model.Analysis()
            faces = analysis(img)
            face_embeds = faces[0].normed_embedding
            face_embeds = torch.from_numpy(face_embeds).unsqueeze(0).to(clip_image_embeds)
        
            if self.ip_adapter.plus:
                cond_params = { "face_embeds": face_embeds, "shortcut": self.ip_adapter.is_faceid_v2, "scale": 1.0 }
                uncond_params = { "face_embeds": torch.zeros_like(face_embeds), "shortcut": self.ip_adapter.is_faceid_v2, "scale": 1.0 }
            else:
                clip_image_embeds = face_embeds
                uncond_clip_image_embeds = torch.zeros_like(face_embeds)

        with precision_scope(devices.get_autocast_device(self.load_device), self.dtype):
            model_loader.load_model_gpu(self.image_proj_model)
            cond = self.image_proj_model.model(clip_image_embeds, **cond_params).to(device=self.load_device, dtype=self.dtype)
            uncond = self.image_proj_model.model(uncond_clip_image_embeds, **uncond_params).to(cond)

            model_loader.load_model_gpu(self.ip_adapter.ip_layers)

            image_emb = [m(cond).cpu() for m in self.ip_adapter.ip_layers.model.to_kvs]
            uncond_image_emb = [m(uncond).cpu() for m in self.ip_adapter.ip_layers.model.to_kvs]

        return image_emb, uncond_image_emb

    @torch.no_grad()
    @torch.inference_mode()
    def patch_model(self, model, image_emb, uncond_image_emb, weight, start_at=0.0, end_at=1.0):
        new_model = model.clone()

        sigma_start = new_model.model.model_sampling.percent_to_sigma(start_at)
        sigma_end = new_model.model.model_sampling.percent_to_sigma(end_at)

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
            "dtype": self.dtype,
            "cond": image_emb,
            "uncond": uncond_image_emb,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
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
    
def attention_ipadapter(q, k, v, extra_options):
    if not hasattr(F, "multi_head_attention_forward"):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=extra_options["n_heads"]), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * (extra_options["dim_head"] ** -0.5)
        sim = F.softmax(sim, dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=extra_options["n_heads"])
    else:
        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.view(b, -1, extra_options["n_heads"], extra_options["dim_head"]).transpose(1, 2),
            (q, k, v),
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, extra_options["n_heads"] * extra_options["dim_head"])
    return out

def sdp(q, k, v, extra_options):
    return attention.optimized_attention(q, k, v, heads=extra_options["n_heads"], mask=None)

class CrossAttentionPatch:
    # forward for patching
    def __init__(self, weight, dtype, number, cond, uncond, sigma_start=0.0, sigma_end=1.0):
        self.weights = [weight]
        self.conds = [cond]
        self.unconds = [uncond]
        self.dtype = dtype
        self.number = number
        self.sigma_start = [sigma_start]
        self.sigma_end = [sigma_end]
    
    def set_new_condition(self, weight, cond, uncond, dtype, number, sigma_start=0.0, sigma_end=1.0):
        self.weights.append(weight)
        self.conds.append(cond)
        self.unconds.append(uncond)
        self.sigma_start.append(sigma_start)
        self.sigma_end.append(sigma_end)


    def __call__(self, n, context_attn2, value_attn2, extra_options):
        org_dtype = n.dtype
        cond_or_uncond = extra_options["cond_or_uncond"]

        with torch.autocast("cuda", dtype=self.dtype):
            q = n
            k = []
            v = []
            sigma = extra_options["sigmas"][0].item() if 'sigmas' in extra_options else 999999999.9
            batch_size, sequence_length, inner_dim = q.shape
            batch_prompt = batch_size // len(cond_or_uncond)
            n_heads = extra_options["n_heads"]
            dim_head = extra_options["dim_head"]

            for weight, cond, uncond, sigma_start, sigma_end in zip(self.weights, self.conds, self.unconds, self.sigma_start, self.sigma_end):
                if sigma > sigma_start or sigma < sigma_end:
                    continue

                ip_k_c = cond[self.number * 2].repeat(batch_prompt, 1, 1).to(q)
                ip_k_uc = uncond[self.number * 2].repeat(batch_prompt, 1, 1).to(q)
                ip_v_c = cond[self.number * 2 + 1].repeat(batch_prompt, 1, 1).to(q)
                ip_v_uc = uncond[self.number * 2 + 1].repeat(batch_prompt, 1, 1).to(q)
                
            #     ip_v_mean = torch.mean(ip_v, dim=1, keepdim=True)
            #     ip_v_offset = ip_v - ip_v_mean

            #     # B, F, C = ip_k.shape
            #     # channel_penalty = float(C) / 1280.0
            #     # weight = weight * channel_penalty

            #     ip_k = ip_k * weight
            #     ip_v = ip_v_offset + ip_v_mean * weight

                ip_k = torch.cat([(ip_k_c, ip_k_uc)[i] for i in cond_or_uncond], dim=0) * weight
                ip_v = torch.cat([(ip_v_c, ip_v_uc)[i] for i in cond_or_uncond], dim=0) * weight

                k.append(ip_k)
                v.append(ip_v)
            
            out = attention_ipadapter(q, context_attn2, value_attn2, extra_options)
            # for ik, iv in zip(k, v):
            #     out += sdp(q, ik, iv, extra_options)
            if k:
                ip_k_all = torch.cat(k, dim=1)
                ip_v_all = torch.cat(v, dim=1)
                out += attention_ipadapter(q, ip_k_all, ip_v_all, extra_options)

        return out.to(dtype=org_dtype)