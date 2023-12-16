import time
import torch
from hashlib import md5

from modules import shared
from modules.model.model_base import SDXL, SDXLRefiner
from modules.model import sdxl_clip

@torch.no_grad()
@torch.inference_mode()
def clip_encode(clip, texts, model_type, pool_top_k=1):
    if len(texts) == 0:
        return None

    cond_list = []
    pooled_acc = 0

    for i, text in enumerate(texts):
        cond, pooled = clip_encode_single(clip, text, model_type)
        cond_list.append(cond)
        if i < pool_top_k:
            pooled_acc += pooled

    return [[torch.cat(cond_list, dim=1), {"pooled_output": pooled_acc}]]

@torch.no_grad()
@torch.inference_mode()
def clip_encode_single(clip, text, model_type, use_cache=True):
    start_time = time.perf_counter()
    clip_skip = clip.layer_idx or ""
    md5_text = f"{model_type}_{clip_skip}_{md5(text.encode()).hexdigest()}"
    cached = shared.clip_cond_cache.get(md5_text, None) if use_cache else None
    if cached is not None:
        print(f'[CLIP Cached][{time.perf_counter() - start_time:.2f}s][cache] {text}, ')
        return cached
    tokens = clip.tokenize(text)
    result = clip.encode_from_tokens(tokens, return_pooled=True)
    shared.clip_cond_cache[md5_text] = result
    print(f'[CLIP Encode][{time.perf_counter() - start_time:.2f}s][{clip.patcher.current_device}] {text}')
    
    return result

@torch.no_grad()
@torch.inference_mode()
def clip_separate_inner(c, p, target_model=None, target_clip=None):
    if target_model is None or isinstance(target_model, SDXLRefiner):
        c = c[..., -1280:].clone()
    elif isinstance(target_model, SDXL):
        c = c.clone()
    elif hasattr(target_clip, "cond_stage_model") and \
            (isinstance(target_clip.cond_stage_model, sdxl_clip.SDXLClipModel) or isinstance(target_clip.cond_stage_model, sdxl_clip.SDXLRefinerClipModel)):
        p = None
        c = c[..., :768].clone()

        final_layer_norm = target_clip.cond_stage_model.clip_l.transformer.text_model.final_layer_norm

        final_layer_norm_origin_device = final_layer_norm.weight.device
        final_layer_norm_origin_dtype = final_layer_norm.weight.dtype

        c_origin_device = c.device
        c_origin_dtype = c.dtype

        final_layer_norm.to(device='cpu', dtype=torch.float32)
        c = c.to(device='cpu', dtype=torch.float32)

        c = torch.chunk(c, int(c.size(1)) // 77, 1)
        c = [final_layer_norm(ci) for ci in c]
        c = torch.cat(c, dim=1)

        final_layer_norm.to(device=final_layer_norm_origin_device, dtype=final_layer_norm_origin_dtype)
        c = c.to(device=c_origin_device, dtype=c_origin_dtype)
    else:
        c = c.clone()

    return c, p

@torch.no_grad()
@torch.inference_mode()
def clip_separate(cond, target_model=None, target_clip=None):
    results = []

    for c, px in cond:
        p = px.get('pooled_output', None)
        c, p = clip_separate_inner(c, p, target_model=target_model, target_clip=target_clip)
        p = {} if p is None else {'pooled_output': p.clone()}
        results.append([c, p])

    return results