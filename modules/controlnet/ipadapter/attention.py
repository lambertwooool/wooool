import torch
from einops import rearrange
import torch.nn.functional as F
import modules.model.ldm.modules.attention as attention

def patch_unet_model(model, patch_kwargs):
    new_model = model.clone()
    is_sdxl = patch_kwargs.pop("sdxl")

    # From https://github.com/laksjdjf/IPAdapter-ComfyUI
    if not is_sdxl:
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

def tensor_to_size(source, dest_size):
    if isinstance(dest_size, torch.Tensor):
        dest_size = dest_size.shape[0]
    source_size = source.shape[0]

    if source_size < dest_size:
        shape = [dest_size - source_size] + [1]*(source.dim()-1)
        source = torch.cat((source, source[-1:].repeat(shape)), dim=0)
    elif source_size > dest_size:
        source = source[:dest_size]

    return source

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
    def __init__(self, weight, dtype, number, cond, uncond, attn_mask=None, sigma_start=0.0, sigma_end=1.0):
        self.weights = [weight]
        self.conds = [cond]
        self.unconds = [uncond]
        self.dtype = dtype
        self.number = number
        self.attn_masks = [attn_mask]
        self.sigma_start = [sigma_start]
        self.sigma_end = [sigma_end]
    
    def set_new_condition(self, weight, cond, uncond, dtype, number, attn_mask=None, sigma_start=0.0, sigma_end=1.0):
        self.weights.append(weight)
        self.conds.append(cond)
        self.unconds.append(uncond)
        self.attn_masks.append(attn_mask)
        self.sigma_start.append(sigma_start)
        self.sigma_end.append(sigma_end)


    def __call__(self, n, context_attn2, value_attn2, extra_options):
        org_dtype = n.dtype
        cond_or_uncond = extra_options["cond_or_uncond"]

        with torch.autocast("cuda", dtype=self.dtype):
            q = n
            k = []
            v = []
            w = []
            m = []
            sigma = extra_options["sigmas"][0] if "sigmas" in extra_options else None
            sigma = sigma.item() if sigma is not None else 999999999.9
            batch_size, seq_len, inner_dim = q.shape
            batch_prompt = batch_size // len(cond_or_uncond)
            n_heads = extra_options["n_heads"]
            dim_head = extra_options["dim_head"]
            _, _, oh, ow = extra_options["original_shape"]

            for weight, cond, uncond, mask, sigma_start, sigma_end in zip(self.weights, self.conds, self.unconds, self.attn_masks, self.sigma_start, self.sigma_end):
                if sigma > sigma_start or sigma < sigma_end:
                    continue

                ip_k_c = cond[self.number * 2].repeat(batch_prompt, 1, 1).to(q)
                ip_k_uc = uncond[self.number * 2].repeat(batch_prompt, 1, 1).to(q)
                ip_v_c = cond[self.number * 2 + 1].repeat(batch_prompt, 1, 1).to(q)
                ip_v_uc = uncond[self.number * 2 + 1].repeat(batch_prompt, 1, 1).to(q)

                ip_k = torch.cat([(ip_k_c, ip_k_uc)[i] for i in cond_or_uncond], dim=0)
                ip_v = torch.cat([(ip_v_c, ip_v_uc)[i] for i in cond_or_uncond], dim=0)

                B, F, C = ip_k.shape
                channel_penalty = float(C) / 1280.0
                weight = weight * channel_penalty

                k.append(ip_k * weight)
                v.append(ip_v * weight)
                w.append(weight)

                if mask is not None:
                    mask_h = oh / math.sqrt(oh * ow / seq_len)
                    mask_h = int(mask_h) + int((seq_len % int(mask_h)) != 0)
                    mask_w = seq_len // mask_h

                    mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
                    mask = tensor_to_size(mask, batch_prompt)

                    mask = mask.repeat(len(cond_or_uncond), 1, 1)
                    mask = mask.view(mask.shape[0], -1, 1).repeat(1, 1, out.shape[2])

                    # covers cases where extreme aspect ratios can cause the mask to have a wrong size
                    mask_len = mask_h * mask_w
                    if mask_len < seq_len:
                        pad_len = seq_len - mask_len
                        pad1 = pad_len // 2
                        pad2 = pad_len - pad1
                        mask = F.pad(mask, (0, 0, pad1, pad2), value=0.0)
                    elif mask_len > seq_len:
                        crop_start = (mask_len - seq_len) // 2
                        mask = mask[:, crop_start:crop_start+seq_len, :]

                    m.append(mask)
                else:
                    m.append(tensor.ones_like(batch_prompt))
            
            out = attention_ipadapter(q, context_attn2, value_attn2, extra_options)
            if len(k) > 0:
                ip_k_all = torch.cat(k, dim=1)
                ip_v_all = torch.cat(v, dim=1)
                ip_mask_all = torch.cat(m, dim=1)
                ip_weight = sum(w) / len(w)

                ip_v_mean = torch.mean(ip_v_all, dim=1, keepdim=True)
                ip_v_offset = ip_v_all - ip_v_mean
                ip_v_all = (ip_v_offset + ip_v_mean)

                out = out * (1 - ip_weight) + attention_ipadapter(q, ip_k_all, ip_v_all, extra_options) * ip_mask_all

        return out.to(dtype=org_dtype)