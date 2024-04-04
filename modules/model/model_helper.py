import importlib
import json
import math
import os
import re
import time
import torch
import struct
import safetensors.torch
import numpy as np
from PIL import Image

import modules.paths
from modules import civitai, devices, util, options
from modules.model import checkpoint_pickle, supported_models_base, supported_models

base_files = []
default_base_name = ""
default_refiner_name = ""

def load_torch_file(ckpt, safe_load=False, device=None, dtype=None):
    if not os.path.exists(ckpt):      
        print(f"[Load File Error] {ckpt} not exist.")
        return None

    start_time = time.perf_counter()
    if device is None:
        device = torch.device("cpu")
    ext = os.path.splitext(ckpt)[1]
    if ext == ".safetensors":
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    elif ".torchscript" in ckpt:
        sd = torch.jit.load(ckpt, map_location=device)
    else:
        if safe_load:
            if not 'weights_only' in torch.load.__code__.co_varnames:
                print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location=device, pickle_module=checkpoint_pickle)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd

    if dtype is not None:
        convert_sd_dtype(sd, dtype)

    print(f"[Load File][{time.perf_counter() - start_time:.2f}s] {ckpt}")
    return sd

def save_torch_file(sd, ckpt, metadata=None):
    if metadata is not None:
        safetensors.torch.save_file(sd, ckpt, metadata=metadata)
    else:
        safetensors.torch.save_file(sd, ckpt)

def convert_sd_dtype(sd, dtype):
    def convert(sd, dtype):
        for k in sd:
            sd[k] = sd[k].to(dtype) if hasattr(sd[k], "to") else convert(sd[k], dtype)
        return sd
    convert(sd, dtype)

def get_model_type_by_size(model_path):
    filesize = os.path.getsize(model_path)
    return "sdxl" if filesize > 4 * (1024 ** 3) else "sd15"

def get_model_info(model_path, calc_hash=True):
    config_file = f"{model_path}.wooool"
    data = None
    if os.path.exists(config_file):
        data = util.load_json(config_file)
    
    if data is None or not data or (calc_hash and not data.get("sha256", "")):
        if os.path.exists(f"{model_path}{civitai.suffix}"):
            civitai_info = civitai.get_model_versions(model_path)
            if civitai_info is not None:
                model_types = {
                    "SD 1.5": "sd15",
                    "SDXL 1.0": "sdxl",
                }
                baseMode = civitai_info["baseModel"]
                data = {
                    "sha256": civitai_info["files"][0]["hashes"]["SHA256"].lower(),
                    "base_model": model_types.get(baseMode) or baseMode,
                }
        
        if data is None:
            data = {
                "base_model": get_model_type_by_size(model_path),
            }
            if calc_hash:
                data["sha256"] = util.gen_file_sha256(model_path)

        if data and data.get("sha256", ""):
            with open(config_file, "w") as file:
                file.write(json.dumps(data, indent=4))
    
    return data or {}

def get_base_list():
    global base_files, default_base_name, default_refiner_name

    base_files = modules.paths.modelfile_path
    files = util.list_files(base_files, ["checkpoint", "safetensors"], search_subdir=True)
    files = { re.sub(r"^(\w+)\\", r"[ \1 ] ", os.path.relpath(os.path.splitext(x)[0], base_files)) + f" ({get_model_info(x, calc_hash=False).get('base_model', '')})" : x for x in files }
    base_files = files
    refiner_files = files
    base_name = os.path.splitext(options.default_base_name)[0]
    default_base_name = ([k for k in files if base_name in k] + list(base_files.keys()) + [f"{base_name} (sdxl)"])[0]
    # default_base_name = base_name if base_name in base_files else list(base_files.keys())[0]
    options.default["base_model"] = default_base_name

    refiner_name = os.path.splitext(options.default_refiner_name)[0]
    default_refiner_name = ([k for k in files if refiner_name in k] + [f"{refiner_name} (sdxl)"])[0]
    # default_refiner_name = refiner_name if refiner_name in refiner_files else None
    
    return {
        default_base_name: os.path.join(modules.paths.modelfile_path, options.default_base_name),
        default_refiner_name: os.path.join(modules.paths.modelfile_path, options.default_refiner_name),
    } | files

def get_model_filename(model_name):
    filename = base_files.get(model_name, "")
    filename = os.path.split(filename)[1]
    filename = os.path.splitext(filename)[0]

    return filename    

base_files = get_base_list()

def count_blocks(state_dict_keys, prefix_string):
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if c == False:
            break
        count += 1
    return count

def detect_unet_config(state_dict, key_prefix, dtype):
    state_dict_keys = list(state_dict.keys())

    unet_config = { }

    y_input = '{}label_emb.0.0.weight'.format(key_prefix)
    if y_input in state_dict_keys:
        unet_config["num_classes"] = "sequential"
        unet_config["adm_in_channels"] = state_dict[y_input].shape[1]
    else:
        unet_config["adm_in_channels"] = None

    unet_config["dtype"] = dtype
    model_channels = state_dict['{}input_blocks.0.0.weight'.format(key_prefix)].shape[0]
    in_channels = state_dict['{}input_blocks.0.0.weight'.format(key_prefix)].shape[1]

    num_res_blocks = []
    channel_mult = []
    attention_resolutions = []
    transformer_depth = []
    context_dim = None
    use_linear_in_transformer = False

    current_res = 1
    count = 0

    last_res_blocks = 0
    last_transformer_depth = 0
    last_channel_mult = 0

    while True:
        prefix = '{}input_blocks.{}.'.format(key_prefix, count)
        block_keys = sorted(list(filter(lambda a: a.startswith(prefix), state_dict_keys)))
        if len(block_keys) == 0:
            break

        if "{}0.op.weight".format(prefix) in block_keys: #new layer
            if last_transformer_depth > 0:
                attention_resolutions.append(current_res)
            transformer_depth.append(last_transformer_depth)
            num_res_blocks.append(last_res_blocks)
            channel_mult.append(last_channel_mult)

            current_res *= 2
            last_res_blocks = 0
            last_transformer_depth = 0
            last_channel_mult = 0
        else:
            res_block_prefix = "{}0.in_layers.0.weight".format(prefix)
            if res_block_prefix in block_keys:
                last_res_blocks += 1
                last_channel_mult = state_dict["{}0.out_layers.3.weight".format(prefix)].shape[0] // model_channels

            transformer_prefix = prefix + "1.transformer_blocks."
            transformer_keys = sorted(list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys)))
            if len(transformer_keys) > 0:
                last_transformer_depth = count_blocks(state_dict_keys, transformer_prefix + '{}')
                if context_dim is None:
                    context_dim = state_dict['{}0.attn2.to_k.weight'.format(transformer_prefix)].shape[1]
                    use_linear_in_transformer = len(state_dict['{}1.proj_in.weight'.format(prefix)].shape) == 2

        count += 1

    if last_transformer_depth > 0:
        attention_resolutions.append(current_res)
    transformer_depth.append(last_transformer_depth)
    num_res_blocks.append(last_res_blocks)
    channel_mult.append(last_channel_mult)
    transformer_depth_middle = count_blocks(state_dict_keys, '{}middle_block.1.transformer_blocks.'.format(key_prefix) + '{}')

    if len(set(num_res_blocks)) == 1:
        num_res_blocks = num_res_blocks[0]

    if len(set(transformer_depth)) == 1:
        transformer_depth = transformer_depth[0]

    unet_config["use_checkpoint"] = False
    unet_config["image_size"] = 32
    unet_config["use_spatial_transformer"] = True
    unet_config["legacy"] = False
    unet_config["in_channels"] = in_channels
    unet_config["out_channels"] = 4
    unet_config["model_channels"] = model_channels
    unet_config["num_res_blocks"] = num_res_blocks
    unet_config["attention_resolutions"] = attention_resolutions
    unet_config["transformer_depth"] = transformer_depth
    unet_config["channel_mult"] = channel_mult
    unet_config["transformer_depth_middle"] = transformer_depth_middle
    unet_config['use_linear_in_transformer'] = use_linear_in_transformer
    unet_config["context_dim"] = context_dim

    return unet_config

def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            params += sd[k].nelement()
    return params

def state_dict_key_replace(state_dict, keys_to_replace):
    for x in keys_to_replace:
        if x in state_dict:
            state_dict[keys_to_replace[x]] = state_dict.pop(x)
    return state_dict

def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])), filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out

def clip_text_transformers_convert(sd, prefix_from, prefix_to):
    sd = transformers_convert(sd, prefix_from, "{}text_model.".format(prefix_to), 32)

    tp = "{}text_projection.weight".format(prefix_from)
    if tp in sd:
        sd["{}text_projection.weight".format(prefix_to)] = sd.pop(tp)

    tp = "{}text_projection".format(prefix_from)
    if tp in sd:
        sd["{}text_projection.weight".format(prefix_to)] = sd.pop(tp).transpose(0, 1).contiguous()
    return sd

def transformers_convert(sd, prefix_from, prefix_to, number):
    keys_to_replace = {
        "{}positional_embedding": "{}embeddings.position_embedding.weight",
        "{}token_embedding.weight": "{}embeddings.token_embedding.weight",
        "{}ln_final.weight": "{}final_layer_norm.weight",
        "{}ln_final.bias": "{}final_layer_norm.bias",
    }

    for k in keys_to_replace:
        x = k.format(prefix_from)
        if x in sd:
            sd[keys_to_replace[k].format(prefix_to)] = sd.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                    sd[k_to] = weights[shape_from*x:shape_from*(x + 1)]
    return sd

UNET_MAP_ATTENTIONS = {
    "proj_in.weight",
    "proj_in.bias",
    "proj_out.weight",
    "proj_out.bias",
    "norm.weight",
    "norm.bias",
}

TRANSFORMER_BLOCKS = {
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "norm3.weight",
    "norm3.bias",
    "attn1.to_q.weight",
    "attn1.to_k.weight",
    "attn1.to_v.weight",
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn2.to_q.weight",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "ff.net.0.proj.weight",
    "ff.net.0.proj.bias",
    "ff.net.2.weight",
    "ff.net.2.bias",
}

UNET_MAP_RESNET = {
    "in_layers.2.weight": "conv1.weight",
    "in_layers.2.bias": "conv1.bias",
    "emb_layers.1.weight": "time_emb_proj.weight",
    "emb_layers.1.bias": "time_emb_proj.bias",
    "out_layers.3.weight": "conv2.weight",
    "out_layers.3.bias": "conv2.bias",
    "skip_connection.weight": "conv_shortcut.weight",
    "skip_connection.bias": "conv_shortcut.bias",
    "in_layers.0.weight": "norm1.weight",
    "in_layers.0.bias": "norm1.bias",
    "out_layers.0.weight": "norm2.weight",
    "out_layers.0.bias": "norm2.bias",
}

UNET_MAP_BASIC = {
    ("label_emb.0.0.weight", "class_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "class_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "class_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "class_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias")
}

def unet_to_diffusers(unet_config):
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    transformer_depth = unet_config["transformer_depth"][:]
    transformer_depth_output = unet_config["transformer_depth_output"][:]
    num_blocks = len(channel_mult)

    transformers_mid = unet_config.get("transformer_depth_middle", None)

    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.{}".format(n, b)
            num_transformers = transformer_depth.pop(0)
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["down_blocks.{}.attentions.{}.{}".format(x, i, b)] = "input_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            n += 1
        for k in ["weight", "bias"]:
            diffusers_unet_map["down_blocks.{}.downsamplers.0.conv.{}".format(x, k)] = "input_blocks.{}.0.op.{}".format(n, k)

    i = 0
    for b in UNET_MAP_ATTENTIONS:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, b)] = "middle_block.1.{}".format(b)
    for t in range(transformers_mid):
        for b in TRANSFORMER_BLOCKS:
            diffusers_unet_map["mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b)] = "middle_block.1.transformer_blocks.{}.{}".format(t, b)

    for i, n in enumerate([0, 2]):
        for b in UNET_MAP_RESNET:
            diffusers_unet_map["mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.{}".format(n, b)

    num_res_blocks = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks[x] + 1) * x
        l = num_res_blocks[x] + 1
        for i in range(l):
            c = 0
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.{}".format(n, b)
            c += 1
            num_transformers = transformer_depth_output.pop()
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, b)] = "output_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "output_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            if i == l - 1:
                for k in ["weight", "bias"]:
                    diffusers_unet_map["up_blocks.{}.upsamplers.0.conv.{}".format(x, k)] = "output_blocks.{}.{}.conv.{}".format(n, c, k)
            n += 1

    for k in UNET_MAP_BASIC:
        diffusers_unet_map[k[1]] = k[0]

    return diffusers_unet_map

def repeat_to_batch_size(tensor, batch_size):
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]
    elif tensor.shape[0] < batch_size:
        return tensor.repeat([math.ceil(batch_size / tensor.shape[0])] + [1] * (len(tensor.shape) - 1))[:batch_size]
    return tensor

def convert_sd_to(state_dict, dtype):
    keys = list(state_dict.keys())
    for k in keys:
        state_dict[k] = state_dict[k].to(dtype)
    return state_dict

def safetensors_header(safetensors_path, max_size=100*1024*1024):
    with open(safetensors_path, "rb") as f:
        header = f.read(8)
        length_of_header = struct.unpack('<Q', header)[0]
        if length_of_header > max_size:
            return None
        return f.read(length_of_header)

def resolve_lowvram_weight(weight, model, key):
    if weight.device == torch.device("meta"): #lowvram NOTE: this depends on the inner working of the accelerate library so it might break.
        key_split = key.split('.')              # I have no idea why they don't just leave the weight there instead of using the meta device.
        op = get_attr(model, '.'.join(key_split[:-1]))
        weight = op._hf_hook.weights_map[key_split[-1]]
    return weight

def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    setattr(obj, attrs[-1], torch.nn.Parameter(value))
    del prev

def set_attr_param(obj, attr, value):
    return set_attr(obj, attr, torch.nn.Parameter(value, requires_grad=False))

def copy_to_param(obj, attr, value):
    # inplace update tensor instead of replacing it
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)

def get_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj

def batch_area_memory(area):
    # if xformers_enabled() or pytorch_attention_flash_attention():
    if devices.is_xformers_enable():
        #TODO: these formulas are copied from maximum_batch_area below
        return (area / 20) * (1024 * 1024)
    else:
        return (((area * 0.6) / 0.9) + 1024) * (1024 * 1024)

def maximum_batch_area():
    memory_free = devices.get_free_memory() / (1024 * 1024)
    # if xformers_enabled() or pytorch_attention_flash_attention():
    if devices.is_xformers_enable():
        #TODO: this needs to be tweaked
        area = 20 * memory_free
    else:
        #TODO: this formula is because AMD sucks and has memory management issues which might be fixed in the future
        area = ((memory_free - 1024) * 0.9) / (0.6)
    return int(max(area, 0))

def bislerp(samples, width, height):
    def slerp(b1, b2, r):
        '''slerps batches b1, b2 according to ratio r, batches should be flat e.g. NxC'''
        
        c = b1.shape[-1]

        #norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        #normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        #zero when norms are zero
        b1_normalized[b1_norms.expand(-1,c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1,c) == 0.0] = 0.0

        #slerp
        dot = (b1_normalized*b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        #technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0-r.squeeze(1))*omega)/so).unsqueeze(1)*b1_normalized + (torch.sin(r.squeeze(1)*omega)/so).unsqueeze(1) * b2_normalized
        res *= (b1_norms * (1.0-r) + b2_norms * r).expand(-1,c)

        #edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5] 
        res[dot < 1e-5 - 1] = (b1 * (1.0-r) + b2 * r)[dot < 1e-5 - 1]
        return res
    
    def generate_bilinear_data(length_old, length_new):
        coords_1 = torch.arange(length_old).reshape((1,1,1,-1)).to(torch.float32)
        coords_1 = torch.nn.functional.interpolate(coords_1, size=(1, length_new), mode="bilinear")
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)
        
        coords_2 = torch.arange(length_old).reshape((1,1,1,-1)).to(torch.float32) + 1
        coords_2[:,:,:,-1] -= 1
        coords_2 = torch.nn.functional.interpolate(coords_2, size=(1, length_new), mode="bilinear")
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2
    
    n,c,h,w = samples.shape
    h_new, w_new = (height, width)
    
    #linear w
    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = samples.gather(-1,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    #linear h
    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new)
    coords_1 = coords_1.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1,1,-1,1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = result.gather(-2,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result

def lanczos(samples, width, height):
    images = [Image.fromarray(np.clip(255. * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(np.array(image).astype(np.float32) / 255.0).movedim(-1, 0) for image in images]
    result = torch.stack(images)
    return result

def common_upscale(samples, width, height, upscale_method, crop):
        if crop == "center":
            old_width = samples.shape[3]
            old_height = samples.shape[2]
            old_aspect = old_width / old_height
            new_aspect = width / height
            x = 0
            y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            s = samples[:,:,y:old_height-y,x:old_width-x]
        else:
            s = samples

        if upscale_method == "bislerp":
            return bislerp(s, width, height)
        elif upscale_method == "lanczos":
            return lanczos(s, width, height)
        else:
            return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

#TODO: might be cleaner to put this somewhere else
import threading

class InterruptProcessingException(Exception):
    pass

interrupt_processing_mutex = threading.RLock()

interrupt_processing = False
def interrupt_current_processing(value=True):
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        interrupt_processing = value

def processing_interrupted():
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        return interrupt_processing

def throw_exception_if_processing_interrupted():
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        if interrupt_processing:
            interrupt_processing = False
            raise InterruptProcessingException()
        
def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))

@torch.inference_mode()
def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap = 8, upscale_amount = 4, out_channels = 3, pbar = None):
    output = torch.empty((samples.shape[0], out_channels, round(samples.shape[2] * upscale_amount), round(samples.shape[3] * upscale_amount)), device="cpu")
    for b in range(samples.shape[0]):
        s = samples[b:b+1]
        out = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device="cpu")
        out_div = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device="cpu")
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                s_in = s[:,:,y:y+tile_y,x:x+tile_x]

                ps = function(s_in).cpu()
                mask = torch.ones_like(ps)
                feather = round(overlap * upscale_amount)
                for t in range(feather):
                        mask[:,:,t:1+t,:] *= ((1.0/feather) * (t + 1))
                        mask[:,:,mask.shape[2] -1 -t: mask.shape[2]-t,:] *= ((1.0/feather) * (t + 1))
                        mask[:,:,:,t:1+t] *= ((1.0/feather) * (t + 1))
                        mask[:,:,:,mask.shape[3]- 1 - t: mask.shape[3]- t] *= ((1.0/feather) * (t + 1))
                out[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),round(x*upscale_amount):round((x+tile_x)*upscale_amount)] += ps * mask
                out_div[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),round(x*upscale_amount):round((x+tile_x)*upscale_amount)] += mask
                if pbar is not None:
                    pbar.update(1)

        output[b:b+1] = out/out_div
    return output