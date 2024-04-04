import os
import cv2
import numpy as np
import torch
from PIL import Image

import modules.paths
from modules.model import controlnet, clip_vision, model_loader
from modules import devices, util
from modules.controlnet.processor import Processor as controlnet_processor

controlnet_files = {}

def get_controlnets():
    global controlnet_files
    files = util.list_files(modules.paths.controlnet_models_path, ["bin", "pth", "safetensors"])
    controlnet_files = { os.path.splitext(os.path.split(x)[-1])[0]: os.path.split(x)[-1] for x in files }

    return controlnet_files

def processor(controlnets, unet_model, width, height, image_mask, dtype_ctrl=None, dtype_ipa=None):
    ctrl_procs = load_controlnets_by_task(controlnets, dtype_ctrl=dtype_ctrl, dtype_ipa=dtype_ipa)
    ctrls = []
    ip_procs = []

    clip_vision_path = os.path.join(modules.paths.clip_vision_models_path, 'clip_vision_vit_h.safetensors')
    clip_vision_model = None
    ip_proc = None

    for cn in controlnets:
        cn_type, cn_img, cn_weight, cn_model_name, cn_mask, start_percent, end_percent = cn
        cn_items = cn_type.split(",")
        cn_model = None
        cn_img = cv2.cvtColor(cn_img, cv2.COLOR_BGR2RGB)
        
        if cn_mask is not None:
            cn_mask = util.resize_image(cn_mask, width, height)[:, :, 0]
            cn_mask = util.numpy_to_pytorch(cn_mask)

        for cn_item in cn_items:
            # if cn_item in ctrl_procs.keys():
            if cn_model_name in ctrl_procs.keys():
                if cn_item in ["lama_inpaint"]:
                    lama_mask = image_mask.copy()
                    lama_mask = lama_mask[..., np.newaxis]
                    cn_img = np.concatenate([cn_img[:lama_mask.shape[0], :lama_mask.shape[1], :], lama_mask], axis=-1)

                # ctrl_proc = ctrl_procs[cn_item]
                ctrl_proc = ctrl_procs[cn_model_name]

                if cn_item in ["ip_adapter", "ip_adapter_face"]:
                    clip_vision_model = clip_vision_model or clip_vision.load(clip_vision_path)
                    ip_proc = ctrl_proc.processor
                    image_emb, uncond_image_emb = ip_proc.preprocess(cn_img, clip_vision_model)
                    # ip_adapters.append((ip_proc.preprocess(ip_img, clip_vision_model), 1, cn_weight))
                    unet_model = ip_proc.patch_model(unet_model, image_emb, uncond_image_emb, cn_weight, cn_mask, start_percent, end_percent)
                    ip_procs.append(ip_proc)
                else:
                    cn_img = util.resize_image(cn_img, width=width, height=height)
                    cn_img = util.HWC3(ctrl_proc(cn_img))
                    if cn_model is None:
                        cn_model = ctrl_proc.load_controlnet(cn_model_name, want_use_dtype=dtype_ctrl)

                util.save_temp_image(cn_img, f"{cn_item}.png")
            else:
                print(f"unknow controlnet {cn_item}")
        
        
        if cn_model is not None:
            if "recolor" in cn_model_name:
                end_percent = 1.0
            ctrls.append((cn_type, cn_img, cn_weight, cn_model, cn_mask, start_percent, end_percent))
    
    return ctrls, ip_procs, unet_model

def load_controlnets_by_task(cn_types, dtype_ctrl=None, dtype_ipa=None):
    ctrls = {}

    for cn_type, cn_image, cn_weight, cn_model_name, cn_mask, start_percent, end_percent in cn_types:
        cn_items = cn_type.split(",")
        for cn_item in cn_items:
            # if cn_item not in ctrls:
            if cn_model_name not in ctrls:
                if cn_item in controlnet_processor.model_keys() and cn_model_name is not None and end_percent > start_percent:
                    if cn_item in ["ip_adapter", "ip_adapter_face"]:
                        proc = controlnet_processor(cn_item)
                        proc.load_processor()
                        proc.processor.load(cn_item, cn_model_name, want_use_dtype=dtype_ipa)
                    else:
                        proc = controlnet_processor(cn_item)
                    
                    ctrls[cn_model_name] = proc

    return ctrls

@torch.no_grad()
@torch.inference_mode()
def apply_controlnets(positive_cond, negative_cond, ctrls):
    for cn in ctrls:
        cn_type, cn_img, cn_weight, cn_model, cn_mask, start_percent, end_percent = cn

        if cn_model is not None :
            positive_cond, negative_cond = apply_controlnet(
                positive_cond, negative_cond,
                cn_model, util.numpy_to_pytorch(cn_img),
                util.numpy_to_pytorch(cn_mask) if cn_mask is not None else None,
                cn_weight, start_percent, end_percent)
    
    return positive_cond, negative_cond

@torch.no_grad()
@torch.inference_mode()
def apply_controlnet(positive, negative, control_net, image, mask, strength, start_percent, end_percent):
    if strength == 0:
        return (positive, negative)

    control_hint = image.movedim(-1,1)
    cnets = {}

    out = []
    for conditioning in [positive, negative]:
        c = []
        for t in conditioning:
            d = t[1].copy()

            prev_cnet = d.get('control', None)
            if prev_cnet in cnets:
                c_net = cnets[prev_cnet]
            else:
                c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent))
                # c_net.device = devices.get_torch_device()
                c_net.set_previous_controlnet(prev_cnet)
                cnets[prev_cnet] = c_net

            d['control'] = c_net
            d['control_apply_to_uncond'] = False
            n = [t[0], d]
            c.append(n)
        out.append(c)
    return (out[0], out[1])