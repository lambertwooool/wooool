import os
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

def processor(controlnets, unet_model, width, height, image_mask):
    ctrl_procs = load_controlnets_by_task(controlnets)
    ctrls = []

    clip_vision_path = os.path.join(modules.paths.clip_vision_models_path, 'clip_vision_vit_h.safetensors')
    clip_vision_model = None
    ip_proc = None

    for cn in controlnets:
        cn_type, cn_img, cn_weight, cn_model_name, start_percent, end_percent = cn
        cn_items = cn_type.split(",")
        cn_model = None

        for cn_item in cn_items:
            if cn_item in ctrl_procs.keys():
                if cn_item in ["lama_inpaint"]:
                    cn_mask = image_mask.copy()
                    cn_mask = cn_mask[..., np.newaxis]
                    cn_img = np.concatenate([cn_img[:cn_mask.shape[0], :cn_mask.shape[1], :], cn_mask], axis=-1)

                if cn_item in ["ip_adapter", "ip_adapter_face"]:
                    if cn_item == "ip_adapter_face":
                        _, face_img = util.get_faces(cn_img)
                        cn_img = face_img if face_img is not None else cn_img
                    # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                    ip_img = util.resize_image(cn_img, width=224, height=224) # , resize_mode=0
                    util.save_temp_image(ip_img, f"{cn_item}_224.png")
                    clip_vision_model = clip_vision_model or clip_vision.load(clip_vision_path)
                    ip_proc = ctrl_procs[cn_item].processor
                    ip_proc.preprocess(ip_img, clip_vision_model)
                    # ip_adapters.append((ip_proc.preprocess(ip_img, clip_vision_model), 1, cn_weight))
                    unet_model = ip_proc.patch_model(unet_model, cn_weight)
                else:
                    cn_img = util.resize_image(cn_img, width=width, height=height)
                    cn_img = util.HWC3(ctrl_procs[cn_item](cn_img))
                    if cn_model is None:
                        cn_model = ctrl_procs[cn_item].load_controlnet(cn_model_name)

                util.save_temp_image(cn_img, f"{cn_item}.png")
            else:
                print(f"unknow controlnet {cn_item}")
        
        
        if cn_model is not None:
            # start_percent = 0
            # end_percent = 0.5
            if "recolor" in cn_model_name:
                end_percent = 1.0
            # ctrls.append((cn_item, cn_img, cn_weight, cn_model, ctrl_procs[cn_item]))
            ctrls.append((cn_type, cn_img, cn_weight, cn_model, start_percent, end_percent))
    
    return ctrls, unet_model

def load_controlnets_by_task(cn_types):
    ctrls = {}

    for cn_type, cn_image, cn_weight, cn_model_name, start_percent, end_percent in cn_types:
        cn_items = cn_type.split(",")
        for cn_item in cn_items:
            if cn_item not in ctrls:
                if cn_item in controlnet_processor.model_keys() and cn_model_name is not None and end_percent > start_percent:
                    proc = controlnet_processor(cn_item)
                    if cn_item in ["ip_adapter", "ip_adapter_face"]:
                        proc.processor.load(cn_item, cn_model_name)
                    ctrls[cn_item] = proc

    return ctrls

@torch.no_grad()
@torch.inference_mode()
def apply_controlnets(positive_cond, negative_cond, ctrls):
    for cn in ctrls:
        cn_type, cn_img, cn_weight, cn_model, start_percent, end_percent = cn

        if cn_model is not None :
            positive_cond, negative_cond = apply_controlnet(
                positive_cond, negative_cond,
                cn_model, util.numpy_to_pytorch(cn_img), cn_weight, start_percent, end_percent)
    
    return positive_cond, negative_cond

@torch.no_grad()
@torch.inference_mode()
def apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent):
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