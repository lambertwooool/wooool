from modules.patches import patch_all

import os
import re
import torch
import numpy as np
import modules.paths
from PIL import Image
from modules.model import sd, model_helper
from modules.paths import embeddings_path
from modules import lora, util, shared, vae_helper
from modules.model.model_base import BaseModel, SDXL, SDXLRefiner

patches = patch_all()

class StableDiffusionModel:
    def __init__(self, unet, vae, clip, clip_vision):
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision

@torch.no_grad()
@torch.inference_mode()
def load_model(ckpt_filename, output_clip=True, output_vae=True):
    unet, clip, vae, clip_vision = sd.load_checkpoint_guess_config(ckpt_filename, embedding_directory=embeddings_path, output_clip=output_clip, output_vae=output_vae)
    return StableDiffusionModel(unet=unet, clip=clip, vae=vae, clip_vision=clip_vision)

@torch.no_grad()
@torch.inference_mode()
def load_sd_lora(model, lora_filename, strength_model=1.0, strength_clip=1.0):
    if strength_model == 0 and strength_clip == 0:
        return model

    lora_model = model_helper.load_torch_file(lora_filename, safe_load=False)
    new_unet, new_clip = sd.load_lora_for_models(model.unet, model.clip, lora_model, strength_model, strength_clip)

    return StableDiffusionModel(unet=new_unet, clip=new_clip, vae=model.vae, clip_vision=model.clip_vision)

def get_sd_model(base_path, refiner_path, steps, loras):
    xl_base = get_base_model(base_path)
    xl_base_patched = get_loras(xl_base, loras)

    if steps[1] > 0:
        xl_refiner = get_refiner_model(refiner_path)
        # xl_refiner.clip = xl_base.clip
        # xl_refiner = get_loras(xl_refiner, loras)
    else:
        xl_refiner = None
    
    return xl_base, xl_base_patched, xl_refiner

def get_base_model(model_path):
    if shared.xl_base is not None and shared.xl_base[0] == model_path:
        xl_base = shared.xl_base[1]
        print(f'Base model loaded from cache: {model_path}')
    else:
        xl_base = load_model(model_path)
        shared.xl_base = (model_path, xl_base)
        print(f'Base model loaded: {model_path}')
    
    return xl_base

def get_refiner_model(model_path):
    if shared.xl_refiner is not None and shared.xl_refiner[0] == model_path:
        xl_refiner = shared.xl_refiner[1]
        print(f'Refiner model loaded from cache: {model_path}')
    else:
        xl_refiner = load_model(model_path)
        shared.xl_refiner = (model_path, xl_refiner)
        print(f'Refiner model loaded: {model_path}')

        if isinstance(xl_refiner.unet.model, SDXL):
            xl_refiner.clip = None
            xl_refiner.vae = None
        elif isinstance(xl_refiner.unet.model, SDXLRefiner):
            xl_refiner.clip = None
            xl_refiner.vae = None
        else:
            xl_refiner.clip = None

    return xl_refiner

def get_loras(model, loras):
    xl_base_patched = model

    for name, weight, prompt in loras:
        # lora_path = os.path.join(modules.paths.lorafile_path, name)
        lora_path = lora.get_lora_path(name)
        if lora_path and os.path.exists(lora_path):
            xl_base_patched = load_sd_lora(xl_base_patched, lora_path, strength_model=weight, strength_clip=weight)
            print(f'[LoRA loaded] {name}, weight ({weight})')
        else:
            print(f'[LoRA Ignore] {name}')

    return xl_base_patched

@torch.no_grad()
@torch.inference_mode()
def vae_sampled(sampled_latent, vae_model, tiled, task, cur_batch, cur_seed, cur_subseed, filename, image_pixel, image_mask, image_pixel_orgin, re_zoom_point):
    decoded_latent = vae_helper.decode_vae(vae=vae_model, latent_image=sampled_latent, tiled=tiled)
    images = util.pytorch_to_numpy(decoded_latent)

    prompt, negative_prompt = task["prompt"].pop(0), task["negative"].pop(0)
    info = create_infotext(task, prompt, negative_prompt, cur_seed, cur_subseed, cur_batch)
    atime, mtime = os.path.getatime(filename), os.path.getmtime(filename)

    for idx, image in enumerate(images):
        if image_mask is not None and image_pixel is not None:
            if image_pixel.shape != image.shape:
                image_pixel = util.resize_image(image_pixel, image.shape[1], image.shape[0])
                image_mask = util.resize_image(image_mask, image.shape[1], image.shape[0])[:,:,0]
            all_mask = image_mask.astype(np.float32) / 255
            all_mask = all_mask.reshape(all_mask.shape[0], all_mask.shape[1], 1)
            image = (image_pixel * (1 - all_mask)  + image * all_mask).astype(np.uint8)
        if image_pixel_orgin is not None:
            min_x, min_y, max_x, max_y = re_zoom_point
            util.save_temp_image(image, "inpaint.png")
            image_inpaint = util.resize_image(image, max_x - min_x, max_y - min_y)
            image = image_pixel_orgin.copy()
            if image_mask is not None:
                image_inpaint_mask = util.resize_image(image_mask, max_x - min_x, max_y - min_y)[:,:,0]
                image_inpaint_mask = image_inpaint_mask.astype(np.float32) / 255
                image_inpaint_mask = image_inpaint_mask.reshape(image_inpaint_mask.shape[0], image_inpaint_mask.shape[1], 1)
                image[min_y:max_y, min_x:max_x, :] = (image[min_y:max_y, min_x:max_x, :] * (1 - image_inpaint_mask)  + image_inpaint * image_inpaint_mask).astype(np.uint8)
            else:
                image[min_y:max_y, min_x:max_x, :] = image_inpaint
        
        if idx == 0:
            filename_idx = os.path.join(modules.paths.temp_outputs_path, filename)
        else:
            filename_idx = f"_{idx}".join(os.path.splitext(os.path.split(filename)[-1]))
            filename_idx = os.path.join(modules.paths.temp_outputs_path, filename_idx)

        util.save_image_with_geninfo(Image.fromarray(image), info, filename_idx)
        if idx == 0:
            os.utime(filename_idx, (atime, mtime))
        print(filename_idx)

def generate_empty_latent(width=1024, height=1024, batch_size=1):
    latent = torch.zeros([batch_size, 4, height // 8, width // 8])
    return {"samples":latent}

def assert_model_integrity(xl_base, xl_refiner):
    error_message = None
    model_type = None
    refiner_model_type = None

    if xl_base is None:
        error_message = 'You have not selected base model.'

    if isinstance(xl_base.unet.model, SDXL):
        model_type = "sdxl"
    elif isinstance(xl_base.unet.model, BaseModel):
        model_type = "sd15"
    else:
        error_message = 'You have selected base model other than SDXL or SD15. This is not supported yet.'

    if xl_refiner is not None:
        if xl_refiner.unet is None or xl_refiner.unet.model is None:
            error_message = 'You have selected an invalid refiner!'
        elif not isinstance(xl_refiner.unet.model, SDXL) and not isinstance(xl_refiner.unet.model, SDXLRefiner):
            refiner_model_type = "sdxl"
        elif isinstance(xl_base.unet.model, BaseModel):
            refiner_model_type = "sd15"
        else:
            error_message = 'refiner model other than SDXL or SD15, not supported.'

    if error_message is not None:
        raise NotImplementedError(error_message)

    return model_type, refiner_model_type

def create_infotext(task, prompt, negative_prompt, seed, subseed, batch_index=0):
    step_base, step_refiner = task["steps"]
    width, height = task["size"]
    # style_type, sytle_name = task.get("style", ("", ""))
    re_line = r"[\n\r\s]"
    re_s = r"[\s,]"
    loras = task.get("lora", [])
    if task.get("prompt_temp"):
        prompt.pop(0)
    prompt = ",".join(prompt)
    prompt += " " + ", ".join([f"<lora:{os.path.splitext(lo_name)[0]}:{lo_weight}>" for lo_name, lo_weight, lo_trained_words in loras if lo_name])
    main_character = task.get("main_character")

    generation_params = {
        "Prompt": prompt,
        "Negative prompt": negative_prompt,
        "Steps": step_base + step_refiner,
        "Refiner Steps": step_refiner,
        "Sampler": task["sampler"],
        "Scheduler": task["scheduler"],
        "CFG scale": task["cfg_scale"],
        "Clip skip": abs(task.get("clip_skip", 2)),
        "Seed": seed,
        "Sub seed": subseed,
        "Size": f'{width}x{height}',
        "Model": task.get("base_model", "") and model_helper.get_model_filename(task["base_model"]),
        "Model hash": task.get("base_hash", None) or "",
        "Refiner model": task.get("refiner_model", "") and model_helper.get_model_filename(task["refiner_model"]),
        "Refiner model hash": task.get("refiner_hash", None) or "",
        "Denoising strength": round(task.get("denoise"), 2) if task.get("denoise") else None,
        # "Loras": task.get("lora", []),
        "Controlnets": [[x[0], x[2]] for x in task.get("controlnet", [])],
        "SDXL style": task.get("style", None),
        "Main character": main_character[1] if isinstance(main_character, dict) else "",
        "Version": "Wooool",
    }

    generation_params = { k: v for k, v in generation_params.items() if v }

    if generation_params.get("Denoising strength") == 1.0:
        generation_params.pop("Denoising strength")
    if seed == subseed:
        generation_params.pop("Sub seed")

    prompt_text = generation_params.pop("Prompt")
    negative_prompt_text = re.sub(re_line, " ", f'Negative prompt: {util.listJoin(generation_params.pop("Negative prompt"))}')
    generation_params_text = ", ".join([f'{k}: {re.sub(re_s, " ", str(v))}' for k, v in generation_params.items() if v is not None])
    # generation_params_text += ", \n" + ", \n".join([f'{k}: {re.sub(re_s, " ", str(v))}' for k, v in extra_params.items() if v is not None])
    
    return f"{prompt_text}\n{negative_prompt_text}\n{generation_params_text}".strip()

def calculate_sigmas_all(sampler, model, scheduler, steps):
    from modules.model.samplers import calculate_sigmas_scheduler

    discard_penultimate_sigma = False
    if sampler in ['dpm_2', 'dpm_2_ancestral']:
        steps += 1
        discard_penultimate_sigma = True

    sigmas = calculate_sigmas_scheduler(model, scheduler, steps)

    if discard_penultimate_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas

def calculate_sigmas(sampler, model, scheduler, steps, denoise):
    if denoise is None or denoise > 0.9999:
        sigmas = calculate_sigmas_all(sampler, model, scheduler, steps)
    else:
        new_steps = int(steps / denoise)
        sigmas = calculate_sigmas_all(sampler, model, scheduler, new_steps)
        sigmas = sigmas[-(steps + 1):]
    return sigmas