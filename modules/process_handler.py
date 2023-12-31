import copy
import json
import math
import os
import random
import re
import threading
import time
import einops
import numpy as np
import torch
from PIL import Image

import modules.options as opts
import modules.paths
from modules import clip_helper, civitai, core, controlnet_helper, devices, lora, prompt_helper, shared, util, vae_helper
from modules.model.model_base import BaseModel, SDXL, SDXLRefiner
from modules.model import model_loader, model_helper, sample, samplers

def handler(task):
    style = task.get("style", None)
    batch_size = task.get("batch", 1)
    round_batch_size = min(8, batch_size)

    seed = int(task.get("seed", 0))
    max_seed = 2 ** 63 - 1
    seed = (seed if seed > 0 else random.randint(1, max_seed)) % max_seed
    subseed = int(task.get("subseed", 0))
    subseed = (subseed if subseed > 0 else seed) % max_seed
    fixed_seed = task.get("fixed_seed", False)
    subseed_strength = task.get("subseed_strength", 1) if fixed_seed else 1
    seeds = [seed + (0 if subseed_strength < 1 else i) for i in range(round_batch_size)]
    subseeds = [subseed + i for i in range(round_batch_size)]

    loras = task.get("lora", [])

    base_name = task.get("base_name", "") or model_helper.default_base_name
    base_path = model_helper.base_files[base_name]
    if os.path.exists(base_path):
        civitai.get_model_versions(base_path)
        base_info = model_helper.get_model_info(base_path)
        base_hash = base_info.get("sha256")[:10]
        sd_type = "sd15" if base_info["base_model"] == "sd15" else "sdxl"
    else:
        base_hash = ""
        sd_type = "sd15" if "(sd15)" in base_name else "sdxl"

    skip_prompt = task.get("skip_prompt", False)

    positive, negative, style, loras = prompt_helper.generate(task.get("main_character", ("", "")), task.get("prompt_main", ""), task.get("prompt_negative", ""),
                                                        round_batch_size, style, task.get("style_weight", 1.0), task.get("options", {}), loras, task.get("lang", ""), seeds, sd_type, skip_prompt)
    
    quality = opts.options["quality_setting"][str(task.get("quality", 1))][sd_type]
    pixels, default_base_step, default_refiner_step, default_sampler, default_scheduler = quality
    
    # base_step_scale, refiner_step_scale = task.get("step_scale", 1.0), task.get("refiner_step_scale", 1.0)
    steps = task.get("steps", (default_base_step, default_refiner_step))
    # steps = (int(steps[0] * base_step_scale), int(steps[1] * refiner_step_scale))
    steps = (int(steps[0] or default_base_step), int(steps[1] or default_refiner_step))
    base_step, refiner_step = steps
    skip_step = task.get("skip_step", 0)
    skip_step = max(0, int(round(skip_step if skip_step >= 1 else base_step * skip_step)))
    
    refiner_name = task.get("refiner_name", "") or model_helper.default_refiner_name if refiner_step > 0 else None
    refiner_path = (refiner_name and model_helper.base_files[refiner_name]) or None if refiner_step > 0 else None
    if refiner_path is not None and os.path.exists(refiner_path):
        refiner_info = refiner_path and model_helper.get_model_info(refiner_path) if refiner_step > 0 else None
        refiner_hash = (refiner_info and refiner_info.get("sha256")[:10]) or "" if refiner_step > 0 else None
    else:
        refiner_hash = ""

    size = task.get("size", util.ratios2size(task.get("aspect_ratios", None), pixels ** 2))
    cfg_scale = task.get("cfg_scale", 7.0)
    cfg_scale_to = task.get("cfg_scale_to", cfg_scale)
    clip_skip = int(task.get("clip_skip") or -2)

    detail = float(task.get("detail", 0))
    noise_scale = 1 + detail * 0.05
    if detail != 0:
        detail_lora = { "sd15": "add_detail.safetensors", "sdxl": "add-detail-xl.safetensors" }[sd_type]
        loras.append((detail_lora, detail, ""))
    more_art = float(task.get("more_art_weight", 0))
    if more_art != 0:
        more_art_lora = { "sd15": "", "sdxl": "xl_more_art-full_v1.safetensors" }[sd_type]
        loras.append((more_art_lora, more_art, ""))
    denoise = round(task.get("denoise", 1.0), 2)
    sampler = task.get("sampler", "") or default_sampler
    scheduler = task.get("scheduler", "") or default_scheduler
    file_format = task.get("file_format", "jpeg").lower()   
    image, mask = task.get("image", (None, None))
    if size is None:
        size = util.ratios2size(util.size2ratio(image.shape[1], image.shape[0]) if image is not None else (1, 1), pixels ** 2)
    width, height = size
    size = (width // 8 * 8, height // 8 * 8)
    single_vae = task.get("single_vae", True)
    controlnets = [[    ref_mode, image_refer, sl_rate,
                        controlnet_helper.controlnet_files.get(opt_model) or opts.options["ref_mode"][opt_type][1][sd_type],
                        start_percent, end_percent ]
                    for opt_type, ref_mode, image_refer, sl_rate, opt_model, start_percent, end_percent in task.get("controlnet", []) \
                        if opt_model or opts.options["ref_mode"][opt_type][1][sd_type] is not None]

    tiled = False
    initial_latent = None

    task["prompt"] = positive
    task["negative"] = negative
    task["style"] = style
    task["batch"] = batch_size
    task["base_model"] = base_name
    task["base_hash"] = base_hash
    task["refiner_model"] = refiner_name
    task["refiner_hash"] = refiner_hash
    task["cfg_scale"] = cfg_scale
    task["cfg_scale_to"] = cfg_scale_to
    task["noise_scale"] = noise_scale
    task["denoise"] = denoise
    task["seed"] = seeds
    task["sampler"] = sampler
    task["scheduler"] = scheduler
    task["size"] = size
    task["steps"] = steps
    task["skip_step"] = skip_step
    task["clip_skip"] = clip_skip
    task["file_format"] = file_format
    task["lora"] = loras
    task["controlnet"] = controlnets

    # print(task)
    print(  '[positive]:', positive[1], \
            '\n[negative]:', negative[1], \
            '\n[style]:', style, \
            '\n[model]:', (f"{base_name}({base_hash})", f"{refiner_name}({refiner_hash})"), \
            '\n[size]:', size, \
            '\n[cfg_scale]:', (cfg_scale, cfg_scale_to), \
            '\n[image & mask]:', (True if image is not None else False, True if mask is not None else False), \
            '\n[clip_skip]:', clip_skip, \
            '\n[noise_scale]:', noise_scale, \
            '\n[denoise]:', denoise, \
            '\n[batch_size]:', batch_size, \
            '\n[quality]:', quality, \
            '\n[steps]:', steps, \
            '\n[skip_step]:', skip_step, \
            '\n[sampler, scheduler]', [sampler, scheduler], \
            '\n[seeds]:', seeds, \
            '\n[subseeds]:', subseeds, \
            '\n[subseed strength]:', subseed_strength, \
            '\n[file_format]:', file_format, \
            '\n[loras]:', loras, \
            '\n[controlnets]:', [(x[0], x[2], x[3], x[4], x[5]) for x in controlnets], \
            '\n-------------------------------'
        )

    task["batch_step"] = base_step + refiner_step + 2 # +2 = switch_model, vae
    task["cur_batch"] = 1

    def callback(*args):
        return

    process_diffusion(
                task=task,
                base_path=base_path,
                refiner_path=refiner_path,
                positive=positive,
                negative=negative,
                steps=steps,
                skip_step=skip_step,
                batch_size=batch_size,
                size=size,
                seeds=seeds,
                subseeds=subseeds,
                callback=callback,
                sampler_name=sampler,
                scheduler_name=scheduler,
                latent=initial_latent,
                image=(image, mask),
                noise_scale=noise_scale,
                denoise=denoise,
                tiled=tiled,
                cfg_scale=cfg_scale,
                cfg_scale_to=cfg_scale_to,
                loras=loras,
                controlnets=controlnets,
                single_vae=single_vae,
                round_batch_size=round_batch_size,
                subseed_strength=subseed_strength,
                clip_skip=clip_skip,
            )
    
    # model_loader.free_memory(1024 ** 4, torch.device(torch.cuda.current_device()))

    progress_output(task, "finished")

def stop(task):
    task["stop"] = True

def skip(task):
    task["skip"] = True

def update_model_info_base(model_path, model):
    config_file = f"{model_path}.wooool"
    
    if os.path.exists(config_file):
        data = util.load_json(config_file)
        base_model = data.get("base_model")
        if isinstance(model.unet.model, SDXL):
            if base_model == "sd15":
                base_model = "sdxl"
        elif isinstance(model.unet.model, SDXLRefiner):
            base_model = "sdxl refiner"

        if data["base_model"] != base_model:
            data["base_model"] = base_model
            with open(config_file, "w") as file:
                file.write(json.dumps(data, indent=4))


def progress_output(task, step_type, params=(), picture=None):
    wait_percent = 5
    batch = task["batch"]
    cur_batch = task["cur_batch"]
    create_time = task["create_time"]
    batch_step = task["batch_step"]
    total_step = batch_step * batch + 2 # clip, load_model
    cur_step = task.get("cur_step", 0) + 1
    task["cur_step"] = cur_step
    step_percent = (100 - wait_percent) / total_step
    cur_percent = cur_step * step_percent

    prefix = f"generate {cur_batch}/{batch} sample, "
    step_message = {
        "clip": "encoding prompt ...",
        "load_model": "loading model ...",
        "base_ksampler": prefix + "drawing {}/{} ...",
        "switch_model": prefix + "switch model ...",
        "refiner_ksampler": prefix + "refiner {}/{} ...",
        "vae": prefix + "vae sample ...",
        "save": prefix + "save sample ...",
        "finished": f"{batch} sample finished."
    }[step_type].format(*params)
    cur_percent = max(0, min(99, cur_percent) if step_type != "finished" else cur_percent)
    finished = False if step_type != "finished" else True

    message = f"<span>{step_message}</span> <span>({int(cur_percent)})% {time.time() - create_time:.1f}s</span>"

    util.output(task, cur_percent, message, finished, picture)

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
        xl_base = core.load_model(model_path)
        shared.xl_base = (model_path, xl_base)
        print(f'Base model loaded: {model_path}')
    
    return xl_base

def get_refiner_model(model_path):
    if shared.xl_refiner is not None and shared.xl_refiner[0] == model_path:
        xl_refiner = shared.xl_refiner[1]
        print(f'Refiner model loaded from cache: {model_path}')
    else:
        xl_refiner = core.load_model(model_path)
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
            xl_base_patched = core.load_sd_lora(xl_base_patched, lora_path, strength_model=weight, strength_clip=weight)
            print(f'[LoRA loaded] {name}, weight ({weight})')
        else:
            print(f'[LoRA Ignore] {name}')

    return xl_base_patched

@torch.no_grad()
@torch.inference_mode()
def vae_sampled(sampled_latent, vae_model, tiled, task, cur_batch, cur_seed, cur_subseed, filename, image_pixel, image_mask, image_pixel_orgin, re_zoom_point):
    decoded_latent = vae_helper.decode_vae(vae=vae_model, latent_image=sampled_latent, tiled=tiled)
    images = util.pytorch_to_numpy(decoded_latent)

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
        prompt, negative_prompt = task["prompt"].pop(0), task["negative"].pop(0)
        info = create_infotext(task, prompt, negative_prompt, cur_seed, cur_subseed, cur_batch)
        atime, mtime = os.path.getatime(filename), os.path.getmtime(filename)
        filename = os.path.join(modules.paths.temp_outputs_path, filename)
        util.save_image_with_geninfo(Image.fromarray(image), info, filename)
        os.utime(filename, (atime, mtime))
        print(filename)
        progress_output(task, "save", picture=image)

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

class UserStopException(Exception):
    pass

@torch.no_grad()
@torch.inference_mode()
def process_diffusion(task, base_path, refiner_path, positive, negative, steps, skip_step, size, seeds, subseeds,
        callback, sampler_name, scheduler_name,
        latent=None, image=None, denoise=1.0, noise_scale=1.0, cfg_scale=7.0, cfg_scale_to=7.0, batch_size=1, loras=[], controlnets=[],
        tiled=False, single_vae=True, round_batch_size=8, subseed_strength=1.0, clip_skip=0):

    devices.torch_gc()
    progress_output(task, "load_model")
    
    xl_base, xl_base_patched, xl_refiner = get_sd_model(base_path, refiner_path, steps, loras)
    model_type, refiner_model_type = assert_model_integrity(xl_base, xl_refiner)

    update_model_info_base(base_path, xl_base)

    progress_output(task, "clip")

    clip_model = xl_base_patched.clip
    clip_model.clip_layer(clip_skip)
    positive_template = positive.pop(0)
    negative_template = negative.pop(0)
    positive_cond = []
    negative_cond = []
    def clip_worker(positive, negative):
        for x, y in zip(positive, negative):
            positive_cond.append(clip_helper.clip_encode(clip_model, x, model_type))
            negative_cond.append(clip_helper.clip_encode(clip_model, y, model_type))
    
    thread_clip = threading.Thread(target=clip_worker, args=(positive, negative)).start()

    width, height = size
    step_base, step_refiner = steps
    step_total = step_base + step_refiner

    unet_model = xl_base_patched.unet
    refiner_unet_model = xl_refiner.unet if xl_refiner is not None else None
    
    vae_model = xl_refiner.vae if xl_refiner is not None and xl_refiner.vae is not None else xl_base_patched.vae
    vae_model.device = torch.device("cpu")
    single_vae = True if batch_size < 3 else single_vae
    vae_model_unet = vae_model if single_vae else copy.deepcopy(vae_model)
    vae_model_unet.device = unet_model.load_device

    image_pixel, image_mask = image
    image_pixel = util.resize_image(image_pixel, width, height)
    image_mask = util.resize_image(image_mask, width, height)
    image_pixel_orgin = None
    re_zoom_point = (0, 0, 0, 0)
    if latent is None:
        if image_mask is not None:
            p = np.where(image_mask > 0)
            min_y, min_x, max_y, max_x = np.min(p[0]), np.min(p[1]), np.max(p[0]), np.max(p[1])
            w, h = max_x - min_x, max_y - min_y
            pad_scale = 0.5
            min_x = max(0, int((min_x - w * pad_scale) / 8) * 8)
            min_y = max(0, int((min_y - h * pad_scale) / 8) * 8)
            max_x = min(image_mask.shape[1], round((max_x + w * pad_scale) / 8) * 8)
            max_y = min(image_mask.shape[0], round((max_y + h * pad_scale) / 8) * 8)
            w, h = max_x - min_x, max_y - min_y
            re_zoom = math.sqrt(w * h / 1152 ** 2)
            if re_zoom < 1.0:
                re_zoom_point = (min_x, min_y, max_x, max_y)
                image_mask = image_mask[min_y:max_y, min_x:max_x, :]
                # w, h = [round(x / 8) * 8 for x in util.ratios2size(util.size2ratio(w, h), 1024 ** 2)]
                w, h = [round(x / re_zoom / 8) * 8 for x in [w, h]]
                image_mask = util.resize_image(image_mask, w, h, resize_mode=0)
                height, width = image_mask.shape[:2]

                if image_pixel is not None:
                    image_pixel_orgin = image_pixel.copy()
                    image_pixel = image_pixel[min_y:max_y, min_x:max_x, :]
                    image_pixel = util.resize_image(image_pixel, w, h, resize_mode=0)
                
            for ctrl in controlnets:
                if ctrl[1] is None:
                    ctrl[1] = image_pixel.copy()
                elif image_pixel_orgin is not None and image_pixel_orgin.shape == ctrl[1].shape and not np.any(image_pixel_orgin - ctrl[1]):
                    ctrl[1] = image_pixel.copy()
                    
        else:
            image_mask = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        image_mask = image_mask[:, :, 0]

        if image_pixel is None:
            denoise = 1.0
            latent = generate_empty_latent(width=width, height=height, batch_size=1)
        else:
            image_pixel_torch = util.numpy_to_pytorch(image_pixel)
            image_mask_torch = util.numpy_to_pytorch(image_mask)
            latent = vae_helper.encode_vae_mask(vae_model_unet, image_pixel_torch, image_mask_torch, tiled=tiled)

    ctrls, unet_model = controlnet_helper.processor(controlnets, unet_model, width, height)
    
    sampled_latents = []
    vae_worker_running = True
    # VAE_approx_model = vae_helper.get_vae_approx(unet_model)

    def left_vae(sampled_latents, set_None=False):
        left_sampled_latents = sampled_latents[:]
        sampled_latents.clear()
        if set_None:
            sampled_latents.append(None)

        while left_sampled_latents:
            progress_output(task, "vae")
            sampled_latent, cur_batch, cur_seed, cur_subseed, filename = left_sampled_latents.pop(0)
            vae_sampled(sampled_latent, vae_model_unet, tiled, task, cur_batch, cur_seed, cur_subseed, filename, image_pixel, image_mask, image_pixel_orgin, re_zoom_point)
            print("cuda", len(left_sampled_latents))
    
    def vae_worker():
        while True and None not in sampled_latents:
            if sampled_latents and vae_worker_running:
                progress_output(task, "vae")
                sampled_latent, cur_batch, cur_seed, subseeds, filename = sampled_latents.pop(0)
                vae_sampled(sampled_latent, vae_model, tiled, task, cur_batch, cur_seed, subseeds, filename, image_pixel, image_mask, image_pixel_orgin, re_zoom_point)
                if sampled_latents is not None:
                    print("cpu", len(sampled_latents))
                else:
                    print("cpu", "last")
            time.sleep(0.01)
    
    thread_vae = threading.Thread(target=vae_worker)
    thread_vae.start()

    def callback_base_sample(step, x0, x, total_steps, is_refiner=False):
        if task.get("stop", False) or task.get("skip", False):
            task["skip"] = False
            raise UserStopException()
        
        preview_image = vae_helper.decode_vae_preview(refiner_unet_model if is_refiner else unet_model, x0)
        
        switch_step = step_base - 1
        if step == switch_step:
             progress_output(task, "switch_model")
        elif step < switch_step:
            progress_output(task, "base_ksampler", (step + 1, step_total), picture=preview_image)
            util.save_temp_image(preview_image, "generate.png")
        elif step > switch_step:
            progress_output(task, "refiner_ksampler", (step + 1, step_total), picture=preview_image)
            util.save_temp_image(preview_image, "generate.png")
    
    def callback_refine_sample(step, x0, x, total_steps):
        callback_base_sample(step + step_base, x0, x, total_steps, is_refiner=True)
    
    def sampler_cfg_function(args):
        cond = args.get("cond")
        uncond = args.get("uncond")
        cond_scale = args.get("cond_scale")
        timestep = args.get("timestep")
        alpha = cfg_scale_to + (cond_scale - cfg_scale_to) * timestep / 15
        return uncond + (cond - uncond) * alpha
    
    def model_function_wrapper(func, args):
        cond_or_uncond = args.pop("cond_or_uncond")
        return func(args.pop("input"), args.pop("timestep"), **args.pop("c"))

    unet_model.model_options["sampler_cfg_function"] = sampler_cfg_function
    unet_model.model_options["model_function_wrapper"] = model_function_wrapper
    
    latent_image = latent["samples"]
    latent_mask = latent.get("noise_mask", None)

    for i in range(batch_size):
        task["cur_batch"] = i + 1

        while not (positive_cond and negative_cond):
            time.sleep(0.01)
        
        cur_positive_cond, cur_negative_cond = controlnet_helper.apply_controlnets(positive_cond.pop(0), negative_cond.pop(0), ctrls)
        cur_seed = seeds.pop(0)
        cur_subseed = subseeds.pop(0)

        try:
            cur_noise = sample.prepare_noise(latent_image, cur_seed, None) * noise_scale
            start_step = skip_step

            if subseed_strength > 0 and subseed_strength < 1:
                # sub_seed = cur_seed + i
                sub_noise = sample.prepare_noise(latent_image, cur_subseed, None)
                cur_noise = util.slerp(subseed_strength, cur_noise, sub_noise)

            if step_base > start_step:
                sampled_latent = sample.sample(
                    model=unet_model,
                    positive=cur_positive_cond,
                    negative=cur_negative_cond,
                    latent_image=latent_image,
                    noise=cur_noise,
                    steps=step_total, start_step=start_step, last_step=step_base,
                    cfg=cfg_scale,
                    seed=cur_seed,
                    sampler_name=sampler_name,
                    scheduler=scheduler_name,
                    denoise=denoise, disable_noise=False, force_full_denoise=False,
                    callback=callback_base_sample,
                    noise_mask=latent_mask,
                    sigmas=None,
                    disable_pbar=False
                )
            else:
                sampled_latent = latent_image.clone()

            if refiner_unet_model is not None and step_refiner > 0:
                # org_image = vae_helper.decode_vae(xl_base.vae, sampled_latent)
                # sampled_latent = vae_interpose.parse(sampled_latent)
                # sampled_latent = vae_helper.encode_vae(xl_refiner.vae, org_image)["samples"]

                if latent_mask is not None:
                    pre_mask = sample.prepare_mask(latent_mask, sampled_latent.shape, torch.device("cpu"))
                    sampled_latent = sampled_latent * pre_mask + latent_image * (1 - pre_mask)
                    del pre_mask

                sigmas = None

                refiner_positive = clip_helper.clip_separate(cur_positive_cond, target_model=refiner_unet_model.model, target_clip=clip_model)
                del cur_positive_cond
                refiner_negative = clip_helper.clip_separate(cur_negative_cond, target_model=refiner_unet_model.model, target_clip=clip_model)
                del cur_negative_cond

                model_loader.free_memory(1024 ** 4, devices.get_torch_device())

                sampled_latent = sample.sample(
                    model=refiner_unet_model,
                    positive=refiner_positive,
                    negative=refiner_negative,
                    latent_image=sampled_latent,
                    noise=torch.zeros(sampled_latent.size(), dtype=sampled_latent.dtype, layout=sampled_latent.layout, device="cpu"),
                    # noise=cur_noise,
                    steps=step_total, start_step=step_base, last_step=step_total,
                    cfg=cfg_scale,
                    seed=cur_seed,
                    sampler_name=sampler_name,
                    scheduler=scheduler_name,
                    denoise=denoise, disable_noise=False, force_full_denoise=True,
                    callback=callback_refine_sample,
                    # noise_mask=latent_mask,
                    noise_mask=None,
                    sigmas=sigmas,
                    disable_pbar=False
                )

                del refiner_positive, refiner_negative
            else:
                del cur_positive_cond, cur_negative_cond
            
            model_loader.free_memory(1024 ** 4, devices.get_torch_device())

            pool_low_limit = 4
            if batch_size > round_batch_size and len(positive_cond) < pool_low_limit:
                positive_new, negative_new = prompt_helper.re_generate(positive_template, negative_template, round_batch_size - pool_low_limit)
                threading.Thread(target=clip_worker, args=(positive_new, negative_new)).start()
                positive += positive_new
                negative += negative_new
                seeds += [seeds[-1] + (0 if subseed_strength < 1 else i + 1) for i in range(round_batch_size - pool_low_limit)]
                subseeds += [subseeds[-1] + i + 1 for i in range(round_batch_size - pool_low_limit)]

            file_format = task["file_format"] or "jpeg"
            filename = f"{time.strftime('%Y%m%d%H%M%S')}_{i}.{file_format}"
            filename = os.path.join(modules.paths.temp_outputs_path, filename)
            # preview_image = vae_helper.decode_vae_preview(refiner_unet_model if step_refiner > 0 else unet_model, sampled_latent)
            util.save_image_with_geninfo(Image.open(os.path.join(f"{modules.paths.temp_outputs_path}/temp", "generate.png")), "pre_vae: True", filename, quality=60)

            if i > batch_size - 3:
                vae_worker_running = False

            sampled_latents.append((sampled_latent, i, cur_seed, cur_subseed, filename))
            # if single_vae or not vae_worker_running:
            if single_vae or len(sampled_latents) > 5:
                left_vae(sampled_latents)

        except UserStopException:
            if 'cur_positive_cond' in vars(): del cur_positive_cond, cur_negative_cond
            if 'refiner_positive' in vars(): del refiner_positive, refiner_negative
        finally:
            model_loader.free_memory(1024 ** 4, devices.get_torch_device())

        if task.get("stop", False):
            break

    del ctrls, unet_model
    left_vae(sampled_latents, True)
    thread_vae.join()

    model_loader.free_memory(1024 ** 4, devices.get_torch_device())
    model_loader.current_loaded_models.clear()