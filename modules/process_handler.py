import copy
import json
import math
import os
import random
import re
import threading
import time
import cv2
import einops
import numpy as np
import torch
from PIL import Image

import modules.options as opts
import modules.paths
from modules import clip_helper, civitai, core, controlnet_helper, devices, latent_interposer, prompt_helper, shared, util, vae_helper, upscaler_esrgan, gfpgan_model
from modules.model.model_base import BaseModel, SDXL, SDXLRefiner
from modules.model import model_loader, model_helper, sample, samplers
from modules.controlnet.processor import Processor as controlnet_processor

def handler(task):
    style = task.get("style", None)
    batch_size = task.get("batch", 1)
    round_batch_size = min(8, batch_size)

    seed = int(task.get("seed", 0))
    max_seed = 2 ** 63 - 1
    random.seed()
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

    positive, negative, style, loras = prompt_helper.generate(task.get("main_character", ("", "")), task.get("prompt_main", ""), task.get("prompt_temp", ""), task.get("prompt_negative", ""),
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
    clip_skip = -1 * abs(int(task.get("clip_skip") or 1))

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
        "controlnet": "loading controlnet ...",
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

class UserStopException(Exception):
    pass

@torch.no_grad()
@torch.inference_mode()
def process_diffusion(task, base_path, refiner_path, positive, negative, steps, skip_step, size, seeds, subseeds,
        callback, sampler_name, scheduler_name,
        latent=None, image=None, denoise=1.0, noise_scale=1.0, cfg_scale=7.0, cfg_scale_to=7.0, batch_size=1, loras=[], controlnets=[],
        tiled=False, round_batch_size=8, subseed_strength=1.0, clip_skip=0):

    devices.torch_gc()
    progress_output(task, "load_model")
    
    xl_base, xl_base_patched, xl_refiner = core.get_sd_model(base_path, refiner_path, steps, loras)
    model_type, refiner_model_type = core.assert_model_integrity(xl_base, xl_refiner)

    progress_output(task, "clip")

    clip_model = xl_base_patched.clip
    clip_model.clip_layer(clip_skip)
    if xl_refiner is not None:
        clip_refiner_model = xl_refiner.clip
        clip_refiner_model.clip_layer(clip_skip)
    else:
        clip_refiner_model = None
    positive_template = positive.pop(0)
    negative_template = negative.pop(0)
    positive_cond = []
    negative_cond = []
    positive_refiner_cond = []
    negative_refiner_cond = []
    def clip_worker(positive, negative):
        for x, y in zip(positive, negative):
            positive_cond.append(clip_helper.clip_encode(clip_model, x, model_type))
            negative_cond.append(clip_helper.clip_encode(clip_model, y, model_type))
            if clip_refiner_model is not None:
                positive_refiner_cond.append(clip_helper.clip_encode(clip_refiner_model, x, refiner_model_type))
                negative_refiner_cond.append(clip_helper.clip_encode(clip_refiner_model, y, refiner_model_type))
    
    thread_clip = threading.Thread(target=clip_worker, args=(positive, negative)).start()

    update_model_info_base(base_path, xl_base)

    step_base, step_refiner = steps
    step_total = step_base + step_refiner

    unet_model = xl_base_patched.unet
    refiner_unet_model = xl_refiner.unet if xl_refiner is not None else None
    
    vae_model = xl_refiner.vae if xl_refiner is not None and xl_refiner.vae is not None else xl_base_patched.vae

    width, height = size
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
                    
                    image_pixel = cv2.cvtColor(image_pixel, cv2.COLOR_BGR2RGB)

                    if width * height < 1024 ** 2:
                        upscaler = upscaler_esrgan.UpscalerESRGAN("4x-UltraSharp.pth")
                        image_pixel = upscaler(image_pixel)

                    image_pixel = util.resize_image(image_pixel, w, h, resize_mode=0)

                    if util.get_faces(image_pixel)[1] is not None:
                        gfpgan = gfpgan_model.GFPGan()
                        image_pixel = gfpgan(image_pixel, only_center_face=False)

                    image_pixel = cv2.cvtColor(image_pixel, cv2.COLOR_BGR2RGB)
                
            if task.get("mask_use_lama", False):
                lama_inpaint = controlnet_processor("lama_inpaint")
                image_pixel = lama_inpaint(image_pixel, image_mask)
                
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
            latent = core.generate_empty_latent(width=width, height=height, batch_size=1)
        else:
            image_pixel_torch = util.numpy_to_pytorch(image_pixel)
            image_mask_torch = util.numpy_to_pytorch(image_mask)
            latent = vae_helper.encode_vae_mask(vae_model, image_pixel_torch, image_mask_torch, tiled=tiled)

    if controlnets:
        progress_output(task, "controlnet")
        ctrls, unet_model = controlnet_helper.processor(controlnets, unet_model, width, height, image_mask)
    else:
        ctrls = []
    
    sampled_latents = []
    vae_worker_running = True

    def left_vae(sampled_latents, set_None=False):
        left_sampled_latents = sampled_latents[:]
        sampled_latents.clear()
        if set_None:
            sampled_latents.append(None)

        while left_sampled_latents:
            progress_output(task, "vae")
            sampled_latent, cur_batch, cur_seed, cur_subseed, filename = left_sampled_latents.pop(0)
            core.vae_sampled(sampled_latent, vae_model, tiled, task, cur_batch, cur_seed, cur_subseed, filename, image_pixel, image_mask, image_pixel_orgin, re_zoom_point)
            print("cuda", len(left_sampled_latents))
    
    # def vae_worker():
    #     while True and None not in sampled_latents:
    #         if sampled_latents and vae_worker_running:
    #             progress_output(task, "vae")
    #             sampled_latent, cur_batch, cur_seed, subseeds, filename = sampled_latents.pop(0)
    #             core.vae_sampled(sampled_latent, vae_model, tiled, task, cur_batch, cur_seed, subseeds, filename, image_pixel, image_mask, image_pixel_orgin, re_zoom_point)
    #             if sampled_latents is not None:
    #                 print("cpu", len(sampled_latents))
    #             else:
    #                 print("cpu", "last")
    #         time.sleep(0.01)
    
    # thread_vae = threading.Thread(target=vae_worker)
    # thread_vae.start()

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
        alpha = cfg_scale_to + (cond_scale - cfg_scale_to) * timestep[0] / 15
        return uncond + (cond - uncond) * alpha
    
    def model_function_wrapper(func, args):
        cond_or_uncond = args.pop("cond_or_uncond")
        return func(args.pop("input"), args.pop("timestep"), **args.pop("c"))

    unet_model.model_options["sampler_cfg_function"] = sampler_cfg_function
    unet_model.model_options["model_function_wrapper"] = model_function_wrapper
    
    latent_image = latent["samples"]
    latent_mask = latent.get("noise_mask", None)

    # latent_convert = None
    # if model_type != refiner_model_type and step_refiner > 0:
    #     latent_convert = latent_interposer.LatentInterposer(model_type, refiner_model_type)

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
                    # steps=step_total, start_step=start_step, last_step=step_base,
                    steps=step_base, start_step=start_step, last_step=step_base,
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

            # if refiner_unet_model is not None and step_refiner > 0 and (model_type != "sd15" or refiner_model_type == "sd15"):
            if refiner_unet_model is not None and step_refiner > 0:
                # if latent_convert is not None:
                if model_type != refiner_model_type and step_refiner > 0:
                    # sampled_latent = latent_convert(sampled_latent)
                    sampled_latent = vae_helper.decode_vae(xl_base.vae, sampled_latent)
                    sampled_latent = vae_helper.encode_vae(xl_refiner.vae, sampled_latent)["samples"]

                if latent_mask is not None:
                    pre_mask = sample.prepare_mask(latent_mask, sampled_latent.shape, torch.device("cpu"))
                    sampled_latent = sampled_latent * pre_mask + latent_image * (1 - pre_mask)
                    del pre_mask

                sigmas = None

                # refiner_positive = clip_helper.clip_separate(cur_positive_cond, target_model=refiner_unet_model.model, target_clip=clip_model)
                del cur_positive_cond
                # refiner_negative = clip_helper.clip_separate(cur_negative_cond, target_model=refiner_unet_model.model, target_clip=clip_model)
                del cur_negative_cond
                # refiner_positive, refiner_negative = controlnet_helper.apply_controlnets(positive_refiner_cond.pop(0), negative_refiner_cond.pop(0), ctrls)
                refiner_positive, refiner_negative = positive_refiner_cond.pop(0), negative_refiner_cond.pop(0)
                refiner_positive = clip_helper.clip_separate(refiner_positive, target_model=refiner_unet_model.model, target_clip=clip_refiner_model)
                refiner_negative = clip_helper.clip_separate(refiner_negative, target_model=refiner_unet_model.model, target_clip=clip_refiner_model)

                # model_loader.free_memory(1024 ** 4, devices.get_torch_device())

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
            
            # model_loader.free_memory(1024 ** 4, devices.get_torch_device())

            pool_low_limit = 4
            if batch_size > round_batch_size and len(positive_cond) < pool_low_limit:
                positive_new, negative_new = prompt_helper.re_generate(positive_template, negative_template, round_batch_size - pool_low_limit)
                threading.Thread(target=clip_worker, args=(positive_new, negative_new)).start()
                positive += positive_new
                negative += negative_new
                seeds += [seeds[-1] + (0 if subseed_strength < 1 else i + 1) for i in range(round_batch_size - pool_low_limit)]
                subseeds += [subseeds[-1] + i + 1 for i in range(round_batch_size - pool_low_limit)]

            file_format = task["file_format"] or "jpeg"
            filepath = os.path.join(modules.paths.temp_outputs_path, f"{time.strftime('%Y%m')}")
            if not os.path.exists(filepath):
                os.mkdir(filepath)
            filename = f"{filepath}/{time.strftime('%Y%m%d%H%M%S')}_{i}.{file_format}"
            filename = os.path.join(filepath, filename)
            util.save_image_with_geninfo(Image.open(os.path.join(f"{modules.paths.temp_outputs_path}/temp", "generate.png")), "pre_vae: True", filename, quality=60)

            # if i > batch_size - 3:
            #     vae_worker_running = False

            sampled_latents.append((sampled_latent, i, cur_seed, cur_subseed, filename))
            if len(sampled_latents) >= 2:
                left_vae(sampled_latents)

        except UserStopException:
            if 'cur_positive_cond' in vars(): del cur_positive_cond, cur_negative_cond
            if 'refiner_positive' in vars(): del refiner_positive, refiner_negative
            model_loader.free_memory(1024 ** 4, devices.get_torch_device())            

        if task.get("stop", False):
            break

    del ctrls, unet_model
    left_vae(sampled_latents, True)
    # thread_vae.join()

    model_loader.free_memory(1024 ** 4, devices.get_torch_device())
    model_loader.current_loaded_models.clear()