import math
import os
import random
import re
import cv2
import numpy as np
from insightface.app import FaceAnalysis

import modules.paths
import modules.options as opts
from modules import lora, prompt_helper, util, wd14tagger

def UseControlnet(ref_type, ref_image, weight, ref_mode=None, model=None, start_percent=0, end_percent=0.5):
    ref_mode = ref_mode or opts.options["ref_mode"][ref_type][0]
    return [ref_type, ref_mode, ref_image, weight, model, start_percent, end_percent]

def VarySubtle(ref_image, params, pnginfo, gen_opts, weight=0.5, skip_step=0.2):
    # subseed_strength = 0.05
    # weight = max(0.2, weight)
    params["action"] = "generate"
    params["seed"] = 0
    params["skip_step"] = skip_step
    params["aspect_ratios"] = gen_opts.get("ratios")
    # params["fixed_seed"] = True
    # params["subseed"] = random.randint(1, 1024 ** 3)
    # params["subseed_strength"] = subseed_strength

    # ref_mode, _ = opts.options["ref_mode"]["Ref All"]
    if weight > 0:
        params["controlnet"].append(UseControlnet("Ref All", ref_image, weight))

    params["image"] = (ref_image, None)

    return params

def VaryStrong(ref_image, params, pnginfo, gen_opts):
    weight = round(random.uniform(0.3, 0.5), 2)

    params["action"] = "generate"
    params["aspect_ratios"] = gen_opts.get("ratios")
    params["seed"] = 0
    params["skip_step"] = 0.2
    # params["fixed_seed"] = False

    # ref_mode, _ = opts.options["ref_mode"]["Ref All"]
    params["controlnet"].append(UseControlnet("Ref All", ref_image, weight))

    params["image"] = (ref_image, None)
    
    return params

def VaryCustom(ref_image, params, pnginfo, gen_opts, img_vary_editor):
    prompt = gen_opts.get("vary_prompt", "")
    if prompt:
        params["prompt_main"] = prompt +","+ params["prompt_main"]

    vary_strength = round(gen_opts.get("vary_custom_strength", 0.4), 2)
    weight = 1 - vary_strength
    skip_step = max(0, round(0.5 - vary_strength / 2, 2))
    params = VarySubtle(ref_image, params, pnginfo, gen_opts, weight=weight, skip_step=skip_step)
    ref_image = img_vary_editor["image"]
    mask = img_vary_editor["mask"]
    mask = mask if (mask > 0).any() else None

    if mask is not None:
        if not gen_opts.get("vary_custom_area", True):
            mask = 255 - mask
        mask = util.blur(mask, 64)
        mask[mask > 32] = 255
        mask = util.blur(mask, 32)
        params["mask_use_lama"] = gen_opts.get("mask_use_lama", False)
        util.save_temp_image(mask, "vary_custom_mask.png")
        if " (sd15)" in params.get("base_name", ""):
            # ref_mode = random.choices(["default", "lama_inpaint"])[0]
            # ref_mode = "lama_inpaint" if weight < 0.5 else "default"
            ref_mode = "default"
            params["controlnet"] = [UseControlnet("Others", ref_image, weight, ref_mode=ref_mode, model="control_v11p_sd15_inpaint")]
    util.save_temp_image(ref_image, "vary_custom_image.png")
    
    params["image"] = (ref_image, mask)
    params["size"] = (ref_image.shape[1], ref_image.shape[0])

    return params

def image_pad(orgin_image, pads, mask_pad=32, blur_alpha=1.0):
    top, bottom, left, right = pads
    rect_top = mask_pad if top > 0 else 0
    rect_bottom = orgin_image.shape[0] - mask_pad if bottom > 0 else orgin_image.shape[0]
    rect_left = mask_pad if left > 0 else 0
    rect_right = orgin_image.shape[1] - mask_pad if left > 0 else orgin_image.shape[1]
    ref_image = orgin_image[rect_top:rect_bottom, rect_left:rect_right]
    top, bottom, left, right = [x + (mask_pad if x > 0 else 0) for x in [top, bottom, left, right]]
    # width_mask, height_mask = ref_image.shape[1], ref_image.shape[0]

    pad_image = ref_image.copy()
    pad_step = 128
    step = 0

    mask_top, mask_bottom, mask_left, mask_right = [ min(x, mask_pad * 2) for x in [top, bottom, left, right] ]
    pad_mark = np.ones((pad_image.shape[0] - mask_top - mask_bottom, pad_image.shape[1] - mask_left - mask_right, 3), dtype=np.uint8)
    pad_mark = np.pad(pad_mark, [[mask_top, mask_bottom], [mask_left, mask_right], [0, 0]], mode="constant", constant_values=255)
    ref_mask = np.pad(pad_mark, [[top, bottom], [left, right], [0, 0]], mode="constant", constant_values=255)
    ref_mask = util.blur(ref_mask, mask_pad)

    while True:
        pad_top, pad_bottom, pad_left, pad_right = [ max(0, min(pad_step, x - pad_step * step)) for x in [top, bottom, left, right] ]
        if pad_top + pad_bottom + pad_left + pad_right == 0:
            break
        
        pad_mark[:, :, :] = 0
        pad_mark = np.pad(pad_mark, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode="constant", constant_values=255)
        pad_mark = util.blur(pad_mark, pad_step // 4)
        # util.save_temp_image(pad_mark, f"pad_mark_{step}.png")
        cur_pad_mark = pad_mark[:,:,0].astype(np.float32) / 255 * (1 - blur_alpha)
        cur_pad_mark = cur_pad_mark.reshape(cur_pad_mark.shape[0], cur_pad_mark.shape[1], 1)
        
        pad_image = np.pad(pad_image, [[0, 0], [pad_left, pad_right], [0, 0]], mode="edge")
        blur_pad_image = util.resize_image(pad_image, pad_image.shape[1], pad_image.shape[0])

        pad_image = np.pad(pad_image, [[pad_top, pad_bottom], [0, 0], [0, 0]], mode="edge")
        blur_pad_image = np.pad(blur_pad_image, [[pad_top, pad_bottom], [0, 0], [0, 0]], mode="edge")

        if step > 0:
            blur_pad_image = util.shuffle(blur_pad_image, f=1024)
        
        blur_pad_image = util.blur(blur_pad_image, step * pad_step // 8)

        pad_image = (pad_image * (1 - cur_pad_mark) + blur_pad_image * cur_pad_mark).astype(np.uint8)
        # util.save_temp_image(pad_image, f"pad_image_{step}.png")

        step += 1

    util.save_temp_image(pad_image, "pad_image.png")

    return pad_image, ref_mask

def ZoomOut(ref_image, params, pnginfo, gen_opts, zoom = 1.5):
    base_step = 25
    mask_pad = 32
    max_pixel = 1280 ** 2
    blur_alpha = gen_opts.get("zoom_blur_alpha", 0.5)
    orgin_image = ref_image.copy()

    if isinstance(zoom, list):
        top, bottom, left, right = zoom
        height, width = ref_image.shape[:2]
        height += top + bottom
        width += left + right
        pads = [top, bottom, left, right]
    else:
        # zoom = util.diagonal_fov(24) / util.diagonal_fov(24 * focal_zoom)
        zoom = math.sqrt(zoom)
        height_zoom, width_zoom = ref_image.shape[:2]
        height_zoom, width_zoom = int(height_zoom / zoom), int(width_zoom / zoom)

        width, height = int(width_zoom * zoom), int(height_zoom * zoom)
        orgin_image = util.resize_image(orgin_image, width_zoom, height_zoom)
        width_mask, height_mask = orgin_image.shape[1], orgin_image.shape[0]
        width_pad, height_pad = width - width_mask, height - height_mask

        pads = [height_pad // 2, (height_pad - height_pad // 2), width_pad // 2, (width_pad - width_pad // 2)]

    ref_image, ref_mask = image_pad(ref_image, pads, mask_pad, blur_alpha=blur_alpha)
    height, width = ref_image.shape[:2]

    re_zoom = math.sqrt(width * height / max_pixel)
    if re_zoom > 1:
        width, height = int(width / re_zoom), int(height / re_zoom)
        ref_image = util.resize_image(ref_image, width, height)
        ref_mask = util.resize_image(ref_mask, width, height)

    style = params.get("style", "")
    params["action"] = "generate"
    params["prompt_negative"] = params.get("prompt_negative", "") or gen_opts.get("prompt_negative")
    params["seed"] = 0
    params["denoise"] = gen_opts.get("zoom_denoise", 0.95)
    # params["skip_step"] = 0.1

    prompt = gen_opts.get("zoom_prompt", "")
    if prompt:
        params["prompt_temp"] = prompt + ",".join(wd14tagger.tag(ref_image)[:12])

    if style != "":
        params["sample"] = "dpmpp_3m_sde_gpu"
        params["controlnet"].append(UseControlnet("Ref Depth", ref_image, 0.6))
    else:
        params["sample"] = "dpmpp_2m_sde_gpu"
        params["controlnet"].append(UseControlnet("Ref All", ref_image, 0.7))
    
    params["image"] = (ref_image, ref_mask)
    params["size"] = (ref_image.shape[1], ref_image.shape[0])
    params["steps"] = (base_step, int(base_step // 2.5))

    if " (sd15)" in params.get("base_name", ""):
        params["sample"] = "dpmpp_2m"
        params["controlnet"] = [UseControlnet("Others", ref_image, 0.8, ref_mode="default", model="control_v11p_sd15_inpaint", end_percent=0.3)]
        params["controlnet"].append(UseControlnet("Ref All", orgin_image, 0.5))
        params["steps"] = (base_step, 0)
        # params["denoise"] = 1.0

    util.save_temp_image(ref_image, "zoom.png")
    util.save_temp_image(ref_mask, "zoom_mask.png")

    return params

def ZoomOut20(ref_image, params, pnginfo, gen_opts):
    return ZoomOut(ref_image, params, pnginfo, gen_opts, zoom=2.0)

def ZoomOutCustom(ref_image, params, pnginfo, gen_opts):
    zoom = gen_opts.get("zoom_custom", 2.0)
    return ZoomOut(ref_image, params, pnginfo, gen_opts, zoom=zoom)

def ChangeStyle(ref_image, params, pnginfo, gen_opts, gl_style_list, num_selected_style):
    seed = random.randint(1, 1024 ** 3)
    style = gl_style_list[int(num_selected_style)][1]
    if style == "random-style":
        style = f"__style_{gen_opts.get('style')}__"

    simple_styles = [   "sai-line art", "sai-origami", "Pencil Sketch Drawing",
                        "papercraft-collage", "papercraft-flat papercut", "papercraft-kirigami", "papercraft-paper mache", "papercraft-paper quilling", "papercraft-papercut collage",
                        "papercraft-papercut shadow box", "papercraft-stacked papercut", "papercraft-thick layered papercut",
                        "photo-film noir", "photo-silhouette"]

    main_character = pnginfo.get("main character", None)
    orgin_style = params.get("style", "")
    prompt_main = ",".join(wd14tagger.tag(ref_image)[:12])
    mc_prompt = ""
    if main_character is not None:
        mc_prompt = prompt_helper.dynamic_prompt(main_character, seeds=[params.get("seed", seed)], lang=gen_opts.get("lang", None))[0][0]
        mc_prompt = " ".join(mc_prompt.split(" ")[1:]).strip("()[]\"'")
        # prompt_main += f",{mc_prompt}"
    
    if style in simple_styles:
        re_colors = r"(light|dark)?(aqua|black|blue|brown|fuchsia|gray|green|lime|maroon|navy|olive|orange|purple|red|silver|teal|white|yellow|makeup)"
        prompt_main = re.sub(re_colors, "", prompt_main, flags=re.IGNORECASE)
        prompt_main = f"({prompt_main}:0.8)"
    else:
        prompt_main = f"({prompt_main}:1.1)"

    height, width = ref_image.shape[:2]
        
    params["action"] = "generate"
    params["prompt_main"] = prompt_main
    params["main_character"] = ("", mc_prompt)
    params["prompt_negative"] = gen_opts.get("prompt_negative", "")
    params["style"] = style
    params["style_weight"] = 1.1
    params["skip_prompt"] = False
    params["sample"] = "dpmpp_3m_sde_gpu"
    params["seed"] = seed
    params["size"] = (width, height)

    if style not in orgin_style:
        if orgin_style not in simple_styles:
            params["controlnet"].append(UseControlnet("Ref Depth", ref_image, random.randint(50, 80) / 100))

            if random.randint(0, 100) > 30 and style not in simple_styles: # random use img2img
                params["image"] = (ref_image, None)
                params["denoise"] = 0.95
        else:
            params["prompt_main"] = re.sub(r"((simple|grey|white) background|spot color)", "", params["prompt_main"], flags=re.IGNORECASE)
            if random.randint(0, 100) > 50:
                params["controlnet"].append(UseControlnet("Ref Stuct", ref_image, 0.4))
            else:
                params["controlnet"].append(UseControlnet("Ref Depth", ref_image, 0.7))
    else:
        params["image"] = (ref_image, None)
        params["denoise"] = 0.70
    
    return params

def Resize(ref_image, params, pnginfo, gen_opts):
    ratios_width, ratios_height = gen_opts.pop("resize_ratios")
    ratios_target = ratios_width / ratios_height
    height_orgin, width_orgin = ref_image.shape[:2]
    ratios_orgin = width_orgin / height_orgin

    width, height = width_orgin, height_orgin

    if ratios_target > ratios_orgin:
        width = int(width_orgin * (ratios_target / ratios_orgin))
    elif ratios_target < ratios_orgin:
        height = int(height_orgin * (ratios_orgin / ratios_target))
    
    width_pad, height_pad = width - width_orgin, height - height_orgin
    pads = [height_pad // 4, (height_pad - height_pad // 4), width_pad // 2, (width_pad - width_pad // 2)]

    params = ZoomOut(ref_image, params, pnginfo, gen_opts, zoom=pads)
    params["aspect_ratios"] = [ratios_width, ratios_height]

    # params["denoise"] = gen_opts.get("zoom_denoise", 1.0)

    return params


def Refiner(ref_image, params, pnginfo, gen_opts, mask=None, type="image"):
    # blurred = cv2.GaussianBlur(ref_image, (5, 5), 2)
    # ref_image = cv2.addWeighted(ref_image, 1.5, blurred, -0.5, 0)
    height, width = ref_image.shape[:2]
    max_pixel = 1344 ** 2
    re_zoom = math.sqrt(width * height / max_pixel)
    if re_zoom < 1:
        width, height = round(width / re_zoom / 8) * 8, round(height / re_zoom / 8) * 8
        ref_image = util.resize_image(ref_image, width, height)
    util.save_temp_image(ref_image, f"refiner_{type}.png")

    # step_base = max(20, pnginfo.get("all_steps", (20, 0))[0])
    step_base = 25

    params["action"] = "generate"
    # params["batch"] = gen_opts.pop("pic_num")
    params["prompt_main"] = lora.remove_prompt_lora(params["prompt_main"], ["add-detail-xl", "add_detail"])[0]
    
    denoise = gen_opts.get(f"refiner_{type}_denoise", 0.4) * 0.8
    params["steps"] = (step_base, max(0, int(denoise * 10)))
    params["denoise"] = denoise
    params["skip_step"] = int(step_base * 0.1)
    
    params["seed"] = 0
    params["sample"] = "dpmpp_2m"
    params["clip_skip"] = 1
    params["refiner_name"] = gen_opts.get("refiner_model", params.get("refiner_name", ""))
    
    detail = gen_opts.get(f"refiner_{type}_detail", 0.3)
    if detail != 0:
        if " (sd15)" in params.get("base_name", ""):
            params["lora"] = [("add-detail.safetensors", detail, "")]
            params["controlnet"].append(UseControlnet("Others", ref_image, detail, ref_mode="default", model="control_v11f1e_sd15_tile", end_percent=0.8))
        else:
            params["lora"] = [("add-detail-xl.safetensors", detail, "")]

    params["prompt_temp"] = gen_opts.get("refiner_prompt", "") + "(UHD,8K,ultra detailed)"
    params["prompt_negative"] = params.get("prompt_negative", "") or "ugly,deformed,noisy,blurry,NSFW"

    if mask is None:
        params["prompt_temp"] += "," + ",".join(wd14tagger.tag(ref_image)[:12])

        app = FaceAnalysis(root=modules.paths.face_models_path)
        app.prepare(ctx_id=0, det_size=(640, 640))
        faces = app.get(ref_image)
        mask = None

        if faces:
            min_face_area = 768 ** 2
            landmarks = [face.landmark_2d_106 for face in faces if (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]) < min_face_area]
            if landmarks:
                mask = util.face_mask(ref_image, landmarks)
                mask = 255 - mask
    
    params["image"] = (ref_image, mask)
    params["size"] = (width, height)

    # params["controlnet"].append(UseControlnet("Ref Stuct", ref_image, 0.75))

    return params

def RefinerFace(ref_image, params, pnginfo, gen_opts):
    app = FaceAnalysis(root=modules.paths.face_models_path)
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = app.get(ref_image)
    faces = sorted(faces, key = lambda x : ((x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1])), reverse=True)

    if faces:
        face_index = gen_opts.get(f"refiner_face_index", 0)
        if face_index >= len(faces):
            return None
        face = faces[face_index]
        landmarks = [face.landmark_2d_106]
        ref_mask = util.face_mask(ref_image, landmarks)
        face_box = [int(x) for x in face.bbox]
        face_image = ref_image[face_box[1]:face_box[3], face_box[0]:face_box[2]]
        age = face.age
        sex = face.sex # M, F
        # face_mask = ref_mask[:,:,0].astype(np.float32) / 255
        # face_mask = face_mask.reshape(ref_image.shape[0], ref_image.shape[1], 1)
        # ref_image = ref_image * (1 - face_mask) + util.blur(ref_image, 20) * face_mask
        # ref_image = ref_image.astype(np.uint8)
        # ref_image = util.blur(ref_image, 2)
        # gen_opts["detail"] = 0.95

        params = Refiner(ref_image, params, pnginfo, gen_opts, mask=ref_mask, type="face")
        params["prompt_temp"] = gen_opts.get("refiner_prompt", "") + "(UHD,8K,detail face,detailed skin,perfect eyes:1.2),{lang}" + f",{age}yo"
        params["prompt_temp"] += "," + ",".join(wd14tagger.tag(face_image)[:12])

        if params["denoise"] > 0.5 and " (sd15)" not in params.get("base_name", ""):
            params["controlnet"] = [UseControlnet("Ref Depth", None, 0.6)]
            # params["controlnet"] = [UseControlnet("Ref Pose", None, 0.6, ref_mode="dwpose_face")]
        
        return params
    
def ReFace(ref_image, params, pnginfo, gen_opts, face_image):
    if face_image is not None:
        params = {}
        params["action"] = "reface"
        params["image"] = np.array(ref_image)
        params["face"] = np.array(face_image)

        pnginfo.pop("all_steps")
        params["pnginfo"] = pnginfo

        return params

def Upscale(ref_image, params, pnginfo, gen_opts):
    params = {
        "model": gen_opts.get("upscale_model"),
        "factor": gen_opts.get("upscale_factor", 2.0),
        "repair_face": gen_opts.get("upscale_repair_face", True),
        "origin_visibility": gen_opts.get("upscale_origin", 0)
    }
    params["action"] = "upscale"
    params["file_format"] = gen_opts.get("file_format")
    params["image"] = np.array(ref_image)
    params["pnginfo"] = pnginfo

    return params
        