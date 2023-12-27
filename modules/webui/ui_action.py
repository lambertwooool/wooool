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

def UseControlnet(ref_type, ref_image, weight):
    ref_mode, _ = opts.options["ref_mode"][ref_type]
    return [ref_type, ref_mode, ref_image, weight, None]

def VarySubtle(ref_image, params, pnginfo, gen_opts, weight=0.6, skip_step=0.3):
    # subseed_strength = 0.05

    weight = max(0.3, weight)
    params["action"] = "generate"
    params["seed"] = 0
    params["skip_step"] = skip_step
    params["aspect_ratios"] = gen_opts.get("ratios")
    # params["fixed_seed"] = True
    # params["subseed"] = random.randint(1, 1024 ** 3)
    # params["subseed_strength"] = subseed_strength

    # ref_mode, _ = opts.options["ref_mode"]["Ref All"]
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
        mask = util.blur(mask, 16)
        util.save_temp_image(mask, "vary_custom_mask.png")
    util.save_temp_image(ref_image, "vary_custom_image.png")
    
    params["image"] = (ref_image, mask)

    return params

def image_pad(image, top=0, bottom=0, left=0, right=0):
    pad_image = image.copy()
    pad_step = 128
    blur_scale = 8
    step = 1

    while True:
        pad_top, pad_bottom, pad_left, pad_right = [ max(0, min(pad_step, x - pad_step * step)) for x in [top, bottom, left, right] ]
        step += 1
        if pad_top + pad_bottom + pad_left + pad_right == 0:
            break
        
        pad_mark = np.ones((pad_image.shape[1], pad_image.shape[0], 3), dtype=np.uint8)
        pad_mark = np.pad(pad_mark, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode="constant", constant_values=255)
        pad_mark = util.blur(pad_mark, step * blur_scale)
        util.save_temp_image(pad_mark, f"pad_mark_{step}.png")
        pad_mark = pad_mark[:,:,0].astype(np.float32) / 255
        pad_mark = pad_mark.reshape(pad_mark.shape[0], pad_mark.shape[1], 1)
        
        pad_image = np.pad(pad_image, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode="edge")
        blur_pad_image = util.blur(pad_image, step * blur_scale)

        pad_image = (pad_image * (1 - pad_mark) + blur_pad_image * pad_mark).astype(np.uint8)
        util.save_temp_image(pad_image, f"pad_image_{step}.png")


    util.save_temp_image(pad_image, "pad_image.png")

    return pad_image

def ZoomOut(ref_image, params, pnginfo, gen_opts, zoom = 1.5):
    subseed_strength = 1.0
    # zoom = util.diagonal_fov(24) / util.diagonal_fov(24 * focal_zoom)
    zoom = math.sqrt(zoom)
    base_step = 25

    orgin_image = ref_image.copy()
    mask_pad = 32
    blur_weight = 12
    # min_pixel = 1024 ** 2
    max_pixel = 1280 ** 2

    height_zoom, width_zoom = ref_image.shape[:2]
    height_zoom, width_zoom = int(height_zoom / zoom), int(width_zoom / zoom)
    # ratio = util.aspect2ratio(width_zoom, height_zoom)
    # re_zoom = math.sqrt(width_zoom * height_zoom / min_pixel)
    # if re_zoom < 1:
    #     # width_zoom, height_zoom = util.ratios2size(ratio, min_pixel)
    #     width_zoom, height_zoom = width_zoom / re_zoom, height_zoom / re_zoom
    width, height = int(width_zoom * zoom), int(height_zoom * zoom)
    re_zoom = math.sqrt(width * height / max_pixel)
    if re_zoom > 1:
        width_zoom, height_zoom = int(width_zoom * re_zoom), int(height_zoom * re_zoom)
        # width, height = util.ratios2size(ratio, max_pixel)
        # width_zoom, height_zoom = util.ratios2size(ratio, max_pixel / zoom)
    orgin_image = util.resize_image(orgin_image, width_zoom, height_zoom)
    ref_image = orgin_image[mask_pad:(orgin_image.shape[0] - mask_pad), mask_pad:(orgin_image.shape[1] - mask_pad)]
    width_mask, height_mask = ref_image.shape[1], ref_image.shape[0]

    # image_pad(ref_image, height // 2, height // 2, width // 2, width // 2)

    ref_image = np.pad(ref_image, [[(height - height_mask) // 2, 0], [0, 0], [0, 0]], mode="edge")
    ref_image = np.pad(ref_image, [[0, 0], [(width - width_mask) // 2] * 2, [0, 0]], mode="edge") # random.choice(["edge", "mean"])
    
    ref_mask = np.ones((height_mask, width_mask, 3), dtype=np.uint8)
    ref_mask = np.pad(ref_mask, [[(height - height_mask) // 2, 0], [(width - width_mask) // 2] * 2, [0, 0]], mode="constant", constant_values=255)
    # ref_image = ref_image[0:ref_image.shape[0] - mask_pad // 2, 0:ref_image.shape[1]]
    # ref_mask = np.pad(ref_mask, [[0, (ref_image.shape[0] - ref_mask.shape[0])], [0, 0], [0, 0]], mode="constant", constant_values=255)
    if ref_image.shape[0] > ref_mask.shape[0]:
        ref_mask = np.pad(ref_mask, [[0, (ref_image.shape[0] - ref_mask.shape[0])], [0, 0], [0, 0]], mode="edge")

    ref_image[ref_mask > 127] = util.blur(ref_image, blur_weight)[ref_mask > 127]

    ref_image = np.pad(ref_image, [[0, height - ref_image.shape[0]], [0, 0], [0, 0]], mode="edge")
    ref_mask = np.pad(ref_mask, [[0, height - ref_mask.shape[0]], [0, 0], [0, 0]], mode="constant", constant_values=255)
    ref_mask = util.blur(ref_mask, mask_pad)

    style = params.get("style", "")
    params["action"] = "generate"
    params["sample"] = "dpmpp_3m_sde_gpu"
    params["prompt_main"] = params.get("prompt_main", "") or ",".join(wd14tagger.tag(orgin_image)[:12])
    params["prompt_negative"] = params.get("prompt_negative", "") or gen_opts.get("prompt_negative")
    if style != "":
        # params["fixed_seed"] = True
        params["seed"] = 0
        # params["subseed"] = random.randint(1, 1024 ** 3)
        # params["subseed_strength"] = subseed_strength
        # params["denoise"] = int(sl_denoise) / 100
        params["denoise"] = 0.9
        params["skip_step"] = 0.1

        # ref_mode, _ = opts.options["ref_mode"]["Ref Depth"]
        # params["controlnet"].append(["Ref Depth", ref_mode, ref_image, 0.6])
        params["controlnet"].append(UseControlnet("Ref Depth", ref_image, 0.6))
    else:
        params["skip_step"] = 0.1
        
        # ref_mode, _ = opts.options["ref_mode"]["Ref All"]
        # params["controlnet"].append(["Ref All", ref_mode, orgin_image, 0.9])
        params["controlnet"].append(UseControlnet("Ref All", orgin_image, 0.9))
    
    params["image"] = (ref_image, ref_mask)
    params["size"] = (width, height)
    params["steps"] = (base_step, int(base_step // 2.5))

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
            # ref_mode, _ = opts.options["ref_mode"]["Ref Depth"]
            # params["controlnet"].append(["Ref Depth", ref_mode, np.array(ref_image), random.randint(50, 80) / 100])
            params["controlnet"].append(UseControlnet("Ref Depth", ref_image, random.randint(50, 80) / 100))

            if random.randint(0, 100) > 30 and style not in simple_styles: # random use img2img
                params["image"] = (ref_image, None)
                params["denoise"] = 0.95
        else:
            params["prompt_main"] = re.sub(r"((simple|grey|white) background|spot color)", "", params["prompt_main"], flags=re.IGNORECASE)
            if random.randint(0, 100) > 50:
                # ref_stuct, _ = opts.options["ref_mode"]["Ref Stuct"]
                # params["controlnet"].append(["Ref Stuct", ref_stuct, np.array(ref_image), 0.4])
                params["controlnet"].append(UseControlnet("Ref Stuct", ref_image, 0.4))
            else:
                # ref_mode, _ = opts.options["ref_mode"]["Ref Depth"]
                # params["controlnet"].append(["Ref Depth", ref_mode, np.array(ref_image), 0.7])
                params["controlnet"].append(UseControlnet("Ref Depth", ref_image, 0.7))
    else:
        params["image"] = (ref_image, None)
        params["denoise"] = 0.70

    return params

def Resize(ref_image, params, pnginfo, gen_opts):
    params["action"] = "generate"
    # params["batch"] = gen_opts.pop("pic_num")
    # params["quality"] = gen_opts.pop("quality")
    params["cfg_scale"] = gen_opts.pop("cfg_scale")
    params["sample"] = "dpmpp_3m_sde_gpu"
    params["seed"] = 0

    ratios_width, ratios_height = gen_opts.pop("resize_ratios")
    # ratios_width, ratios_height = opts.options["ratios"][gen_opts.pop("resize_ratios")]
    ratios_target = ratios_width / ratios_height
    height_orgin, width_orgin = ref_image.shape[:2]
    ratios_orgin = width_orgin / height_orgin

    width, height = width_orgin, height_orgin
    ref_mask = np.ones((height, width, 3), dtype=np.uint8)
    max_pixel = 1280 ** 2
    blur_weight = 8
    base_step = 25
    orgin_image = ref_image.copy()

    if ratios_target > ratios_orgin:
        width = int(width_orgin * (ratios_target / ratios_orgin))
        ref_image = np.pad(ref_image, [[0, 0], [(width - width_orgin) // 2] * 2, [0, 0]], mode="mean")
        ref_mask = np.pad(ref_mask, [[0, 0], [(width - width_orgin) // 2] * 2, [0, 0]], mode="constant", constant_values=255)
    elif ratios_target < ratios_orgin:
        height = int(height_orgin * (ratios_orgin / ratios_target))
        ref_image = np.pad(ref_image, [[int((height - height_orgin) * 0.1), 0],  [0, 0], [0, 0]], mode="edge")
        ref_image = np.pad(ref_image, [[int((height - height_orgin) * 0.1), 0],  [0, 0], [0, 0]], mode="mean")
        ref_mask = np.pad(ref_mask, [[int((height - height_orgin) * 0.2), 0],  [0, 0], [0, 0]], mode="constant", constant_values=255)
        is_first = True
        while ref_image.shape[0] < height:
            ref_image = np.pad(ref_image, [[0, min(height // 4, height - ref_image.shape[0])],  [0, 0], [0, 0]], mode=("edge" if is_first else "mean"))
            is_first = False
            ref_mask = np.pad(ref_mask, [[0, (ref_image.shape[0] - ref_mask.shape[0])],  [0, 0], [0, 0]], mode="constant", constant_values=255)
            ref_image[ref_mask > 127] = util.blur(ref_image, blur_weight)[ref_mask > 127]
    else:
        return
    ref_mask = util.blur(ref_mask, 32)
    
    zoom = width * height / max_pixel
    if zoom > 1:
        width, height = int(width / math.sqrt(zoom)), int(height / math.sqrt(zoom))
        ref_image = util.resize_image(ref_image, width, height)
        ref_mask = util.resize_image(ref_mask, width, height)
    
    style = params.get("style", "")
    params["size"] = (width, height)
    params["steps"] = (base_step, int(base_step // 2.5))
    params["image"] = (ref_image, ref_mask)
    if style != "":
        params["denoise"] = 0.9
        params["skip_step"] = 0.1

        # ref_mode, _ = opts.options["ref_mode"]["Ref Depth"]
        # params["controlnet"].append(["Ref Depth", ref_mode, ref_image, 0.6])
        params["controlnet"].append(UseControlnet("Ref Depth", ref_image, 0.6))
    else:
        params["skip_step"] = 0.2
        
        # ref_mode, _ = opts.options["ref_mode"]["Ref All"]
        # params["controlnet"].append(["Ref All", ref_mode, orgin_image, 0.6])
        params["controlnet"].append(UseControlnet("Ref All", ref_image, 0.6))
    
    util.save_temp_image(ref_image, "change_size.png")
    util.save_temp_image(ref_mask, "change_size_mask.png")

    return params

def Refiner(ref_image, params, pnginfo, gen_opts, mask=None):
    # blurred = cv2.GaussianBlur(ref_image, (5, 5), 2)
    # ref_image = cv2.addWeighted(ref_image, 1.5, blurred, -0.5, 0)
    height, width = ref_image.shape[:2]
    max_pixel = 1280 ** 2
    re_zoom = math.sqrt(width * height / max_pixel)
    if re_zoom < 1:
        width, height = int(width / re_zoom), int(height / re_zoom)
        ref_image = util.resize_image(ref_image, width, height)
    util.save_temp_image(ref_image, "refiner.png")

    # step_base = max(20, pnginfo.get("all_steps", (20, 0))[0])
    step_base = 40

    params["action"] = "generate"
    # params["batch"] = gen_opts.pop("pic_num")
    
    params["steps"] = (step_base, step_base // 2.5)
    params["skip_step"] = int(step_base * 1.0)
    # params["seed"] = pnginfo.get("seed", 0)
    params["seed"] = 0
    params["sample"] = gen_opts.get("sample", "") or "dpmpp_3m_sde_gpu"
    params["clip_skip"] = -7
    params["noise_scale"] = gen_opts.get("detail", 1.0) ** 2
    params["refiner_name"] = gen_opts.get("refiner_model", params.get("refiner_name", ""))

    prompt_main = (params.get("prompt_main", "") or "(UHD,8K,ultra detailed) " + ",".join(wd14tagger.tag(ref_image)[:12]))
    params["prompt_main"] = prompt_main
    params["prompt_negative"] = params.get("prompt_negative", "") or "ugly,deformed,noisy,blurry,NSFW"

    if mask is None:
        app = FaceAnalysis(root=modules.paths.face_models_path)
        app.prepare(ctx_id=0, det_size=(640, 640))
        faces = app.get(ref_image)
        mask = None

        if faces:
            min_face_area = 448 ** 2
            landmarks = [face.landmark_2d_106 for face in faces if (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]) < min_face_area]
            if landmarks:
                mask = util.face_mask(ref_image, landmarks)
                mask = 255 - mask
    
    params["image"] = (ref_image, mask)

    # ref_mode, _ = opts.options["ref_mode"]["Ref Stuct"]
    # params["controlnet"].append(["Ref Stuct", ref_mode, ref_image, 0.75])
    # params["controlnet"].append(UseControlnet("Ref Stuct", ref_image, 0.75))

    return params

def RefinerFace(ref_image, params, pnginfo, gen_opts):
    app = FaceAnalysis(root=modules.paths.face_models_path)
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = app.get(ref_image)
    faces = sorted(faces, key = lambda x : ((x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1])), reverse=True)

    if faces:
        landmarks = [face.landmark_2d_106 for face in faces]
        ref_mask = util.face_mask(ref_image, landmarks)
        # face_mask = ref_mask[:,:,0].astype(np.float32) / 255
        # face_mask = face_mask.reshape(ref_image.shape[0], ref_image.shape[1], 1)
        # ref_image = ref_image * (1 - face_mask) + util.blur(ref_image, 20) * face_mask
        # ref_image = ref_image.astype(np.uint8)
        # ref_image = util.blur(ref_image, 2)
        gen_opts["detail"] = 0.95

        params = Refiner(ref_image, params, pnginfo, gen_opts, mask=ref_mask)
        prompt_main = params["prompt_main"]
        prompt_main = ("(detail face:1.2)" + prompt_main) if "detail face" not in prompt_main else prompt_main
        params["prompt_main"] = prompt_main
        params["steps"] = (params["steps"][0], params["steps"][0] // 8)
        params["skip_step"] = 0.7
        params["controlnet"] = []
        
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
    params = {}
    params["action"] = "upscale"
    params["file_format"] = gen_opts.get("file_format")
    params["image"] = np.array(ref_image)

    return params
        