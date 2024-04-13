import json
import math
import os
import re
import cv2
import gradio as gr
import numpy as np 
import shutil
import time
import random
import torch
import traceback
from PIL import Image

import modules.paths
import modules.options as opts
import modules.worker as worker
from modules import civitai, controlnet_helper, devices, lora, shared, util, wd14tagger
from modules.model import model_helper

progress_html = "<progress value='{}' max='100'></progress><div class='progress_text'>{}</div>"
opt_dict = []
ref_image_count = [5, 3]
lora_count = [6, 3]
cur_taskid = None
action_btns = []
page_size = 128
endless_mode = False
mask_num_selected_sample = -1

def GenerateOne(img_refer, ckb_pro, txt_setting, *args):
    setting = json.loads(txt_setting)
    setting["pic_num"] = 1
    setting["show_endless"] = False
    proc_output = Generate(img_refer, ckb_pro, json.dumps(setting), *args)
    while True:
        try:
            yield next(proc_output)
        except StopIteration:
            break

def Generate(img_refer, ckb_pro, txt_setting, *args):
    args = list(args)
    gen_opts, setting = GetGenOptions(txt_setting)

    radio_mc = list(setting["mc"])[0]

    refiner_model = gen_opts.pop("refiner_model")
    if refiner_model == opts.title["disable_refiner"]:
        refiner_model = ""

    fixed_seed = gen_opts.pop("fixed_seed", False)
    seed = gen_opts.pop("seed") if fixed_seed else 0

    prompt_negative = gen_opts.pop("prompt_negative")
    style_item = gen_opts.pop("style_item")
    loras = []

    if style_item == "random-style":
        style_item = f"__style_{gen_opts.get('style')}__"

    params = {
        "action": "generate",
        "lang": gen_opts.pop("lang"),
        "base_name": gen_opts.pop("base_model"),
        "refiner_name": refiner_model,
        "main_character": (radio_mc, gen_opts.pop("mc")) if radio_mc != "Other" else ("", gen_opts.pop("mc_other")),
        "prompt_main": gen_opts.pop('prompt_main'),
        # "prompt_negative": ",".join(x for x in ([gen_opts.pop("negative")] + [v for n in gen_opts.pop("prompt_negative") for v in n.values() ]) if x != ""),
        "prompt_negative": prompt_negative,
        "cfg_scale": gen_opts.pop("cfg_scale"),
        "cfg_scale_to": gen_opts.pop("cfg_scale_to"),
        "cfg_multiplier": gen_opts.pop("cfg_multiplier"),
        "free_u": gen_opts.pop("free_u"),
        "eta": gen_opts.pop("eta"),
        "aspect_ratios": gen_opts.pop("ratios"),
        "batch": gen_opts.pop("pic_num"),
        "style": style_item if not gen_opts.pop("disable_style", False) else None,
        "sampler": gen_opts.pop("sampler"),
        "scheduler": gen_opts.pop("scheduler"),
        "seed": seed,
        "fixed_seed": fixed_seed,
        "subseed_strength": gen_opts.pop("subseed_strength"),
        "quality": gen_opts.pop("quality"),
        "style_aligned_scale": gen_opts.pop("style_aligned"),
        # "step_scale": gen_opts.pop("step_scale", 1.0),
        # "refiner_step_scale": gen_opts.pop("refiner_step_scale", 1.0),
        "steps": (gen_opts.pop("step_base"), gen_opts.pop("step_refiner")) if gen_opts.pop("custom_step") else (None, None),
        "clip_skip": gen_opts.pop("clip_skip"),
        "detail": gen_opts.pop("detail"),
        "denoise": gen_opts.pop("denoise"),
        "file_format": gen_opts.pop("file_format"),
        "transparent_bg": gen_opts.pop("transparent_bg"),
        "single_vae": gen_opts.pop("single_vae", False),
        "options": {
            k: v if isinstance(v, dict) else { "prompt": v } \
                for k, v in gen_opts.items() if k in ["view", "emo", "location", "weather", "hue"]
        },
        "dtype": {},
        "lowvram_dtype": {},
        "controlnet": [],
        "lora": loras,
    }

    dtypes = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }
    for x in ["model", "clip", "vae", "controlnet", "ipadapter"]:
        params["dtype"][x] = dtypes.get(gen_opts.pop(f"{x}_dtype", None))
        params["lowvram_dtype"][x] = dtypes.get(gen_opts.pop(f"{x}_lowvram_dtype", None))

    if gen_opts.pop("custom_size", False):
        params["size"] = (int(gen_opts.pop("image_width")), int(gen_opts.pop("image_height")))

    for x in ["mc", "style", "view", "emo", "location", "weather", "hue", "prompt_negative", "more_art"]:
        pm_weight = gen_opts.pop(f'{x}_weight', 1.0)
        if pm_weight != 1.0:
            if x == "mc":
                mc_name, mc_prompt = params["main_character"]
                if isinstance(mc_prompt, dict):
                    params["main_character"][1]["weight"] = pm_weight
                else:
                    params["main_character"] = (mc_name, { "prompt": mc_prompt, "weight": pm_weight})
            elif x in ["style", "more_art"]:
                params[f"{x}_weight"] = pm_weight
            elif x == "prompt_negative":
                params["prompt_negative"] = f"({params['prompt_negative']}:{pm_weight})"
            elif x in params["options"]:
                params["options"][x]["weight"] = pm_weight
    
    # Base Image, ControlNet, Lora
    if not ckb_pro:
        if img_refer is not None:
            ref_mode, _ = opts.options["ref_mode"]["Ref All"]
            params["controlnet"].append(["Ref All", ref_mode, img_refer, 0.5])
    else:
        len_ref_ctrl = 13
        ref_image_args = args[: ref_image_count[0] * len_ref_ctrl]
        ref_num = int(gen_opts.get("ref_num", ref_image_count[1]))
        args = args[ref_image_count[0] * len_ref_ctrl :]
        for i in range(ref_image_count[0]):
            opt_type, ckb_enable, image_refer, sl_rate, ckb_words, txt_words, opt_model, ckb_annotator, opt_annotator, ckb_mask, image_attn_mask, sl_start_percent, sl_end_percent = ref_image_args[i * len_ref_ctrl : (i + 1) * len_ref_ctrl]
            if not ckb_enable or i >= ref_num:
                continue

            ref_mode, _ = opts.options["ref_mode"][opt_type]
            if ref_mode == "base_image":
                if image_refer is not None:
                    params["base_image"] = (image_refer, None)
                    skip_rate = sl_rate / 100.0
                    params["skip_step"] = skip_rate if skip_rate < 1 else 0.99
            elif ref_mode == "content":
                params["prompt_main"] += f",({','.join(ckb_words)},{txt_words}:{sl_rate / 100.0 / 0.8:.2f})"
            else:
                if image_refer is not None or params.get("image"):
                    opt_annotator = opt_annotator or ref_mode if ckb_annotator or opt_annotator in ["ip_adapter", "ip_adapter_face"] else "default"
                    image_attn_mask = image_attn_mask if ckb_mask else None
                    params["controlnet"].append([opt_type, opt_annotator, image_refer, sl_rate / 100.0, opt_model, image_attn_mask, sl_start_percent / 100.0, sl_end_percent / 100.0])
        
        if params.get("base_image"):
            params["image"] = params.pop("base_image")

        len_lora_ctrl = 4
        lora_args = args[: lora_count[0] * len_lora_ctrl]
        lora_num = int(gen_opts.get("lora_num", lora_count[1]))
        args = args[lora_count[0] * len_lora_ctrl :]
        for i in range(lora_count[0]):
            opt_lora, ckb_enable, sl_weight, opt_trained_words = lora_args[i * len_lora_ctrl : (i + 1) * len_lora_ctrl]
            if not ckb_enable or i >= lora_num:
                continue
            if lora.lora_files.get(opt_lora, "") != "":
                lora_filename = os.path.split(lora.lora_files.get(opt_lora))[1]
                params["lora"].append((lora_filename, sl_weight / 100.0, opt_trained_words))

    proc_output = ProcessTask(params, hide_history=gen_opts.get("hide_history", False), show_endless=gen_opts.get("show_endless", True))
    while True:
        try:
            yield next(proc_output)
        except StopIteration:
            break

def ProcessTask(params, default_image=None, hide_history=False, show_endless=False):
    global cur_taskid

    disable_actions = tuple([gr.Button(interactive=False) for _ in range(len(action_btns) + 1)]) # 2=btn_delete, btn_top
    enable_actions = tuple([gr.Button(interactive=True) for _ in range(len(action_btns) + 1)]) # 2=btn_delete, btn_top

    if params is not None:
        cur_taskid = worker.append(params)
        finished = False
        batch_start_time = params["create_time"]
        cur_batch = -1
        default_pic_preview = default_image or os.path.join(modules.paths.css_path, "logo-t-s.png")
        simple_list = GetSampleList() if not hide_history else []
        sample_start_time = os.path.getmtime(simple_list[-1][0]) if not hide_history else batch_start_time
        pic_preview = [(default_pic_preview, "Initializing")]

        yield   gr.HTML(visible=False), \
                gr.HTML(value=progress_html.format(0, "Initializing")), \
                gr.Row(visible=True), \
                gr.Row(visible=False), \
                gr.Column(visible=True), \
                gr.Button(interactive=True), \
                gr.Button(interactive=True), \
                gr.Gallery(value=pic_preview + simple_list), \
                *disable_actions

        while not finished:
            if len(shared.outputs):
                task, percent, finished, message, picture = shared.outputs.pop()
                # print(time.time(), percent, finished, len(shared.outputs), message)
                shared.outputs.clear()

                task_cur_batch = task.get("cur_batch", 0)
                refresh_skip = False
                if cur_batch < task_cur_batch:
                    batch_start_time = time.time()
                    cur_batch = task_cur_batch
                    pic_preview = [(default_pic_preview, "Waiting")]
                    refresh_skip = True
                    simple_list = GetSampleList(start_time=sample_start_time, end_time=batch_start_time)

                if picture is not None:
                    pic_preview = [(picture, "Generating")]

                yield   gr.HTML(visible=False), \
                        gr.HTML(value=progress_html.format(percent, message)), \
                        gr.Row(visible=True), \
                        gr.Row(visible=False), \
                        gr.Column(visible=True), \
                        gr.Button(interactive=True) if refresh_skip else gr.Button(), \
                        gr.Button(), \
                        gr.Gallery(value=pic_preview + simple_list), \
                        *disable_actions
                        
            time.sleep(0.01)

    yield *ProcessFinish(show_endless), *enable_actions
    
    cur_taskid = None
    
    execution_time = (time.time() - params.get("create_time", time.time())) if params else 0
    print(f'Total time: {execution_time:.2f} seconds')

def ProcessFinishNoRefresh(endless_mode):
    if endless_mode:
        return  gr.HTML(), \
                gr.HTML(value=progress_html.format(0, 'batch finished and continue next ...')), \
                gr.Row(), \
                gr.Row(), \
                gr.Column(), \
                gr.Button(), \
                gr.Button()
                
    else:
        return  gr.HTML(visible=False), \
                gr.HTML(value=progress_html.format(0, 'finished')), \
                gr.Row(visible=True), \
                gr.Row(visible=True), \
                gr.Column(visible=False), \
                gr.Button(interactive=True), \
                gr.Button(interactive=True)

def ProcessFinish(endless_mode):
    return *ProcessFinishNoRefresh(endless_mode), gr.Gallery(value=GetSampleList())

def Process(process_handler):
    def func(gl_sample_list, num_selected_sample, txt_setting, *args):
        file_path, image = GetSampleImage(gl_sample_list, num_selected_sample)

        if file_path is not None:
            gen_opts, setting = GetGenOptions(txt_setting)
            params, pnginfo = ParseImageToTask(image, return_pnginfo=True)
            ref_image = np.array(image)[:,:,:3]

            params["base_name"] = gen_opts.get("vary_model")
            params["prompt_main"] = params.get("prompt_main", "") or ",".join(wd14tagger.tag(ref_image)[:12])
            params["refiner_name"] = gen_opts.get("refiner_model")
            params["batch"] = gen_opts.get("pic_num")
            params["clip_skip"] = -1 * abs(int(params.get("clip_skip") or gen_opts.get("clip_skip") or 2))
            params["quality"] = gen_opts.get("quality")
            params["file_format"] = gen_opts.get("file_format")
            params["single_vae"] = gen_opts.get("single_vae")
            params["lang"] = gen_opts.get("lang")

            # min_pixel = opts.options["quality_setting"][str(params["quality"])]["sdxl"][0] ** 2
            # h, w = ref_image.shape[:2]

            # if w * h < min_pixel:
            #     rezoom = math.sqrt(w * h / min_pixel)
            #     ref_image = util.resize_image(ref_image, w / rezoom, h / rezoom, size_step=8)
            params.pop("size")

            params = process_handler(ref_image, params, pnginfo, gen_opts, *args)

            proc_output = ProcessTask(params, hide_history=gen_opts.get("hide_history", False))
            while True:
                try:
                    yield next(proc_output)
                except StopIteration:
                    break
    
    return func

def StopProcess():
    worker.stop(cur_taskid)

    return  gr.Button(interactive=False), \
            gr.Button(interactive=False), \
            gr.Checkbox(value=False)

def SkipBatch():
    worker.skip(cur_taskid)

    return  gr.Button(interactive=False)

def ChangeSeed(ckb_seed):
    if not ckb_seed:
        seed = random.randint(1, 2 ** 31 - 1)
        return gr.Number(value=seed)
    else:
        return gr.Number()

def GetModelInfo(opt_model):
    model_file = os.path.join(modules.paths.modelfile_path, model_helper.base_files[opt_model])
    url = None
    if os.path.exists(model_file) or civitai.exists_info(model_file):
        info = civitai.get_model_versions(model_file)
        url = info.get("model_homepage") if info else None

    if url:
        return gr.Button(link=url, interactive=True)
    else:
        return gr.Button(interactive=False)

def GetSampleList(show_count=0, page=0, start_time=None, end_time=None, return_page_count=False):
    page = int(page)
    show_count = show_count if show_count > 0 else page_size
    files = util.list_files(modules.paths.temp_outputs_path, ["jpg", "jpeg", "png"], excludes_dir=["annotator", "recycled", "temp"], search_subdir=True)
    files = sorted(files, key=lambda x: os.path.getmtime(x), reverse=True)
    if start_time is not None and end_time is not None:
        files = filter(lambda x: os.path.getmtime(x) > start_time and os.path.getmtime(x) < end_time, files)
    elif start_time is not None:
        files = filter(lambda x: os.path.getmtime(x) > start_time, files)
    elif end_time is not None:
        files = filter(lambda x: os.path.getmtime(x) < end_time, files)
    files = [(x, os.path.splitext(os.path.split(x)[-1])[0] if os.path.getsize(x) > 50 * 1024 else "Encoding") for x in files]
        
    if not return_page_count:
        return files[page * show_count : (page + 1) * show_count]
    else:
        return files[page * show_count : (page + 1) * show_count], math.ceil(len(files) / show_count)

def GetStyleList(style):
    fliename = f"{modules.paths.sd_style_path}/{opts.options['style'][style]}.json"
    data = [{ "name": "random-style" }] + util.load_json(fliename)
    files = []
    random_covers = []

    for x in data:
        cover = "cover_default"
        picture_path =  f"{modules.paths.sd_style_path}/{cover}/{x['name']}." + "{}"
        picture = [picture_path.format(ext) for ext in ["png", "jpg", "jpeg"] if os.path.exists(picture_path.format(ext))]
        picture = picture[0] if picture else f"{modules.paths.sd_style_path}/default.png"
        files.append([picture, x['name']])

        if len(random_covers) < 4 and x["name"] != "random-style":
            random_covers.append(cv2.cvtColor(cv2.imread(picture), cv2.COLOR_BGR2RGB))

    if len(random_covers) > 0:
        random_cover = util.concat_images(random_covers)
        files[0][0] = random_cover

    return files

def RefreshModels():
    base_models = model_helper.get_base_list()
    shared.xl_base = None
    shared.xl_refiner = None
    devices.torch_gc()

    return gr.Dropdown(base_models), gr.Dropdown(GetRefinerModels("sdxl")), gr.Dropdown([opts.title["disable_refiner"]] + GetRefinerModels("sdxl"))

def RefreshLoras():
    return [ gr.Dropdown(choices=["None"] + list(lora.get_list().keys())) ] * lora_count[0]

def GetRefinerModels(base_model):
    # model_type = "(sd15)" if "sd15" in base_model else "(sdxl)"
    # renfiner_models = [x for x in list(model_helper.base_files) if model_type in x]
    # renfiner_models = [x for x in list(model_helper.base_files) if "(sd15)" not in x]
    renfiner_models = [x for x in list(model_helper.base_files)]
    
    return renfiner_models

def GetSelectSamplePath(gl_sample_list, num_selected_sample):
    selected_index = int(max(0, num_selected_sample or 0))
    file_path = None

    if selected_index >= 0:
        selected_image = gl_sample_list[selected_index]
        filename = os.path.split(selected_image[0]["name"])[1]
        file_path = os.path.join(modules.paths.temp_outputs_path, filename)
        if not os.path.exists(file_path):
            file_path = os.path.join(modules.paths.temp_outputs_path, f"{filename[:6]}/{filename}")
    
    return file_path

def GetImageGenerateData(img_generate_data):
    if img_generate_data is not None:
        image = img_generate_data
        geninfo, items = util.read_info_from_image(image)

        return gr.Textbox(value=geninfo)
    else:
        return gr.Textbox()

def GetGenerateData(generate_data):
    alias = {
        "prompt": "prompt_main",
        "negative": "prompt_negative",
    }
    info = util.parse_generation_parameters(generate_data)
    info = { alias.get(k, k.replace(" ", "_")) : v for k, v in info.items() }

    if info.get("size"):
        size = [int(x) for x in info.get("size").split("x")]
    elif info.get("width") and info.get("height"):
        size = [int(info.get("width")), int(info.get("height"))]
    else:
        size = None
    
    if size:
        width, height = [int(x / 8) * 8 for x in size]
        src_ratio = width / height
        ratios = sorted([(k, v) for k, v in opts.options["ratios"].items()], key=lambda x: abs((x[1][0] / x[1][1]) - src_ratio))[0][1]
        info["aspect_ratios"] = ratios
        info["size"] = (width, height)
    
    if info.get("cfg_scale"):
        info["cfg_scale"] = round(float(info["cfg_scale"]) / 0.5) * 0.5
    
    if not info.get("cfg_scale_to") and info.get("cfg_scale"):
        info["cfg_scale_to"] = float(info.get("cfg_scale"))
    
    if info.get("steps"):
        step_all = int(info.pop("steps", 0))
        step_refiner = int(info.pop("refiner_steps", 0))
        info["steps"] = (max(0, step_all - step_refiner), max(0, step_refiner))
    
    if info.get("seed"):
        info["seed"] = int(info.get("seed"))
    
    sampler = info.get("sampler", "").lower()
    scheduler = info.get("scheduler", "").lower()
    if scheduler:
        scheduler = ([k for k, v in opts.options["scheduler"].items() if k.lower() == scheduler or v.lower() == scheduler] + [None])[0]
    elif sampler:
        scheduler = ([k for k, v in opts.options["scheduler"].items() if sampler.endswith(f" {k.lower()}") or sampler.endswith(f" {v.lower()}")] + [None])[0]
    if scheduler:
        info["scheduler"] = scheduler
    sampler = sampler[:-(len(scheduler) + 1)] if scheduler else sampler

    if sampler:
        sampler = ([k for k, v in opts.options["sampler"].items() if f"{k.lower()}".startswith(sampler) or f"{v.lower()}".startswith(sampler)] + [None])[0]
    info["sampler"] = sampler

    if info.get("clip_skip"):
        info["clip_skip"] = -1 * abs(int(info.get("clip_skip")))

    return info

def ParseGenerateData(txt_generate_data):
    info = GetGenerateData(txt_generate_data)
    info["custom_size"] = True
    info["custom_step"] = True

    ratios = info.get("aspect_ratios")
    if ratios:
        info["ratios"] = f"{ratios[0]}:{ratios[1]}"

    if info.get("seed"):
        info["fixed_seed"] = True

    info["negative"] = info.pop("prompt_negative") if info.get("prompt_negative") else None
    info["recommend_negative"] = []

    if info.get("clip_skip"):
        info["clip_skip"] = abs(info.get("clip_skip"))
    
    if not info.get("sdxl_style"):
        info["disable_style"] = True
    
    if info.get("steps"):
        steps = info.get("steps")
        info["step_base"] = steps[0]
        info["step_refiner"] = steps[1]
    
    width, height = info.get("size")
    info["image_width"] = width
    info["image_height"] = height

    # if info.get("main_character"):
    #     mc = re.search(r"\('(\w+)' .*", info.get("main_character"))
    #     if mc is not None:
    #         mc = mc.groups()[0]
    #     info["mc"] = mc
    # else:
    #     info["mc"] = "Other"
    info["mc"] = "Other"
    info["mc_other"] = ""

    return gr.Textbox(json.dumps(info))

def GenerateByData(txt_generate_data, txt_setting):
    params = GetGenerateData(txt_generate_data)
    gen_opts, setting = GetGenOptions(txt_setting)
    
    if params.get("sampler"):
        params["sampler"] = opts.options["sampler"][params.get("sampler")]
    
    if params.get("scheduler"):
        params["scheduler"] = opts.options["scheduler"][params.get("scheduler")]
    
    if not params.get("cfg_scale"):
        params["cfg_scale"] = gen_opts.pop("cfg_scale")
        params["cfg_scale_to"] = gen_opts.pop("cfg_scale_to")
    
    params["aspect_ratios"] = gen_opts.pop("ratios")
    # params["batch"] = gen_opts.pop("pic_num")
    params["batch"] = 1
    gen_opts["steps"] = (gen_opts.pop("step_base"), gen_opts.pop("step_refiner")) if gen_opts.pop("custom_step") else (None, None)
    
    params["base_name"] = gen_opts.pop("base_model")

    for k, v in gen_opts.items():
        if k in ["quality", "size", "steps", "seed", "sampler", "clip_skip", "scheduler"] and not params.get(k):
            params[k] = v

    proc_output = ProcessTask(params)
    while True:
        try:
            yield next(proc_output)
        except StopIteration:
            break

def ParseImageToTask(image, return_pnginfo=False):
    params = {}
    geninfo, items = util.read_info_from_image(image)

    info = util.parse_generation_parameters(geninfo)

    step_all = int(info.pop("steps", 0))
    step_refiner = int(info.pop("refiner steps", 0))
    info["all_steps"] = (max(step_refiner, step_all - step_refiner), step_refiner)

    params = {
        "prompt_main": info.get("prompt", ""),
        "prompt_negative": info.get("negative", ""),
        "main_character": info.get("main character", ("", "")),
        "style": info.get("sdxl style", ""),
        "cfg_scale": float(info.get("cfg scale", 7.0)),
        "batch": 1,
        "sampler": info.get("sampler", ""),
        "scheduler": info.get("scheduler", ""),
        "seed": int(info.get("seed", -1)),
        # "steps": (max(step_refiner, step_all - step_refiner), step_refiner),
        "clip_skip": int(info["clip skip"]) if info.get("clip skip", None) is not None else None,
        "size": image.size,
        "format": image.format,
        "controlnet": [],
        "lora": [],
        "skip_prompt": True,
    }

    if return_pnginfo:
        return params, info
    else:
        return params

def ChangeEndless(ckb_endless_mode):
    global endless_mode

    endless_mode = ckb_endless_mode

def GetSampleImage(gl_sample_list, num_selected_sample):
    file_path = GetSelectSamplePath(gl_sample_list, num_selected_sample)
    image = None
    
    if file_path is not None and os.path.exists(file_path):
        image = Image.open(file_path)
    return file_path, image

def DeleteSample(gl_sample_list, num_selected_sample, num_page_sample):
    if gl_sample_list is None:
        return gr.Gallery()

    file_path = GetSelectSamplePath(gl_sample_list, num_selected_sample)
    if file_path is not None:
        recycled_path = os.path.join(modules.paths.temp_outputs_path, "recycled")
        if not os.path.exists(recycled_path):
            os.makedirs(recycled_path)
        shutil.move(file_path, os.path.join(recycled_path, os.path.split(file_path)[1]))
    # print(file_path)

    return gr.Gallery(value=GetSampleList(page=num_page_sample))

def ChangePageSize(sl_sample_pagesize):
    global page_size
    page_size = int(sl_sample_pagesize)

    return gr.Gallery(value=GetSampleList()), gr.Number(0)

def SetPageSize(sl_sample_pagesize):
    global page_size
    page_size = int(sl_sample_pagesize)

def VaryClearMask(num_selected_sample):
    global mask_num_selected_sample

    if num_selected_sample == mask_num_selected_sample:
        return gr.Image()
    else:
        mask_num_selected_sample = num_selected_sample
        return gr.Image(value=None)

def VaryCustomInterface(gl_sample_list, num_selected_sample):
    file_path, image = GetSampleImage(gl_sample_list, num_selected_sample)

    return  gr.Column(visible=False), \
            gr.Column(visible=True), \
            gr.Row(visible=True), \
            gr.Column(visible=False), \
            gr.Column(visible=True), \
            gr.Image(value=file_path)

def TopSample(gl_sample_list, num_selected_sample, num_page_sample):
    file_path = GetSelectSamplePath(gl_sample_list, num_selected_sample)
    if file_path is not None:
        os.utime(file_path, (time.time(), time.time()))

    return gr.Gallery(value=GetSampleList(page=num_page_sample))

def FirstPageSample(num_page_sample):
    page = 0
    return gr.Gallery(value=GetSampleList(page=page)), gr.Number(page)

def PrevPageSample(num_page_sample):
    page = max(0, num_page_sample - 1)
    return gr.Gallery(value=GetSampleList(page=page)), gr.Number(page)

def NextPageSample(num_page_sample):
    page = num_page_sample + 1
    list, page_count = GetSampleList(page=page, return_page_count=True)
    page = min(page_count, page)
    return gr.Gallery(value=list), gr.Number(page)

def SelectSample(evt: gr.SelectData, gl_sample_list):  # SelectData is a subclass of EventData
    file_path, image = GetSampleImage(gl_sample_list, evt.index)
    if image is not None:
        geninfo, items = util.read_info_from_image(image)
        w, h, s, f = image.width, image.height, os.path.getsize(file_path), image.format
        wr, hr = util.size2ratio(w, h, 10)
        return evt.index, gr.Text(f"{geninfo}\n\n{file_path}\n{w}x{h}({wr}:{hr}) {util.size_str(s)} {f}".strip())
    else:
        return evt.index, ""

def GetLoraFromPrompt(txt_prompt_main, opt_lora_num, *args):
    _, loras = lora.remove_prompt_lora(txt_prompt_main)
    items = []
    found_loras = []
    args = list(args)
    ctrl_count = 4
    count = 0
    cur_loras = [x for x in args[::ctrl_count] if x != "None"]
    lora_num = int(opts.options["lora_num"].get(opt_lora_num, 3))

    while args:
        opt_lora, ckb_enable, sl_weight, opt_trained_words = [args.pop(0) for _ in range(ctrl_count)]
        is_found = False

        if (not opt_lora or opt_lora == "None") and count < lora_num:
            while loras:
                lora_name, lora_weight = loras.pop()
                lora_path = lora.get_lora_path(lora_name)
                if lora_path is not None:
                    found_loras.append(lora_name)
                    show_name = lora.get_name_by_path(lora_path)
                    if show_name not in cur_loras:
                        items += [gr.Dropdown(value=show_name), gr.Checkbox(value=True), gr.Slider(value=lora_weight * 100), gr.Dropdown()]
                        is_found = True
                        break
        
        if not is_found:
            items += [gr.Dropdown(), gr.Checkbox(), gr.Slider(), gr.Dropdown()]

        count += 1

    prompt_main, _ = lora.remove_prompt_lora(txt_prompt_main, found_loras)
    items = [gr.Textbox(value=prompt_main)] + items

    return items

def GetSettingJson(gl_style_list, *args):
    opt_name_list = list(opt_dict)
    opt_args = args[: len(opt_name_list)]
    # args = args[len(opt_name_list) :]
    gen_opts = {}
    
    for opt_name, opt_key in zip(opt_name_list, opt_args):
        # print(opt_name, opt_key)
        try:
            if opt_name in opts.options and isinstance(opts.options[opt_name], dict):
                if isinstance(opt_key, list):
                    opt_value = [opts.options[opt_name][x] for x in opt_key]
                    gen_opts[opt_name] = [{ k: v } for k, v in zip(opt_key, opt_value)]
                else:
                    opt_value = opts.options[opt_name].get(opt_key, "")
                    gen_opts[opt_name] = { opt_key: opt_value }
                # gen_opts[opt_name] = opt_value
            else:
                # print(opt_name, opt_key)
                if opt_name in opts.mul:
                    opt_value = round(float(opt_key) * opts.mul[opt_name], 2)
                    gen_opts[opt_name] = { opt_key: opt_value }
                else:
                    gen_opts[opt_name] = opt_key
        except:
            traceback.print_exc()

        # opts.default[item]
    
    gen_opts["style_item"] = gl_style_list[int(gen_opts.get("style_index", 0))][1]

    return json.dumps(gen_opts)

def ResetSetting():
    data = {
        "prompt_main": "",
        "negative": "",
        "fixed_seed": False,
        "custom_step": False,
        "custom_size": False,
        "disable_style": False,
    }
    return InitSetting(json.dumps(data), reset=True)

def InitSetting(txt_setting, reset=False):
    config = json.loads(txt_setting)
    gr_contorls = {
        "dropdown": gr.Dropdown,
        "slider": gr.Slider,
        "radio": gr.Radio,
        "textbox": gr.Textbox,
        "number": gr.Number,
        "checkboxgroup": gr.CheckboxGroup,
        "checkbox": gr.Checkbox,
    }
    objs = []
    others = []
    # print(config)
    for k, v in opt_dict.items():
        try:
            if k in ["lang", "style_index"]:
                objs.append(gr_contorls.get(str(v), gr.update)())
            else:
                if isinstance(config.get(k), list):
                    v1 = [list(x)[0] if isinstance(x, dict) else x for x in config[k]]
                elif isinstance(config.get(k), dict):
                    v1 = list(config.get(k))[0]
                else:
                    v1 = config.get(k)

                # print(k, v, v1, v.value, opts.default.get(k))
                if (not v1 and not v.value) or reset:
                    v1 = opts.default.get(k, v1)

                if str(v) in ["dropdown", "radio"]:
                    if v1 not in [x[0] for x in v.choices]:
                        objs.append(gr_contorls.get(str(v), gr.update)())
                        continue
                
                if v1 is not None:
                    objs.append(gr_contorls.get(str(v), gr.update)(value=v1))
                else:
                    objs.append(gr_contorls.get(str(v), gr.update)())
                
        except:
            traceback.print_exc()
            objs.append(gr_contorls.get(str(v), gr.update)())
            
    return objs

def ChangeQuality(opt_quality):
    quality = opts.options["quality"][opt_quality]
    quality_file = opts.options["quality_setting"][str(quality)]["setting"]
    data = json.dumps(util.load_json(os.path.join("./configs/settings", quality_file)))

    return gr.Text(value=data)

def LoadSetting(btn_load_setting):
    data = json.dumps(util.load_json(btn_load_setting.name))
    return gr.Text(value=data)

def GetNegativeText(ckb_grp_negative):
    return ','.join([opts.options["recommend_negative"].get(k, "") for k in ckb_grp_negative])

def ChangeRefBlockNum(sl_refNum):
    return [gr.Column(visible=True)] * sl_refNum + [gr.Column(visible=False)] * (5 - sl_refNum)

def ChangeLoraBlockNum(opt_loraNum):
    loraNum = int(opts.options["lora_num"].get(opt_loraNum, 3))
    return [gr.Column(visible=True)] * loraNum + [gr.Column(visible=False)] * (6 - loraNum)

def GetControlnets(ref_mode, model_type):
    re_sdxl = r"\b(xl|sdxl|control\s*lora)\b"
    re_sd15 = r"\b(14|15|sd14|sd15)(v\d+)?\b"
    re_ref_mode = opts.options["ref_mode"][ref_mode][1].get("keyword")
    # re_ref_mode = re.sub(r"[\-_]", " ", re_ref_mode)
    default_model = opts.options["ref_mode"][ref_mode][1].get(model_type)

    if not isinstance(re_ref_mode, list):
        re_ref_mode = [re_ref_mode]

    def model_filter(filename, ref_mode, model_type):
        if filename == default_model:
            return False

        filename = re.sub(r"[\-_]", " ", filename)
        if model_type == "sdxl":
            if re.search(re_sdxl, filename) is None:
                return False
        elif model_type == "sd15":
            if re.search(re_sd15, filename) is None:
                return False
        else:
            return False
        
        if re_ref_mode is not None and all([re.search(x, filename) is None for x in re_ref_mode]):
            return False

        return True
            
    # files = util.list_files(modules.paths.controlnet_models_path, ["pth", "safetensors"])
    # files = [os.path.split(os.path.splitext(x)[0])[-1] for x in files if model_filter(x, ref_mode, model_type)]
    # files = [os.path.split(os.path.splitext(default_model)[0])[-1]] + files
    files = controlnet_helper.get_controlnets()
    files = [k for k, v in files.items() if model_filter(v.lower(), ref_mode, model_type)]
    default_model = os.path.splitext(default_model)[0] if default_model else None
    files = [default_model] + files if default_model not in files else files
    
    return files, default_model

def GetGenOptions(txt_setting):
    gen_opts = {}
    setting = json.loads(txt_setting)
    for opt_name, opt_value in setting.items():
        if isinstance(opt_value, dict):
            opt_key, opt_value = [(k, v) for k, v in opt_value.items()].pop()
        gen_opts[opt_name] = opt_value
    
    prompt_negative = ",".join(x for x in ([gen_opts.get("negative", "")] + [v for n in gen_opts.get("recommend_negative", []) \
                                for v in n.values() ]) if x != "")
    gen_opts["prompt_negative"] = prompt_negative
    
    return gen_opts, setting