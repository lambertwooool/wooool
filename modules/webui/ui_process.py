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
from PIL import Image

import modules.paths
import modules.options as opts
import modules.worker as worker
from modules import devices, lora, shared, util
from modules.model import model_helper

progress_html = "<progress value='{}' max='100'></progress><div class='progress_text'>{}</div>"
opt_dict = []
ref_image_count = [5, 3]
lora_count = [4, 4]
cur_taskid = None
action_btns = []
page_size = 128

def Generate(img_refer, ckb_pro, txt_setting, *args):
    args = list(args)
    gen_opts, setting = GetGenOptions(txt_setting)

    radio_mc = list(setting["mc"])[0]

    ref_num = gen_opts.pop('ref_num', 0)
    refiner_model = gen_opts.pop("refiner_model")
    if refiner_model == opts.title["disable_refiner"]:
        refiner_model = ""

    seed = gen_opts.pop("seed")
    fixed_seed = gen_opts.pop("fixed_seed", False)

    prompt_negative = gen_opts.pop("prompt_negative")
    for x in ["mc", "style", "view", "emo", "weather", "hue"]:
        if isinstance(gen_opts[x], dict):
            prompt_negative += ", " + gen_opts[x].get("negative_prompt", "")
            gen_opts[x] = gen_opts[x].get("prompt", "")

    params = {
        "action": "generate",
        "lang": gen_opts.pop("lang"),
        "base_name": gen_opts.pop("base_model"),
        "refiner_name": refiner_model,
        "main_character": (radio_mc, gen_opts.pop("mc")) if radio_mc != "Other" else ("", gen_opts.pop("mc_other")),
        "prompt_main": gen_opts.pop('prompt_main'),
        # "prompt_negative": ",".join(x for x in ([gen_opts.pop("negative")] + [v for n in gen_opts.pop("prompt_negative") for v in n.values() ]) if x != ""),
        "prompt_negative": prompt_negative,
        "cfg_scale": gen_opts.pop("cfg"),
        "cfg_scale_to": gen_opts.pop("cfg_to"),
        "aspect_ratios": gen_opts.pop("ratios"),
        "batch": gen_opts.pop("pic_num"),
        "style": gen_opts.pop("style_item"),
        "simpler": gen_opts.pop("simpler"),
        "scheduler": gen_opts.pop("scheduler"),
        "seed": seed,
        "fixed_seed": fixed_seed,
        "subseed_strength": gen_opts.pop("subseed_strength"),
        "quality": gen_opts.pop("quality"),
        "step_scale": gen_opts.pop("step_scale", 1.0),
        "refiner_step_scale": gen_opts.pop("refiner_step_scale", 1.0),
        "clip_skip": gen_opts.pop("clip_skip"),
        "noise_scale": gen_opts.pop("detail"),
        "denoise": gen_opts.pop("denoise"),
        "file_format": gen_opts.pop("file_format"),
        "single_vae": gen_opts.pop("single_vae"),
        "options": {
            k: v for k, v in gen_opts.items() if k in ["view", "emo", "location", "weather", "hue"]
        },
        "controlnet": [],
        # "lora": [('sd_xl_offset_example-lora_1.0', 0.5, '')]
        "lora": [],
    }

    for x in ["mc", "style", "view", "emo", "location", "weather", "hue", "prompt_negative"]:
        pm_weight = gen_opts.pop(f'{x}_weight', 1.0)
        if pm_weight != 1.0:
            if x == "mc":
                mc_name, mc_prompt = params["main_character"]
                params["main_character"] = (mc_name, f"({mc_prompt}:{pm_weight})")
            elif x == "style":
                params["style_weight"] = pm_weight
            elif x == "prompt_negative":
                params["prompt_negative"] = f"({params['prompt_negative']}:{pm_weight})"
            elif x in params["options"] and params["options"].get(x, "") != "":
                params["options"][x] = f"({params['options'][x]}:{pm_weight})"
    
    # Ref Image, ControlNet, Lora
    if not ckb_pro:
        if img_refer is not None:
            ref_mode, _ = opts.options["ref_mode"]["Ref All"]
            params["controlnet"].append(["Ref All", ref_mode, img_refer, 0.5])
    else:
        len_ref_ctrl = 4
        ref_image_args = args[: ref_image_count[0] * len_ref_ctrl]
        args = args[ref_image_count[0] * len_ref_ctrl :]
        for i in range(ref_image_count[0]):
            opt_type, image_refer, sl_rate, ckb_words = ref_image_args[i * len_ref_ctrl : (i + 1) * len_ref_ctrl]
            ref_mode, _ = opts.options["ref_mode"][opt_type]
            if ref_mode == "base_image":
                params["image"] = (image_refer, None)
                skip_rate = sl_rate / 100.0
                params["skip_step"] = skip_rate if skip_rate < 1 else 0.99
            elif ref_mode == "content":
                params["prompt_main"] += f",({','.join(ckb_words)}:{sl_rate / 100.0 / 0.8:.2f})"
            else:
                if image_refer is not None:
                    params["controlnet"].append([opt_type, ref_mode, image_refer, sl_rate / 100.0])

        len_lora_ctrl = 3
        lora_args = args[: lora_count[0] * len_lora_ctrl]
        args = args[lora_count[0] * len_lora_ctrl :]
        for i in range(lora_count[0]):
            opt_lora, sl_weight, opt_trained_words = lora_args[i * len_lora_ctrl : (i + 1) * len_lora_ctrl]
            if lora.lora_files.get(opt_lora, "") != "":
                lora_filename = os.path.split(lora.lora_files.get(opt_lora))[1]
                params["lora"].append((lora_filename, sl_weight / 100.0, opt_trained_words))

    proc_output = ProcessTask(params)
    while True:
        try:
            yield next(proc_output)
        except StopIteration:
            break

def ProcessTask(params, default_image=None):
    global cur_taskid

    cur_taskid = worker.append(params)
    finished = False
    batch_start_time = params["create_time"]
    cur_batch = -1
    default_pic_preview = default_image or os.path.join(modules.paths.css_path, "logo-t-s.png")
    disable_actions = tuple([gr.Button(interactive=False) for _ in range(len(action_btns) + 1)]) # 2=btn_delete, btn_top
    enable_actions = tuple([gr.Button(interactive=True) for _ in range(len(action_btns) + 1)]) # 2=btn_delete, btn_top

    yield   gr.HTML(visible=False), \
            gr.HTML(value=progress_html.format(0, "Initializing")), \
            gr.Gallery(value=[(default_pic_preview, "Initializing")] + GetSampleList(), allow_preview=False), \
            gr.Row(visible=True), \
            gr.Row(visible=False), \
            gr.Column(visible=True), \
            gr.Button(interactive=True), \
            gr.Button(interactive=True), \
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

            if picture is not None:
                pic_preview = [(picture, "Generating")]

            yield   gr.HTML(visible=False), \
                    gr.HTML(value=progress_html.format(percent, message)), \
                    gr.Gallery(value=pic_preview + GetSampleList(end_time=batch_start_time)), \
                    gr.Row(visible=True), \
                    gr.Row(visible=False), \
                    gr.Column(visible=True), \
                    gr.Button(interactive=True) if refresh_skip else gr.Button(), \
                    gr.Button(), \
                    *disable_actions
        time.sleep(0.05)

    yield   gr.HTML(visible=False), \
            gr.HTML(value=progress_html.format(0, 'finished')), \
            gr.Gallery(value=GetSampleList()), \
            gr.Row(visible=True), \
            gr.Row(visible=True), \
            gr.Column(visible=False), \
            gr.Button(interactive=True), \
            gr.Button(interactive=True), \
            *enable_actions
    
    cur_taskid = None
    
    execution_time = time.time() - params["create_time"]
    print(f'Total time: {execution_time:.2f} seconds')

def Process(process_handler):
    def func(gl_sample_list, num_selected_sample, txt_setting, *args):
        file_path, image = GetSampleImage(gl_sample_list, num_selected_sample)

        if file_path is not None:
            gen_opts, setting = GetGenOptions(txt_setting)
            params, pnginfo = ParseImageToTask(image, return_pnginfo=True)
            ref_image = np.array(image)

            params["base_name"] = gen_opts.get("vary_model")
            params["refiner_name"] = gen_opts.get("refiner_model")
            params["batch"] = gen_opts.get("pic_num")
            params["quality"] = gen_opts.get("quality")
            params["file_format"] = gen_opts.get("file_format")
            params["single_vae"] = gen_opts.get("single_vae")

            min_pixel = opts.options["quality_setting"][str(params["quality"])]["sdxl"][0] ** 2
            h, w = ref_image.shape[:2]

            if w * h < min_pixel:
                rezoom = math.sqrt(w * h / min_pixel)
                ref_image = util.resize_image(ref_image, w / rezoom, h / rezoom, size_step=8)
            params.pop("size")

            params = process_handler(ref_image, params, pnginfo, gen_opts, *args)

            proc_output = ProcessTask(params)
            while True:
                try:
                    yield next(proc_output)
                except StopIteration:
                    break
    
    return func

def StopProcess():
    worker.stop(cur_taskid)

    return  gr.Button(interactive=False), \
            gr.Button(interactive=False)

def SkipBatch():
    worker.skip(cur_taskid)

    return  gr.Button(interactive=False)

def ChangeSeed(ckb_seed):
    if not ckb_seed:
        seed = random.randint(1, 1024 ** 3)
        return gr.Number(value=seed)
    else:
        return gr.Number()

def GetSampleList(show_count=0, page=0, end_time=None, return_page_count=False):
    page = int(page)
    show_count = show_count if show_count > 0 else page_size
    excludes = ["temp.png"]
    files = util.list_files(modules.paths.temp_outputs_path, excludes=excludes)
    files = sorted(files, key=lambda x: os.path.getmtime(x), reverse=True)
    if end_time is not None:
        files = filter(lambda x: os.path.getmtime(x) < end_time, files)
    files = [(x, os.path.splitext(os.path.split(x)[-1])[0] if os.path.getsize(x) > 50 * 1024 else "Encoding") for x in files]
        
    if not return_page_count:
        return files[page * show_count : (page + 1) * show_count]
    else:
        return files[page * show_count : (page + 1) * show_count], math.ceil(len(files) / show_count)

def GetStyleList(style):
    fliename = f"{modules.paths.sd_style_path}/{opts.options['style'][style]}.json"
    data = util.load_json(fliename)
    files = []

    for x in data:
        cover = "cover_default"
        picture_path =  f"{modules.paths.sd_style_path}/{cover}/{x['name']}." + "{}"
        picture = [picture_path.format(ext) for ext in ["png", "jpg", "jpeg"] if os.path.exists(picture_path.format(ext))]
        picture = picture[0] if picture else f"{modules.paths.sd_style_path}/default.png"
        files.append((picture, x['name']))
    return files

def RefreshModels():
    base_models = model_helper.get_base_list()
    shared.xl_base = None
    shared.xl_refiner = None
    devices.torch_gc()

    return gr.Dropdown(base_models), gr.Dropdown(GetRefinerModels("sdxl")), gr.Dropdown([opts.title["disable_refiner"]] + GetRefinerModels("sdxl"))

def GetRefinerModels(base_model):
    model_type = "(sd15)" if "sd15" in base_model else "sdxl"
    renfiner_models = [x for x in list(model_helper.base_files) if model_type in x]
    return renfiner_models

def GetSelectSamplePath(gl_sample_list, num_selected_sample):
    selected_index = int(num_selected_sample)
    file_path = None

    if selected_index >= 0:
        selected_image = gl_sample_list[selected_index]
        filename = os.path.split(selected_image[0]["name"])[1]
        file_path = os.path.join(modules.paths.temp_outputs_path, filename)
    
    return file_path

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
        "simpler": info.get("sampler", ""),
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

def GetSampleImage(gl_sample_list, num_selected_sample):
    file_path = GetSelectSamplePath(gl_sample_list, num_selected_sample)
    image = None
    
    if file_path is not None and os.path.exists(file_path):
        image = Image.open(file_path)
    return file_path, image

def DeleteSample(gl_sample_list, num_selected_sample, num_page_sample):
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

    return gr.Gallery(value=GetSampleList(), allow_preview=False), gr.Number(0)

def SetPageSize(sl_sample_pagesize):
    global page_size
    page_size = int(sl_sample_pagesize)

def TopSample(gl_sample_list, num_selected_sample, num_page_sample):
    file_path = GetSelectSamplePath(gl_sample_list, num_selected_sample)
    if file_path is not None:
        os.utime(file_path, (time.time(), time.time()))

    return gr.Gallery(value=GetSampleList(page=num_page_sample), allow_preview=False)

def FirstPageSample(num_page_sample):
    page = 0
    return gr.Gallery(value=GetSampleList(page=page), allow_preview=False), gr.Number(page)

def PrevPageSample(num_page_sample):
    page = max(0, num_page_sample - 1)
    return gr.Gallery(value=GetSampleList(page=page), allow_preview=False), gr.Number(page)

def NextPageSample(num_page_sample):
    page = num_page_sample + 1
    list, page_count = GetSampleList(page=page, return_page_count=True)
    page = min(page_count, page)
    return gr.Gallery(value=list, allow_preview=False), gr.Number(page)

def SelectSample(evt: gr.SelectData, gl_sample_list, ckb_fullscreen):  # SelectData is a subclass of EventData
    file_path, image = GetSampleImage(gl_sample_list, evt.index)
    if image is not None:
        geninfo, items = util.read_info_from_image(image)
        w, h, s, f = image.width, image.height, os.path.getsize(file_path), image.format
        wr, hr = util.size2ratio(w, h, 10)
        return gr.Gallery(allow_preview=True), evt.index, gr.Text(f"{geninfo}\n\n{file_path}\n{w}x{h}({wr}:{hr}) {util.size_str(s)} {f}".strip(), visible=ckb_fullscreen)
    else:
        print(evt.index)
        return gr.Gallery(allow_preview=True) ,evt.index, ""

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
                    opt_value = round(int(opt_key) * opts.mul[opt_name], 2)
                    gen_opts[opt_name] = { opt_key: opt_value }
                else:
                    gen_opts[opt_name] = opt_key
        except:
            pass

        # opts.default[item]
    
    gen_opts["style_item"] = gl_style_list[int(gen_opts.get("style_index", 0))][1]

    return json.dumps(gen_opts)

def InitSetting(txt_setting):
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
    for k, v in opt_dict.items():
        try:
            if k in ["mc", "mc_other", "prompt_main", "lang", "style_index"]:
                objs.append(gr_contorls.get(str(v), gr.update)())
            else:
                if isinstance(config[k], list):
                    v1 = [list(x)[0] if isinstance(x, dict) else x for x in config[k]]
                else:
                    v1 = list(config[k])[0] if isinstance(config[k], dict) else config[k]
                # print(k, v, str(v) in ["dropdown", "slider"], config[k], v1)
                if str(v) in ["dropdown", "radio"]:
                    if v1 not in [x[0] for x in v.choices]:
                        objs.append(gr_contorls.get(str(v), gr.update)())
                        continue
                objs.append(gr_contorls.get(str(v), gr.update)(value=v1))
                
        except:
            objs.append(gr_contorls.get(str(v), gr.update)())
            
    return objs

def GetNegativeText(ckb_grp_negative):
    return ','.join([opts.options["recommend_negative"].get(k, "") for k in ckb_grp_negative])

def ChangeRefBlockNum(sl_refNum):
    return [gr.Column(visible=True)] * sl_refNum + [gr.Column(visible=False)] * (5 - sl_refNum)

def GetControlnets():
    files = util.list_files(modules.paths.controlnet_models_path)
    files = [os.path.split(os.path.splitext(x)[0])[-1] for x in files]
    return files

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