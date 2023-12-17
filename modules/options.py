import os
from modules import util

def load_ui_config(name):
	return util.load_json(os.path.join("./configs/ui", f"{name}.json"))

title = {
	"prompt_placeholder": "Input prompt.",
	"ref_image": "Ref Image",
    "disable_refiner": "Disable Refiner"
}
default = {}
options = {}
mul = {}
default_base_name = "sd_xl_base_1.0_0.9vae.safetensors"
default_refiner_name = "sd_xl_refiner_1.0_0.9vae.safetensors"

# Main Character
options['mc'] = load_ui_config("mc") | {
	"Other": ""
} 
title['mc'] = "Main Character"
default['mc'] = 'Girl'

# Aspect ratio
options['ratios'] = {
	"1:1": (1, 1),
	"16:9": (16, 9),
	"5:3": (5, 3),
	"4:3": (4, 3),
	"3:4": (3, 4),
	"3:5": (3, 5),
	"9:16": (9, 16)
}
title['ratios'] = "Aspect ratio"
default['ratios'] = "16:9"

# Style
options['style'] = load_ui_config("style")
title['style'] = "Style"
default['style'] = "Cinematic & Photographic"

# View
options['view'] = load_ui_config("view")
title['view'] = "View"
default['view'] = "Default view"

# Emo
options['emo'] = load_ui_config("emo")
title['emo'] = "Emo"
default['emo'] = "Happy"

# Location
options['location'] = load_ui_config("location")
title['location'] = "Location"
default['location'] = "Any Where"

# Weather
options['weather'] = load_ui_config("weather")
title['weather'] = "Weather"
default['weather'] = "Any Weather"

# Lighting
options['hue'] = load_ui_config("lighting")
title['hue'] = "Lighting"
default['hue'] = "Rim lighting"

options['pic_num'] = {
	"1 Sample": 1,
	"2 Samples": 2,
	"4 Samples": 4,
	"8 Samples": 8,
    "9999 Samples": 9999,
}
# title['pic_num'] = "Piece"
title['pic_num'] = "Sample Settings"
default['pic_num'] = "2 Samples"

# Ref Num
options['ref_num'] = (1, 5, 1)
title['ref_num'] = "Ref Count"
default['ref_num'] = 3

# Quality
options['quality'] = {
    # "LCM Generate": 5,
	"Ex-Fast Generate": 1,
	"Fast Generate": 2,
	"High Quality": 3,
	"Ultra Quality": 4
}
title['quality'] = "Quality"
default['quality'] = "Fast Generate"

options['quality_setting'] = {
    # '5': { 'sdxl': (1024, 5, 0, 'lcm', 'lcm'), 'sd15': (768, 5, 0, 'lcm', 'lcm') },
	'1': { 'sdxl': (1152, 18, 0, 'ddim', 'karras'), 'sd15': (808, 18, 0, 'ddim', 'karras') },
	'2': { 'sdxl': (1152, 25, 0, 'dpmpp_3m_sde_gpu', 'karras'), 'sd15': (808, 25, 0, 'dpmpp_3m_sde_gpu', 'karras') },
	'3': { 'sdxl': (1152, 25, int(25 / 2.5), 'dpmpp_3m_sde_gpu', 'karras'), 'sd15': (808, 35, 0, 'dpmpp_3m_sde_gpu', 'karras') },
	'4': { 'sdxl': (1152, 35, int(35 / 2.5), 'dpmpp_3m_sde_gpu', 'karras'), 'sd15': (808, 50, 0, 'dpmpp_3m_sde_gpu', 'karras') },
}

# Detail
options['detail'] = {
	"Low Detail": 0.95,
	"Normal Detail": 1.0,
	"High Detail": 1.05,
	"Ultra Detail": 1.1,
}
title['detail'] = "Detail"
default['detail'] = "Normal Detail"

# Ref Mode
options["ref_mode"] = {
	"Ref All": ("ip_adapter",  { "sdxl": "ip-adapter-plus_sdxl_vit-h.bin", "sd15": "ip-adapter_sd15_plus.pth" }),
	"Ref Content": ("content", { "sdxl": None, "sd15": None }),
	"Ref Stuct": ("canny", { "sdxl": "sai_xl_canny_128lora.safetensors", "sd15": "control_v11p_sd15_canny.pth" }),
    "Ref Depth": ("depth_leres", { "sdxl": "sai_xl_depth_128lora.safetensors", "sd15": "control_v11f1p_sd15_depth.pth" }),
	# "Ref Style": ("style", { "sdxl": None, "sd15": None }),
    "Ref Face": ("ip_adapter_face", { "sdxl": "ip-adapter-plus-face_sdxl_vit-h.bin", "sd15": "ip-adapter-plus-face_sd15.bin" }),
	"Ref Pose": ("dwpose", { "sdxl": "t2i-adapter_xl_openpose.safetensors", "sd15": "control_v11p_sd15_openpose.pth" }), # thibaud_xl_openpose_256lora.safetensors
	# "Ref Hue": ("shuffle", { "sdxl": "sai_xl_recolor_128lora.safetensors", "sd15": "control_v11e_sd15_shuffle.pth" }),
	"Base Image": ("base_image", ""),
}
default['ref_mode'] = "Ref All"

# Simpler
options['simpler'] = { "Default Simpler": "" } | load_ui_config("simpler")
title['simpler'] = "Sampler"
default['simpler'] = "Default Simpler"

# scheduler
options['scheduler'] = { "Default Scheduler": "" } | load_ui_config("scheduler")
title['scheduler'] = "Scheduler"
default['scheduler'] = "Default Scheduler"

# CFG
options['cfg'] = (0.5, 20, 0.5)
mul['cfg'] = 1
title['cfg'] = "CFG"
default['cfg'] = 7

# CFG End At
options['cfg_to'] = (0.5, 20, 0.5)
mul['cfg_to'] = 1
title['cfg_to'] = "CFG Work-up to"
default['cfg_to'] = 12

# File Format
options['file_format'] = {
	"JPEG": "jpeg",
	"PNG": "png",
}
title['file_format'] = "File Format"
default['file_format'] = "JPEG"

# VAE Mode
options['single_vae'] = {
	"GPU": True,
	"CPU and GPU": False,
}
title['single_vae'] = "VAE Mode"
default['single_vae'] = "CPU and GPU"

# Step Scale
options['step_scale'] = (0, 200, 5)
mul['step_scale'] = 0.01
title['step_scale'] = "Steps Scale %"
default['step_scale'] = 100

options['refiner_step_scale'] = (0, 200, 5)
mul['refiner_step_scale'] = 0.01
title['refiner_step_scale'] = "Refiner Steps Scale %"
default['refiner_step_scale'] = 100

# Sample Difference
options['subseed_strength'] = (0, 100, 1)
mul['subseed_strength'] = 0.01
title['subseed_strength'] = "Subseed Strength %"
default['subseed_strength'] = 2

# Clip Skip
options['clip_skip'] = (0, 10, 1)
mul['clip_skip'] = -1
title['clip_skip'] = "Clip Skip"
default['clip_skip'] = 5

# Denoise
options['denoise'] = (0, 100, 1)
mul['denoise'] = 0.01
title['denoise'] = "Denoise Strength %"
default['denoise'] = 80

# Sample Pagesize
options['sample_pagesize'] = (128, 1024, 128)
mul['sample_pagesize'] = 1
title['sample_pagesize'] = "Sample Pagesize"
default['sample_pagesize'] = 256

# Negative
options['recommend_negative'] = load_ui_config("negative")
title['recommend_negative'] = "Negative"
default['recommend_negative'] = ["Default", "Watermark", "Ugly body"]

options['prompt_negative_weight'] = (50, 150, 5)
mul['prompt_negative_weight'] = 0.01
title['prompt_negative_weight'] = "Negative Weight %"
default['prompt_negative_weight'] = 100

# Options Weight
for x in ['mc', 'style', 'view', 'emo', 'location', 'weather', 'hue']:
	k = f'{x}_weight'
	options[k] = (50, 150, 5)
	mul[k] = 0.01
	title[k] = f"{title[x]} Weight %"
	default[k] = 100
default['style_weight'] = 110
