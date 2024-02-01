import os
from modules import util, paths

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
default["base_model"] = f"sd_xl_base_1.0_0.9vae (sdxl)"

# Main Character
options["mc"] = load_ui_config("mc") | {
	"Other": ""
} 
title["mc"] = "Main Character"
default["mc"] = "Girl"

# Aspect ratio
options["ratios"] = load_ui_config("aspect_ratio")
title["ratios"] = "Aspect ratio"
default["ratios"] = "4:3"

options["resize_ratios"] = options["ratios"]
title["resize_ratios"] = "Aspect ratio"
default["resize_ratios"] = "1:1"

# Style
options["style"] = load_ui_config("style")
title["style"] = "Style"
default["style"] = "Cinematic & Photographic"

# View
options["view"] = load_ui_config("view")
title["view"] = "View"
default["view"] = "Default view"

# Emo
options["emo"] = load_ui_config("emo")
title["emo"] = "Emo"
default["emo"] = "Happy"

# Location
options["location"] = load_ui_config("location")
title["location"] = "Location"
default["location"] = "Any Where"

# Weather
options["weather"] = load_ui_config("weather")
title["weather"] = "Weather"
default["weather"] = "Any Weather"

# Lighting
options["hue"] = load_ui_config("lighting")
title["hue"] = "Lighting"
default["hue"] = "Natural lighting"

options["pic_num"] = load_ui_config("pic_num")
title["pic_num"] = "Sample Settings"
default["pic_num"] = "2 Samples"

# Ref Num
options["ref_num"] = (1, 5, 1)
title["ref_num"] = "Ref Count"
default["ref_num"] = 3

# Lora Num
options["lora_num"] = {
	"3 LoRA": 3,
	"6 LoRA": 6,
}
title["lora_num"] = "Lora Count"
default["lora_num"] = "3 LoRA"

# Quality
options["quality"] = {
    # "LCM Generate": 5,
	"Ex-Fast Generate": 1,
	"Fast Generate": 2,
	"High Quality": 3,
	"Ultra Quality": 4
}
title["quality"] = "Quality"
default["quality"] = "Fast Generate"

options["quality_setting"] = {
    # "5": { "sdxl": (1024, 5, 0, "lcm", "lcm"), "sd15": (768, 5, 0, "lcm", "lcm") },
	"1": { "sdxl": (1184, 16, 0, "dpmpp_2m", "karras"), "sd15": (768, 16, 0, "dpmpp_2m", "karras") },
	"2": { "sdxl": (1184, 25, 0, "dpmpp_3m_sde_gpu", "karras"), "sd15": (768, 25, 0, "dpmpp_3m_sde_gpu", "karras") },
	"3": { "sdxl": (1184, 25, int(25 / 2.5), "dpmpp_3m_sde_gpu", "karras"), "sd15": (768, 25, int(25 / 2.5), "dpmpp_3m_sde_gpu", "karras") },
	"4": { "sdxl": (1184, 35, int(35 / 2.5), "dpmpp_3m_sde_gpu", "karras"), "sd15": (768, 35, int(35 / 2.5), "dpmpp_3m_sde_gpu", "karras") },
}

# Ref Mode
options["ref_mode"] = {
	"Ref All": (	"ip_adapter",
			 		{	"sdxl": "ip-adapter-plus_sdxl_vit-h.bin", "sd15": "ip-adapter_sd15_plus.pth",
	   					"keyword": [r"ip adapter.*plus(?! face)", r"tile"],
						"annotator": ["ip_adapter", "tile"]
					}),
	"Ref Content": ("content", { "sdxl": None, "sd15": None, "keyword": None }),
	"Ref Stuct": (	"canny",
			   		{	"sdxl": "sai_xl_canny_128lora.safetensors", "sd15": "control_v11p_sd15_canny.pth",
						"keyword": [r"canny", r"^(?!.*t2i)(?=.*lineart).*$", r".*t2i.*lineart", r"sketch", r"mlsd", r"scribble", r"softedge"],
						"annotator": ["canny", ["lineart_coarse", "lineart_realistic", "lineart_anime", "lineart_anime_denoise"], ["lineart_coarse,invert", "lineart_realistic,invert", "lineart_anime,invert", "lineart_anime_denoise,invert"], ["scribble_hed", "softedge_hed", "teed"], ["mlsd", "canny"], "scribble_hed", ["softedge_hed", "teed"] ]
					}),
    "Ref Depth": (	"depth_leres",
				  	{ 	"sdxl": "sai_xl_depth_128lora.safetensors", "sd15": "control_v11f1p_sd15_depth.pth",
						"keyword": [r"^(?!.*t2i)(?!.*depth anything)(?=.*depth).*$", r"depth midas", r"depth zoe", r"depth anything"],
						"annotator": [["depth_leres", "depth_midas", "depth_zoe", "depth_anything"], "depth_midas", "depth_zoe", ["depth_leres,colormap", "depth_midas,colormap", "depth_zoe,colormap", "depth_anything,colormap"]]
					}),
    "Ref Face": (	"ip_adapter_face",
				 	{	"sdxl": "ip-adapter-plus-face_sdxl_vit-h.bin", "sd15": "ip-adapter-plus-face_sd15.bin",
	   					"keyword": [r"(ip adapter plus.*face|ip adapter face.*plus)", r"^(?!.*xl)(?!.*animal)(?=.*openpose)(?!.*xl).*$"],
						"annotator": ["ip_adapter_face", "dwpose_face"]
					}),
	"Ref Pose": (	"dwpose",
			  		{ 	"sdxl": "thibaud_xl_openpose_256lora.safetensors", "sd15": "control_v11p_sd15_openpose.pth",
						"keyword": [r"^(?!.*animal)(?=.*openpose).*$", r".*animal.*openpose"],
						"annotator": ["dwpose", "animal_pose"]
					}),
	"Ref Color": (	"shuffle",
			  		{ 	"sdxl": None, "sd15": "control_v11e_sd15_shuffle.pth",
						"keyword": [r"shuffle", r"\bcolor\b"],
						"annotator": ["shuffle", "color"]
					}),
	"Ref Color": (	"shuffle",
			  		{ 	"sdxl": None, "sd15": "control_v11e_sd15_shuffle.pth",
						"keyword": [r"shuffle", r"\bcolor\b"],
						"annotator": ["shuffle", "color"]
					}),
	"Ref Brightness": (	"brightness",
			  		{ 	"sdxl": None, "sd15": "control_v1p_sd15_brightness.safetensors",
						"keyword": [r"(brightness|qrcode)"],
						"annotator": [["binary", "binary,invert"]]
					}), 
	"Ref Segment": (	"oneformer",
			  		{ 	"sdxl": None, "sd15": "control_v11p_sd15_seg.pth",
						"keyword": [r"\bseg\b"],
						"annotator": [["oneformer", "segment_anything", "remove_bg,anime_segmentation"]]
					}),
	"Ref Normalbae": (	"normal_bae",
			  		{ 	"sdxl": None, "sd15": "control_v11p_sd15_normalbae.pth",
						"keyword": [r"normalbae"],
						"annotator": ["normal_bae"]
					}), 
	"Base Image": ("base_image", { "sdxl": None, "sd15": None, "keyword": None }),
	"Recolor":	(	"default",
			 		{	"sdxl": "sai_xl_recolor_128lora.safetensors", "sd15": "ioclab_sd15_recolor.safetensors",
						"keyword": [r"recolor"],
						"annotator": ["default"]
					}),
	"Others": (		"default",
					{	"sdxl": None, "sd15": None,
	  					"keyword": [r"^(?!.*(ip adapter|tile|canny|lineart|mlsd|scribble|softedge|sketch|depth|openpose|normalbae|shuffle|brightness|qrcode|\bseg\b|\bcolor\b|recolor)).*$"],
						"annotator": ["default"]
					 }),
}
default["ref_mode"] = "Ref All"

# Sampler
options["sampler"] = { "Default Sampler": "" } | load_ui_config("sampler")
title["sampler"] = "Sampler"
default["sampler"] = "Default Sampler"

# scheduler
options["scheduler"] = { "Default Scheduler": "" } | load_ui_config("scheduler")
title["scheduler"] = "Scheduler"
default["scheduler"] = "Default Scheduler"

# CFG
options["cfg_scale"] = (0.5, 20, 0.5)
mul["cfg_scale"] = 1
title["cfg_scale"] = "CFG"
default["cfg_scale"] = 7

# CFG End At
options["cfg_scale_to"] = (0.5, 20, 0.5)
mul["cfg_scale_to"] = 1
title["cfg_scale_to"] = "CFG Work-up to"
default["cfg_scale_to"] = 9

# File Format
options["file_format"] = {
	"JPEG": "jpeg",
	"PNG": "png",
}
title["file_format"] = "File Format"
default["file_format"] = "JPEG"

# VAE Mode
options["single_vae"] = {
	"GPU": True,
	"CPU and GPU": False,
}
title["single_vae"] = "VAE Mode"
default["single_vae"] = "CPU and GPU"

# Base Step
options["step_base"] = (0, 80, 1)
mul["step_base"] = 1
title["step_base"] = "Base Steps"
default["step_base"] = 25

# Refiner Step
options["step_refiner"] = (0, 50, 1)
mul["step_refiner"] = 1
title["step_refiner"] = "Refiner Steps"
default["step_refiner"] = 0

# Width
options["image_width"] = (512, 2048, 8)
mul["image_width"] = 1
title["image_width"] = "Image Width"
default["image_width"] = 1152

# Height
options["image_height"] = options["image_width"]
mul["image_height"] = 1
title["image_height"] = "Image Height"
default["image_height"] = 1152

# Sample Difference
options["subseed_strength"] = (0, 100, 1)
mul["subseed_strength"] = 0.01
title["subseed_strength"] = "Subseed Strength %"
default["subseed_strength"] = 100

# Clip Skip
options["clip_skip"] = (0, 10, 1)
mul["clip_skip"] = -1
title["clip_skip"] = "Clip Skip"
default["clip_skip"] = 5

# Denoise
options["denoise"] = (0, 100, 1)
mul["denoise"] = 0.01
title["denoise"] = "Denoise Strength %"
default["denoise"] = 100

# Sample Pagesize
options["sample_pagesize"] = (128, 1024, 128)
mul["sample_pagesize"] = 1
title["sample_pagesize"] = "Sample Pagesize"
default["sample_pagesize"] = 256

# Negative
options["recommend_negative"] = load_ui_config("negative")
title["recommend_negative"] = "Negative"
default["recommend_negative"] = ["Default", "Watermark", "Ugly body"]

options["prompt_negative_weight"] = (50, 150, 5)
mul["prompt_negative_weight"] = 0.01
title["prompt_negative_weight"] = "Negative Weight %"
default["prompt_negative_weight"] = 100

# Options Weight
for x in ["mc", "style", "view", "emo", "location", "weather", "hue"]:
	k = f"{x}_weight"
	options[k] = (50, 150, 5)
	mul[k] = 0.01
	title[k] = f"{title[x]} Weight %"
	default[k] = 100
default["style_weight"] = 110

# Detail
options["detail"] = (0, 120, 5)
mul["detail"] = 0.01
title["detail"] = "Detail Enhance %"
default["detail"] = 0

# More Art Weight
options["more_art_weight"] = (0, 100, 5)
mul["more_art_weight"] = 0.01
title["more_art_weight"] = "More Art Weight %"
default["more_art_weight"] = 0

# Vary Custom Area
options["vary_custom_area"] = {
	"Mask": True,
	"UnMask": False,
}
title["vary_custom_area"] = "Vary Area"
default["vary_custom_area"] = "Mask"

# Vary Custom Strength
options["vary_custom_strength"] = (10, 100, 5)
mul["vary_custom_strength"] = 0.01
title["vary_custom_strength"] = "Vary Strength %"
default["vary_custom_strength"] = 60

# Zoom Custom
options["zoom_custom"] = (150, 400, 50)
mul["zoom_custom"] = 0.01
title["zoom_custom"] = "Zoom %"
default["zoom_custom"] = 200

# Zoom Denoise
options["zoom_denoise"] = (0, 100, 1)
mul["zoom_denoise"] = 0.01
title["zoom_denoise"] = "Denoise %"
default["zoom_denoise"] = 98

# Zoom Blur Alpha
options["zoom_blur_alpha"] = (0, 100, 1)
mul["zoom_blur_alpha"] = 0.01
title["zoom_blur_alpha"] = "Blur alpha %"
default["zoom_blur_alpha"] = 50

# Refiner Denoise
options["refiner_denoise"] = (0, 100, 5)
mul["refiner_image_denoise"] = mul["refiner_face_denoise"] = 0.01
title["refiner_denoise"] = "Vary Strength %"
default["refiner_denoise"] = 30

# Refiner Detail
options["refiner_detail"] = (-300, 300, 5)
mul["refiner_image_detail"] = mul["refiner_face_detail"] = 0.01
title["refiner_detail"] = "Refiner Detail %"
default["refiner_detail"] = 30

# Refiner Face index
options["refiner_face_index"] = {
	"Max": 0,
	"2nd": 1,
	"3rd": 2,
}
title["refiner_face_index"] = "Specify Face"
default["refiner_face_index"] = "Max"

# Upscale Model
options["upscale_model"] = {
	"Non-use model": "",
} | { os.path.split(x)[1]: os.path.split(x)[1] for x in util.list_files(paths.upscale_models_path, ext=["pt", "pth", "bin"]) }
title["upscale_model"] = "Upscale Model"
default["upscale_model"] = "4x-UltraSharp.pth"

# Upscale Factor
options["upscale_factor"] = (100, 400, 50)
mul["upscale_factor"] = 0.01
title["upscale_factor"] = "Upscale Factor %"
default["upscale_factor"] = 200

# Upscale Origin Visibility
options["upscale_origin"] = (0, 100, 0)
mul["upscale_origin"] = 0.01
title["upscale_origin"] = "Origin Visibility %"
default["upscale_origin"] = 0

