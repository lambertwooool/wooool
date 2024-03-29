import os
import json

from modules import util

config_path = os.path.join("./configs", "user_paths.json")
config_dict = util.load_json(config_path)
# https://hf-mirror.com/

preset = None

def get_path(key, default_value):
    global config_dict
    v = config_dict.get(key, None)
    if isinstance(v, str) and os.path.exists(v) and os.path.isdir(v):
        return v
    else:
        dp = os.path.abspath(os.path.join(os.path.dirname(__file__), default_value))
        os.makedirs(dp, exist_ok=True)
        config_dict[key] = dp
        return dp

modelfile_path = get_path('modelfile_path', '../models/checkpoints/')
unet_path = get_path('unet_path', '../models/unet/')
lorafile_path = get_path('lorafile_path', '../models/loras/')
embeddings_path = get_path('embeddings_path', '../models/embeddings/')
vae_approx_path = get_path('vae_approx_path', '../models/vae_approx/')
upscale_models_path = get_path('upscale_models_path', '../models/upscale/')
inpaint_models_path = get_path('inpaint_models_path', '../models/inpaint/')
controlnet_models_path = get_path('controlnet_models_path', '../models/controlnet/')
clip_vision_models_path = get_path('clip_vision_models_path', '../models/clip_vision/')
annotator_models_path = get_path('annotator_models_path', '../models/annotator/')
face_models_path = get_path('face_models_path', '../models/faces/')
wd14tagger_path = get_path('wd14tagger_path', '../models/wd14tagger/')
layerd_path = get_path('layerd_path', '../models/layerd/')

temp_outputs_path = get_path('temp_outputs_path', '../outputs/')
scripts_path = get_path('scripts_path', '../web/javascript/')
css_path = get_path('css_path', '../web/css/')
localization_path = get_path('localization_path', '../configs/localization/')
sd_style_path = get_path('sd_style_path', '../configs/sd_styles/')
wildcard_path = get_path('wildcard_path', '../configs/wildcard/')

if preset is None:
    # Do not overwrite user config if preset is applied.
    with open(config_path, "w", encoding="utf-8") as json_file:
        json.dump(config_dict, json_file, indent=4)

os.makedirs(temp_outputs_path, exist_ok=True)
