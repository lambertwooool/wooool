import os
import re
import numpy as np
import modules.paths
from modules import util, civitai

keywords = {
    "__test_keywords_1": ("__lora_1", 0.8, ""),
}

lora_files = {}
re_lora = re.compile(r"<lora:([^:>]+)(:([\d\.]+))?>", re.I)

def get_list():
    global lora_files

    files = util.list_files(modules.paths.lorafile_path, "safetensors", search_subdir=True)
    files = { re.sub(r"^(\w+)\\", r"[ \1 ] ", os.path.relpath(os.path.splitext(x)[0], modules.paths.lorafile_path)) : x for x in files }
    lora_files = files
    
    return files

lora_files = get_list()

def get_lora_path(lora_name):
    return ([file for file in lora_files.values() if os.path.split(file)[1] == lora_name] + [None])[0]

def get_info(lora_name):
    lora_file = os.path.join(modules.paths.lorafile_path, lora_files[lora_name])
    info = civitai.get_model_versions(lora_file)
    if info is not None:
        preview_url = info["images"][0]["url"]
        preview_path = f"{lora_file}.preview"
        if not os.path.exists(preview_path):
            util.download_url_to_file(civitai.get_url(preview_url), preview_path)
        info["preview_image"] = preview_path

    return info

def parse_block(prompt):
    lora_prompt = prompt
    loras = []
    while True:
        lora = re.search(re_lora, lora_prompt)
        if lora is not None:
            start, end = lora.span()
            lora_prompt = lora_prompt[:start] + lora_prompt[end:]
            lora_name, _, lora_weight = lora.groups()
            lora_weight = float(lora_weight)
            
            loras.append((lora_name, lora_weight, ""))
        else:
            break

    return lora_prompt, loras

def keyword_parse(prompt):
    loras = {}
    
    for kw in sorted(keywords, key=len, reverse=True):
        kw_re = re.sub(r"[\s+-]+", "[\\\\s+-]", kw)
        if re.search(kw_re, prompt):
            lora_name, weight, trained_words = keywords[kw]
            prompt, count = re.subn(kw_re, "", prompt)
            loras[kw] = (lora_name, weight, trained_words, count)

    return loras

def keyword_loras(prompt):
    loras = []
    for kw, kw_lora in keyword_parse(prompt).items():
        lora_name, lora_weight, lora_trained_words, count = kw_lora
        lora_weight_mul = pow(1.1, count - 1)
        
        loras.append((lora_name, lora_weight * lora_weight_mul, lora_trained_words))

    return loras

def reduce(loras):
    lora_dict = {}

    for lo_name, lo_weight, lo_trained_words in loras:
        if lora_dict.get(lo_name):
            lora_dict[lo_name][0].append(lo_weight)
            if lo_trained_words and lo_trained_words not in lora_dict[lo_name][1]:
                lora_dict[lo_name][1].append(lo_trained_words)
        else:
            lora_dict[lo_name] = [[lo_weight], [lo_trained_words] if lo_trained_words else []]

    loras = [(k, np.mean(v[0]), ",".join(v[1]).strip(",")) for k, v in lora_dict.items()]

    return loras

def remove_prompt_lora(prompt, name):
    lora_prompt = prompt
    return_prompt = ""
    while True:
        lora = re.search(re_lora, lora_prompt)
        if lora is not None:
            lora_name, _, lora_weight = lora.groups()
            start, end = lora.span()
            if lora_name == name:
                return_prompt = lora_prompt[:start]
            else:
                return_prompt = lora_prompt[:end]
                
            lora_prompt = lora_prompt[end:]
        else:
            return_prompt += lora_prompt
            break
    
    return return_prompt