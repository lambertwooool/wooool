import os
import re
import modules.paths
from modules import util, civitai

keywords = {
    "__test_keywords_1": ("__lora_1", 0.8, ""),
}

lora_files = {}

def get_list():
    global lora_files

    files = util.list_files(modules.paths.lorafile_path, "safetensors", search_subdir=True)
    files = { re.sub(r"^(\w+)\\", r"[ \1 ] ", os.path.relpath(os.path.splitext(x)[0], modules.paths.lorafile_path)) : x for x in files }
    lora_files = files
    
    return files

lora_files = get_list()

def get_info(lora_name):
    lora_file = os.path.join(modules.paths.lorafile_path, lora_files[lora_name])
    info = civitai.get_model_versions(lora_file)
    if info is not None:
        preview_url = info["images"][0]["url"]
        preview_path = f"{lora_file}.preview"
        if not os.path.exists(preview_path):
            util.download_url_to_file(civitai.get_image_url(preview_url), preview_path)
        info["preview_image"] = preview_path

    return info

def keyword_parse(prompt):
    loras = {}
    
    for kw in sorted(keywords, key=len, reverse=True):
        kw_re = re.sub(r"[\s+-]+", "[\\\\s+-]", kw)
        if re.search(kw_re, prompt):
            lora_name, weight, trained_words = keywords[kw]
            prompt, count = re.subn(kw_re, "", prompt)
            loras[kw] = (lora_name, weight, trained_words, count)

    return loras

def keyword_loras(prompt, loras):
    for kw, kw_lora in keyword_parse(prompt).items():
        lora_name, lora_weight, lora_trained_words, count = kw_lora
        lora_weight_mul = pow(1.1, count - 1)
        lo_found = False

        for i, lo in enumerate(loras):
            if kw_lora[0] == lo[0]:
                lo_name, lo_weight, lo_trained_words = loras.pop(i)
                loras.insert(i, (lo_name, lo_weight * lora_weight_mul, lo_trained_words))
                lo_found = True
                break

        if not lo_found:
            loras.append((lora_name, lora_weight * lora_weight_mul, lora_trained_words))

    return loras