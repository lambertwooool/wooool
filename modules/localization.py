import json
import os
import modules.paths
from modules.util import load_json, list_files, webpath

def localization_json(current_localization_name: str) -> str:
    fn = f"{modules.paths.localization_path}/{current_localization_name}.json"
    data = load_json(fn)

    return data

def urls():
    files = {}
    for fn_path in list_files(modules.paths.localization_path, "json"):
        lang = os.path.splitext(os.path.split(fn_path)[-1])[0]
        with open(fn_path, "r", encoding="utf8") as file:
            data = json.load(file)
        files[lang] = { "name":data.get("@lang", lang.upper()), "url":webpath(fn_path) }
    return files