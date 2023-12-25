import os
import json
import re
import threading
import time
import requests
import modules.paths
from modules import util

# https://github.com/civitai/sd_civitai_extension/blob/main/civitai/lib.py
base_urls = ["https://civitai.com/api/v1", "https://civitai.work/api/v1", "https://civitai.tech/api/v1"]
re_host = re.compile(r"^http[s]?://([\w\.]+)(/|$)")
base_url = None
base_host = None

api = {
    "models": "/models",
    "model-versions-hash": "/model-versions/by-hash/",
    "model-versions-id": "/model-versions/",
    "creators": "/creators",
    "tags": "/tags",
}

suffix = ".civitai.info"

def speed_test(url, user_agent, timeout):
    global base_url, base_host
    try:
        response = requests.head(f"{url}{api['models']}", headers={"User-Agent": user_agent}, timeout=timeout)
        if response.ok:
            if base_url is None:
                base_url = url
                base_host = re.search(re_host, base_url).groups(0)[0]
                return
    except:
        # print(f"{url} error.")
        pass

def refresh_base_url():
    global base_url
    user_agent = "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko)"
    timeout = 5
    start_time = time.perf_counter()

    for url in base_urls:
        threading.Thread(target=speed_test, args=(url, user_agent, timeout)).start()

    while base_url is None and (time.perf_counter() - start_time) < timeout:
        time.sleep(0.1)
    
    if base_url is None:
        print("no civitai url activate.")
        # base_url = base_urls[0]
        base_url = None
    else:
        print(f"civitai url is {base_url}")

threading.Thread(target=refresh_base_url).start()

def get_url(url):
    if base_host is not None:
        # host = re.search(re_host, url).groups(0)[0]
        url = url.replace("civitai.com", base_host)
    return url

def exists_info(model_path):
    config_file = f"{model_path}{suffix}"
    return os.path.exists(config_file)

def get_model_versions(
        model_path: str
) -> dict:
    config_file = f"{model_path}{suffix}"
    if os.path.exists(config_file):
        data = util.load_json(config_file)
    else:
        if base_url is not None:
            hash_value = util.gen_file_sha256(model_path)[:10]
            url = f"{base_url}{api['model-versions-hash']}{hash_value}"

            data = util.load_url(url)

            with open(config_file, "w") as file:
                file.write(json.dumps(data, indent=4))
        else:
            data = None

    if data and not data.get("model_homepage"):
        model_id = data["modelId"]
        version_id = data["id"]
        url = get_url(f"https://civitai.com/models/{model_id}?modelVersionId={version_id}")
        data["model_homepage"] = url

    return data