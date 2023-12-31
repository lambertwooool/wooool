import os
from .patch_manager import patch, undo, original
from modules import util

import modules.paths
import modules.model.model_helper
import modules.gfpgan_model
import insightface.model_zoo
import modules.wd14tagger

download_urls = util.load_json(os.path.join("./configs", "download.json"))

class DownloadPatches:
    def __init__(self):
        self.load_torch_file = patch(__name__, modules.model.model_helper, 'load_torch_file', load_torch_file)
        self.GFPGan_get_model_path = patch(__name__, modules.gfpgan_model.GFPGan, 'get_model_path', GFPGan_get_model_path)
        self.wd14_tag = patch(__name__, modules.wd14tagger, 'tag', wd14_tag)
        self.model_zoo_get_model = patch(__name__, insightface.model_zoo, 'get_model', model_zoo_get_model)
        self.model_zoo_get_model_inner = patch(__name__, insightface.model_zoo.model_zoo, 'get_model', model_zoo_get_model)

    def undo(self):
        self.load_torch_file = undo(__name__, modules.model.model_helper, 'load_torch_file')
        self.GFPGan_get_model_path = undo(__name__, modules.gfpgan_model.GFPGan, 'get_model_path')
        self.wd14_tag = undo(__name__, modules.wd14tagger, 'tag')
        self.model_zoo_get_model = undo(__name__, insightface.model_zoo, 'get_model')
        self.model_zoo_get_model_inner = patch(__name__, insightface.model_zoo.model_zoo, 'get_model')

def load_or_download_file(dest):
    filename = os.path.split(dest)[1]
    if not os.path.exists(dest) and filename in download_urls:
        url = download_urls[filename]
        util.download_url_to_file(url, dest, download_chunk_size=1*(1024**2))
    
    return os.path.exists(dest)

def load_torch_file(*args, **kwargs):
    load_or_download_file(args[0])
    return original(__name__, modules.model.model_helper, "load_torch_file")(*args, **kwargs)

def GFPGan_get_model_path(self, *args, **kwargs):
    model_path = modules.paths.face_models_path
    load_or_download_file(os.path.join(model_path, args[0]))
    return original(__name__, self.__class__, "get_model_path")(self, *args, **kwargs)

def model_zoo_get_model(*args, **kwargs):
    model_path = modules.paths.face_models_path
    kwargs["root"] = model_path
    load_or_download_file(os.path.join(model_path, args[0]))
    return original(__name__, insightface.model_zoo, "get_model")(*args, **kwargs)

def wd14_tag(*args, **kwargs):
    model_name = kwargs.get("model_name", "wd-v1-4-moat-tagger-v2.onnx")
    model_path = os.path.join(modules.paths.wd14tagger_path, model_name)
    csv_name = os.path.splitext(model_name)[0] + ".csv"
    csv_path = os.path.join(modules.paths.wd14tagger_path, csv_name)
    load_or_download_file(model_path)
    load_or_download_file(csv_path)
    return original(__name__, modules.wd14tagger, "tag")(*args, **kwargs)

