import os
from typing import Any
import insightface
from insightface.app import FaceAnalysis

import modules.paths

insightface.utils.storage.BASE_REPO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7"

class Analysis():
    def __init__(self, name="buffalo_l"):
        model = FaceAnalysis(
            name=name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            root=modules.paths.face_models_path,
            download=True,
            download_zip=True)
        model.prepare(ctx_id=0, det_size=(640, 640))

        self.model = model
    
    def __call__(self, input_image):
        faces = self.model.get(input_image)
        faces = sorted(faces, key = lambda x : ((x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1])), reverse=True)
        return faces

class Swapper():
    def __init__(self, model_name="inswapper_128.onnx"):
        model_path = os.path.join(modules.paths.face_models_path, model_name)
        self.model = insightface.model_zoo.get_model(model_path, root=modules.paths.face_models_path)
        
    def __call__(self, input_image, target_face, source_face, **kwargs):
        return self.model.get(input_image, target_face, source_face, **kwargs)