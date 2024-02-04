import cv2
import numpy as np
from modules import insightface_model
from modules.util import draw_kps
from modules.util import HWC3, image_pad

class KpsDetector:
    def __call__(self, input_image):
        detected_map, remove_pad = image_pad(input_image)

        analysis = insightface_model.Analysis(name="antelopev2")
        faces = analysis(detected_map)

        detected_map = draw_kps(detected_map, faces[0].kps)
            
        return detected_map
