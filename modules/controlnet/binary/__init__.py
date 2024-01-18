import warnings
import cv2
import numpy as np
from modules.util import HWC3, image_pad

class BinaryDetector:
    def __call__(self, input_image, bin_threshold=0):
        detected_map, remove_pad = image_pad(input_image)

        img_gray = cv2.cvtColor(detected_map, cv2.COLOR_RGB2GRAY)
        if bin_threshold == 0 or bin_threshold == 255:
        # Otsu's threshold
            otsu_threshold, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            print("Otsu threshold:", otsu_threshold)
        else:
            _, img_bin = cv2.threshold(img_gray, bin_threshold, 255, cv2.THRESH_BINARY_INV)
        
        detected_map = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB)
        detected_map = HWC3(remove_pad(255 - detected_map))
            
        return detected_map
