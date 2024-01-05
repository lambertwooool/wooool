import cv2
import numpy as np

from modules.util import HWC3


class TileDetector:
    def __call__(self, input_image=None, pyrUp_iters=3):
        H, W, _ = input_image.shape
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
        detected_map = cv2.resize(input_image, (W // (2 ** pyrUp_iters), H // (2 ** pyrUp_iters)))
        detected_map = HWC3(detected_map)

        for _ in range(pyrUp_iters):
            detected_map = cv2.pyrUp(detected_map)

        return detected_map
