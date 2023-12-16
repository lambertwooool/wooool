import warnings
import cv2
import numpy as np
from PIL import Image
from modules.util import norm255

class CannyDetector:
    def __call__(self, input_image, low_threshold=64, high_threshold=128):
        detected_map = self.canny_pyramid(input_image, low_threshold, high_threshold)
            
        return detected_map
    
    def canny_pyramid(self, x, low_threshold, high_threshold):
        # For some reasons, SAI's Control-lora Canny seems to be trained on canny maps with non-standard resolutions.
        # Then we use pyramid to use all resolutions to avoid missing any structure in specific resolutions.

        color_canny = self.pyramid_canny_color(x, low_threshold, high_threshold)
        result = np.sum(color_canny, axis=2)

        return norm255(result, low=1, high=99).clip(0, 255).astype(np.uint8)
    
    def pyramid_canny_color(self, x: np.ndarray, low_threshold, high_threshold):
        assert isinstance(x, np.ndarray)
        assert x.ndim == 3 and x.shape[2] == 3

        H, W, C = x.shape
        acc_edge = None

        for k in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            Hs, Ws = int(H * k), int(W * k)
            small = cv2.resize(x, (Ws, Hs), interpolation=cv2.INTER_AREA)
            edge = self.centered_canny_color(small, low_threshold, high_threshold)
            if acc_edge is None:
                acc_edge = edge
            else:
                acc_edge = cv2.resize(acc_edge, (edge.shape[1], edge.shape[0]), interpolation=cv2.INTER_LINEAR)
                acc_edge = acc_edge * 0.75 + edge * 0.25

        return acc_edge

    def centered_canny(self, x: np.ndarray, canny_low_threshold, canny_high_threshold):
        assert isinstance(x, np.ndarray)
        assert x.ndim == 2 and x.dtype == np.uint8

        y = cv2.Canny(x, int(canny_low_threshold), int(canny_high_threshold))
        y = y.astype(np.float32) / 255.0
        return y

    def centered_canny_color(self, x: np.ndarray, low_threshold, high_threshold):
        assert isinstance(x, np.ndarray)
        assert x.ndim == 3 and x.shape[2] == 3

        result = [self.centered_canny(x[..., i], low_threshold, high_threshold) for i in range(3)]
        result = np.stack(result, axis=2)
        return result
