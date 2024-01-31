import cv2
from modules.util import HWC3, image_pad

def cv2_resize_shortest_edge(image, size):
    h, w = image.shape[:2]
    if h < w:
        new_h = size
        new_w = int(round(w / h * size))
    else:
        new_w = size
        new_h = int(round(h / w * size))
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image

def apply_color(img, res=512):
    img = cv2_resize_shortest_edge(img, res)
    h, w = img.shape[:2]

    input_img_color = cv2.resize(img, (w//64, h//64), interpolation=cv2.INTER_CUBIC)  
    input_img_color = cv2.resize(input_img_color, (w, h), interpolation=cv2.INTER_NEAREST)
    return input_img_color

#Color T2I like multiples-of-64, upscale methods are fixed.
class ColorDetector:
    def __call__(self, input_image):
        input_image = HWC3(input_image)
        detected_map = HWC3(apply_color(input_image))
            
        return detected_map

class GrayDetector:
    def __call__(self, img):
        eps = 1e-5
        X = img.astype(np.float32)
        r, g, b = X[:, :, 0], X[:, :, 1], X[:, :, 2]
        kr, kg, kb = [random.random() + eps for _ in range(3)]
        ks = kr + kg + kb
        kr /= ks
        kg /= ks
        kb /= ks
        Y = r * kr + g * kg + b * kb
        Y = np.stack([Y] * 3, axis=2)
        return Y.clip(0, 255).astype(np.uint8)

class ColorMapDetector:
    def __call__(self, input_image):
        return cv2.applyColorMap(input_image, cv2.COLORMAP_INFERNO)[:, :, ::-1]