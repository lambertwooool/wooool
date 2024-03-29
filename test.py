import cv2
# from modules.controlnet.hed import HEDdetector
# from modules.controlnet.remove_bg import RemoveBackgroundDetector
from modules.controlnet.bria_rmbg import BriaRemoveBackgroundDetector
from modules import devices
from modules.model import ops

def main():
    # img = cv2.imread("./web/css/logo-t.png")
    img = cv2.imread("./samples.jpeg")

    # hed = HEDdetector()
    # det = hed(img, safe=False, scribble=False)

    # rmbg = RemoveBackgroundDetector()
    rmbg = BriaRemoveBackgroundDetector()
    det = rmbg(img)

    cv2.imwrite("./temp.jpg", det)

    with ops.auto_cast():
        pass

if __name__ == "__main__":
    main()

