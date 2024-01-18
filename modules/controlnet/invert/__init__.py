
class InvertDetector:
    def __call__(self, input_image):
        detected_map = 255 - input_image
            
        return detected_map
