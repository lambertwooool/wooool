import facexlib
import os
import gfpgan
import modules.paths
from modules import devices
from modules.model import model_loader

# model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

class GFPGan():
    def __init__(self):
        model_path, model_file = self.get_model_path("GFPGANv1.4.pth")

        load_file_from_url_orig = gfpgan.utils.load_file_from_url
        facex_load_file_from_url_detection_orig = facexlib.detection.load_file_from_url
        facex_load_file_from_url_parsing_orig = facexlib.parsing.load_file_from_url

        def patch_load_file_from_url(**kwargs):
            return load_file_from_url_orig(**dict(kwargs, model_dir=model_path))

        def patch_facex_load_file_from_url_detection(**kwargs):
            return facex_load_file_from_url_detection_orig(**dict(kwargs, save_dir=model_path, model_dir=None))

        def patch_facex_load_file_from_url_parsing(**kwargs):
            return facex_load_file_from_url_parsing_orig(**dict(kwargs, save_dir=model_path, model_dir=None))

        gfpgan.utils.load_file_from_url = patch_load_file_from_url
        facexlib.detection.load_file_from_url = patch_facex_load_file_from_url_detection
        facexlib.parsing.load_file_from_url = patch_facex_load_file_from_url_parsing

        self.model = gfpgan.GFPGANer(model_path=model_file, upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None, device=model_loader.run_device("face"))

    def get_model_path(self, model_name):
        model_path = modules.paths.face_models_path
        return model_path, os.path.join(model_path, model_name)
    
    def __call__(self, input_image, has_aligned=False, only_center_face=False, paste_back=True, get_faces=False):
        cropped_faces, restored_faces, gfpgan_output = self.model.enhance(input_image, has_aligned=has_aligned, only_center_face=only_center_face, paste_back=paste_back)

        if get_faces:
            return gfpgan_output, restored_faces, cropped_faces
        else:
            return gfpgan_output