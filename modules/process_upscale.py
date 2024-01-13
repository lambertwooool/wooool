import re
import cv2
import os
import time
import threading
import modules.paths
import numpy as np
from PIL import Image
from modules import util

from modules import upscaler_esrgan, util, gfpgan_model

def handler(task):
    input_image = task["image"][:,:,:3]
    origin_image = input_image.copy()

    # image = Image.fromarray(input_image)
    # w, h = image.size
    h, w = input_image.shape[:2]
    
    file_format = task.get("file_format", "") or "jpeg"
    filename = task.get("filename", f"{time.strftime('%Y%m%d%H%M%S')}.{file_format}")
    filename = "_4x".join(os.path.splitext(filename))

    upscale_model = task.get("upscale_model")
    repair_face = task.get("repair_face", True)
    factor = task.get("upscale_factor", 2.0)
    origin_visibility = task.get("origin_visibility", 0)
    factor_w, factor_h = int(w * factor), int(h * factor)

    print(  '[Model]:', upscale_model, \
            '\n[Repair face]:', repair_face, \
            '\n[Factor]:', factor, \
            '\n[Origin Visibility]:', origin_visibility, \
            '\n[Target Size]:', (factor_w, factor_h), \
            '\n-------------------------------'
    )

    if (factor_w * factor_h) < 8192 ** 2:
        zoom = 256 / max(w, h)
        preview_image = util.resize_image(input_image, int(w * zoom), int(h * zoom))

        if origin_visibility == 1.0:
            upscale_image = util.resize_image(input_image, factor_w, factor_h)
        else:
            task["total_step"] = 20
            upscale_finished = False
            def progessing(task, steps, preview_image, sleep=1.0):
                for i in range(20):
                    progress_output(task, "upscale", picture=preview_image)
                    # progress_output(task, "upscale")
                    time.sleep(sleep)
                    if upscale_finished:
                        break
            threading.Thread(target=progessing, args=(task, 20, preview_image, 1.0)).start()
            
            if upscale_model:
                upscaler = upscaler_esrgan.UpscalerESRGAN(upscale_model)
                upscale_image = upscaler(input_image)
            else:
                upscale_image = util.resize_image(input_image, w * 4, h * 4)
                blurred = cv2.GaussianBlur(upscale_image, (5, 5), 2)
                upscale_image = cv2.addWeighted(upscale_image, 1.5, blurred, -0.5, 0)

            upscale_finished = True

            if repair_face:
                progress_output(task, "repair_face", picture=preview_image)
                gfpgan = gfpgan_model.GFPGan()
                upscale_image = cv2.cvtColor(upscale_image, cv2.COLOR_BGR2RGB)
                upscale_image = gfpgan(upscale_image, only_center_face=False)
                upscale_image = cv2.cvtColor(upscale_image, cv2.COLOR_BGR2RGB)

            upscale_image = util.resize_image(upscale_image, factor_w, factor_h)

            if origin_visibility > 0:
                origin_image = util.resize_image(origin_image, factor_w, factor_h)
                upscale_image = (origin_visibility * origin_image.astype(np.float32) + (1 - origin_visibility) * upscale_image.astype(np.float32)).astype(np.uint8)

        progress_output(task, "save", picture=preview_image)
        save_filename = os.path.join(modules.paths.temp_outputs_path, filename)
        # Image.fromarray(upscale_image).save(save_filename)
        util.save_image_with_geninfo(Image.fromarray(upscale_image), create_infotext(task.get("pnginfo", {})), save_filename)

    progress_output(task, "finished")

def create_infotext(generation_params):
    re_line = r"[\n\r\s]"
    re_s = r"[\s,]"

    generation_params = { k: v for k, v in generation_params.items() if v }

    prompt_text = generation_params.pop("prompt", "")
    negative_prompt_text = re.sub(re_line, " ", f'Negative prompt: {generation_params.pop("negative", "")}')
    generation_params_text = ", ".join([f'{k}: {re.sub(re_s, " ", str(v))}' for k, v in generation_params.items() if v is not None])
    
    return f"{prompt_text}\n{negative_prompt_text}\n{generation_params_text}".strip()

def progress_output(task, step_type, params=(), picture=None):
    step_message = {
        "upscale": "upscaling image ...",
        "repair_face": "repair face ...",
        "save": "save image ...",
        "finished": f"upscale finished."
    }[step_type].format(*params)

    wait_percent = 5
    cur_step = task.get("cur_step", 0) + 1
    task["cur_step"] = cur_step
    total_step = task.get("total_step", 1)
    step_percent = (100 - wait_percent) / total_step
    cur_percent = cur_step * step_percent

    create_time = task["create_time"]
    message = f"<span>{step_message}</span> <span>({int(cur_percent)})% {time.time() - create_time:.1f}s</span>"

    cur_percent = max(0, min(99, cur_percent) if step_type != "finished" else cur_percent)
    finished = False if step_type != "finished" else True

    util.output(task, cur_percent, message, finished, picture)