import os
import time
import threading
import modules.paths
from PIL import Image
from modules import util

from modules import upscaler_esrgan, util, gfpgan_model

def handler(task):
    print(task)

    input_image = task["image"]

    # image = Image.fromarray(input_image)
    # w, h = image.size
    h, w = input_image.shape[:2]

    if (w * h) < (2048 * 2048):
        file_format = task.get("file_format", "") or "jpeg"
        filename = task.get("filename", f"{time.strftime('%Y%m%d%H%M%S')}.{file_format}")
        filename = "_4x".join(os.path.splitext(filename))

        task["total_step"] = 20
        upscale_finished = False
        def progessing(task, steps, input_image, sleep=1.0):
            for i in range(20):
                # progress_output(task, "upscale", picture=input_image)
                progress_output(task, "upscale")
                time.sleep(sleep)
                if upscale_finished:
                    break
        threading.Thread(target=progessing, args=(task, 20, input_image, 1.0)).start()
        
        upscaler = upscaler_esrgan.UpscalerESRGAN("4x-UltraSharp.pth")
        upscale_image = upscaler(input_image)

        gfpgan = gfpgan_model.GFPGan()
        input_image = gfpgan(input_image, only_center_face=False)
        # upscale_image = util.resize_image(input_image, w*4, h*4)
        upscale_finished = True

        progress_output(task, "save")
        save_filename = os.path.join(modules.paths.temp_outputs_path, filename)
        Image.fromarray(upscale_image).save(save_filename)

    progress_output(task, "finished")


def progress_output(task, step_type, params=(), picture=None):
    step_message = {
        "upscale": "upscaling image ...",
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