import os
import re
import time
import threading

import cv2
import numpy as np
import modules.paths
from PIL import Image
from modules import util

import insightface
from insightface.app import FaceAnalysis
from modules import util, gfpgan_model, insightface_model, upscaler_esrgan

def handler(task):
    print(task)

    input_image = task["image"]
    source_image = task.get("face", None)

    if source_image is not None:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

        file_format = task.get("file_format", "") or ".jpeg"
        filename = task.get("filename", f"{time.strftime('%Y%m%d%H%M%S')}{file_format}")
        filename = "_face".join(os.path.splitext(filename))

        task["total_step"] = 20
        finished = False
        def progessing(task, steps, input_image, sleep=0.1):
            for i in range(20):
                # progress_output(task, "reface", picture=input_image)
                progress_output(task, "reface")
                time.sleep(0.1)
                if finished:
                    break
        threading.Thread(target=progessing, args=(task, 20, input_image, 0.1)).start()
        
        # app = FaceAnalysis(root=modules.paths.face_models_path)
        # app.prepare(ctx_id=0, det_size=(640, 640))
        # faces = app.get(input_image)
        # faces = sorted(faces, key = lambda x : ((x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1])), reverse=True)
        analysis = insightface_model.Analysis()
        faces = analysis(input_image)
        
        # model_name = os.path.join(modules.paths.face_models_path, "inswapper_128.onnx")
        # swapper = insightface.model_zoo.get_model(model_name, root=modules.paths.face_models_path)
        swapper = insightface_model.Swapper()
        gfpgan = gfpgan_model.GFPGan()
        
        source_face = analysis(source_image)[0]
        
        # upscaler = upscaler_esrgan.UpscalerESRGAN("4x-UltraSharp.pth")
        for face in faces:
            bgr_fake, M = None, None
            mask = util.face_mask(input_image, [face.landmark_2d_106])

            for _ in range(3): # more like source face 
                bgr_fake, M = swapper(input_image, face, source_face, paste_back=False)
                input_image = paste_face(input_image, bgr_fake, M, mask)
            
            up = 8
            bgr_fake = util.resize_image(bgr_fake, 128 * up, 128 * up)
            bgr_fake = gfpgan(bgr_fake, only_center_face=True)
            M = M * up
            input_image = paste_face(input_image, bgr_fake, M, mask)
        
        all_mask = util.face_mask(input_image, [face.landmark_2d_106 for face in faces])[:,:,0].astype(np.float32) / 255
        all_mask = all_mask.reshape(input_image.shape[0], input_image.shape[1], 1)
        # input_image[all_mask > 20] = gfpgan(input_image, only_center_face=False)[all_mask > 20]
        input_image = input_image * (1 - all_mask) + gfpgan(input_image, only_center_face=False) * all_mask
        input_image = cv2.cvtColor(input_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        finished = True

        progress_output(task, "save")
        save_filename = os.path.join(modules.paths.temp_outputs_path, filename)
        # Image.fromarray(input_image).save(save_filename)
        util.save_image_with_geninfo(Image.fromarray(input_image), create_infotext(task.get("pnginfo", {})), save_filename)

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
        "reface": "reface image ...",
        "save": "save image ...",
        "finished": f"reface finished."
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

def paste_face(img, bgr_fake, M, mask):
    target_img = img
    fake_diff = bgr_fake.astype(np.float32) - bgr_fake.astype(np.float32)
    fake_diff = np.abs(fake_diff).mean(axis=2)
    fake_diff[:2,:] = 0
    fake_diff[-2:,:] = 0
    fake_diff[:,:2] = 0
    fake_diff[:,-2:] = 0
    IM = cv2.invertAffineTransform(M)
    # img_white = np.full((aimg.shape[0],aimg.shape[1]), 255, dtype=np.float32)
    img_white = mask[:, :, 0].astype(np.float32)
    bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    # img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    img_white[img_white>20] = 255
    fthresh = 10
    fake_diff[fake_diff<fthresh] = 0
    fake_diff[fake_diff>=fthresh] = 255
    img_mask = img_white
    mask_h_inds, mask_w_inds = np.where(img_mask==255)
    mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
    mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
    mask_size = int(np.sqrt(mask_h*mask_w))
    k = max(mask_size//10, 10)
    #k = max(mask_size//20, 6)
    #k = 6
    kernel = np.ones((k,k),np.uint8)
    img_mask = cv2.erode(img_mask,kernel,iterations = 1)
    kernel = np.ones((2,2),np.uint8)
    fake_diff = cv2.dilate(fake_diff,kernel,iterations = 1)
    k = max(mask_size//20, 5)
    #k = 3
    #k = 3
    kernel_size = (k, k)
    blur_size = tuple(2*i+1 for i in kernel_size)
    img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    k = 5
    kernel_size = (k, k)
    blur_size = tuple(2*i+1 for i in kernel_size)
    fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
    img_mask /= 255
    fake_diff /= 255
    #img_mask = fake_diff
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    fake_merged = img_mask * bgr_fake + (1-img_mask) * target_img.astype(np.float32)
    fake_merged = fake_merged.astype(np.uint8)
    return fake_merged