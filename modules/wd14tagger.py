# https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags

import csv
import os
import re
import numpy as np
import onnxruntime as ort
from onnxruntime import InferenceSession
from PIL import Image

import modules.paths
from modules import util

all_models = ["wd-v1-4-moat-tagger-v2.onnx", "wd-swinv2-tagger-v3.onnx",
              "wd-v1-4-convnext-tagger-v2.onnx", "wd-v1-4-convnext-tagger.onnx",
              "wd-v1-4-convnextv2-tagger-v2.onnx", "wd-v1-4-vit-tagger-v2.onnx",
            ]

default_exclude_tags = [    "greyscale", "monochrome", "realistic", "smile", "solo", "solo_focus", "looking_at_viewer", \
                            "blurry", "blurry_background", "depth_of_field" ]

def tag(image, model_name="wd-swinv2-tagger-v3.onnx", threshold=0.35, character_threshold=0.85, exclude_tags=[], only_text=True):
    model_path = os.path.join(modules.paths.wd14tagger_path, model_name)
    model = InferenceSession(model_path, providers=ort.get_available_providers())

    input = model.get_inputs()[0]
    height = input.shape[1]

    # image = util.resize_image(image, height, height)
    # image = image[:, :, ::-1]  # RGB -> BGR
    # image = np.expand_dims(image, 0)

    # Reduce to max size and pad with white
    image = Image.fromarray(image)
    ratio = float(height)/max(image.size)
    new_size = tuple([int(x*ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image, ((height-new_size[0])//2, (height-new_size[1])//2))

    image = np.array(square).astype(np.float32)
    image = image[:, :, ::-1]  # RGB -> BGR
    image = np.expand_dims(image, 0)

    # Read all tags from csv and locate start of each category
    tags = []
    general_index = None
    character_index = None
    with open(os.path.join(modules.paths.wd14tagger_path, os.path.splitext(model_name)[0] + ".csv")) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if general_index is None and row[2] == "0":
                general_index = reader.line_num - 2
            elif character_index is None and row[2] == "4":
                character_index = reader.line_num - 2
            tags.append(row[1])

    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input.name: image})[0]

    result = list(zip(tags, probs[0]))

    # rating = max(result[:general_index], key=lambda x: x[1])
    general = [item for item in result[general_index:character_index] if item[1] > threshold]
    character = [item for item in result[character_index:] if item[1] > character_threshold]

    all = character + general
    remove = [s.lower().strip() for s in default_exclude_tags + exclude_tags]
    all = sorted([(re.sub(r"_", " ", tag[0]).capitalize(), tag[1]) for tag in all \
                    if tag[0].lower().strip() not in remove],\
                key=lambda x: x[1], reverse=True)
    
    if only_text:
        all = [x[0] for x in all]

    # res = ", ".join((item[0].replace("(", "\\(").replace(")", "\\)") for item in all))

    return all