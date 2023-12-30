import re
import cv2
import dlib
import fractions
import hashlib
import io
import json 
import math
import numpy as np 
import os
import piexif
import piexif.helper
import random
import requests
import shutil
import tempfile
import time
import torch
from collections import namedtuple
from tqdm import tqdm
from urllib.parse import urlparse
from typing import Optional, Union
from PIL import Image, PngImagePlugin, ImageFilter

from modules import shared, paths

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
modules_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(modules_path)

def webpath(fn):
    if fn.startswith(root_path):
        web_path = os.path.relpath(fn, root_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)

    return f'file={web_path}?{os.path.getmtime(fn)}'

# def list_files(dir, ext=None, exclude=None):
#     file_list = []
#     ext = ext if isinstance(ext, list) or ext is None else [ext]

#     for filename in sorted(os.listdir(dir)):
#         if ext is None or os.path.splitext(filename)[-1][1:] in ext:
#             file_list.append(os.path.join(dir, filename))

#     return file_list

def list_files(dir, ext=None, search_subdir=False, excludes=None):
    file_list = []
    ext = ext if isinstance(ext, list) or ext is None else [ext]
    excludes = excludes if isinstance(excludes, list) or excludes is None else [excludes]

    for cur_path, dirs, files in os.walk(dir, topdown=True, followlinks=True):
        # print(root, dirs, files)
        for filename in files:
            if  (ext is None or os.path.splitext(filename)[-1][1:] in ext) and \
                (excludes is None or not any(True if x in filename else False for x in excludes)):
                file_list.append(os.path.join(cur_path, filename))
        if not search_subdir:
            break

    return file_list

def load_json(filename: str) -> str:
    data = {}
    if filename is not None:
        try:
            with open(filename, "r", encoding="utf8") as file:
                data = json.load(file)
        except Exception:
            print(f"Error loading from {filename}")

    return data

def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(file):
        download_url_to_file(url, file, progress=progress)
    return file

def download_url_to_file(url, dest, progress=True, user_agent="Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko)", download_chunk_size=1024**2):
    if os.path.exists(dest):
        print(f'File already exists: {dest}')

    print(f'Downloading: "{url}" to {dest}\n')

    try:
        response = requests.get(url, stream=True, headers={"User-Agent": user_agent})
        total = int(response.headers.get('content-length', 0))
        start_time = time.time()

        dest = os.path.expanduser(dest)
        dst_dir = os.path.dirname(dest)
        f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

        current = 0
        if progress:
            bar = tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024)

        for data in response.iter_content(chunk_size=download_chunk_size):
            if response.status_code < 300:
                current += len(data)
                pos = f.write(data)
                if progress:
                    bar.update(pos)
            else:
                print(f"Url error [{response.status_code}] {response.reason}")
                return
        
        f.close()
        shutil.move(f.name, dest)
    except OSError as e:
       print(f"Could not write file to {dest}")
       print(e)
    finally:
        if "f" in vars():
            f.close()
            if os.path.exists(f.name):
                os.remove(f.name)

def load_url(
        url: str,
        user_agent: str = "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko)"
) -> Union[str, dict]:
    response = requests.get(url, headers={"User-Agent": user_agent})
    content = None

    try:
        if response.ok:
            if "application/json" in response.headers["Content-Type"]:
                content = json.loads(response.text)
            else:
                content = response.text
        else:
            print(f"get error: {response.status_code} {response.reason} {url}")
    except Exception as e:
        print(e)

    return content

def dictJoin(items):
    return ', '.join(filter(lambda v : v != '', [str(v) for k, v in items.items()]))

def listJoin(items):
    return ', '.join(filter(lambda x : x != '', [str(x) for x in items]))

def ratios2size(ratios, base_size):
    if ratios is None:
        return None
    wr, hr = ratios
    x = math.sqrt(base_size / (wr * hr))
    return (int(wr * x), int(hr * x))

def size2ratio(width, height, limit=10):
    ratio = width / height
    ratio_fraction = fractions.Fraction(ratio).limit_denominator(limit)
    return ratio_fraction.numerator, ratio_fraction.denominator

def output(task, percent ,message, finished=False, picture=None):
    shared.outputs.append((task, percent, finished, message, picture))

def state(task, status):
    name, time_type = {
        "new": ("waiting", "create_time"),
        "start": ("processing", "start_time"),
        "done": ("finished", "finish_time")
    }[status]

    task[time_type] = time.time()
    task["state"] = name

    print(f"- {name} {task['guid']}")

def save_temp_image(image, filename):
    temp_path = f"{paths.temp_outputs_path}/temp"
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    filename = os.path.join(temp_path, filename)
    Image.fromarray(image).save(filename)

def read_chunks(file, size=io.DEFAULT_BUFFER_SIZE):
    """Yield pieces of data from a file-like object until EOF."""
    while True:
        chunk = file.read(size)
        if not chunk:
            break
        yield chunk

def gen_file_sha256(filename):
    hash_value = None
    if filename is not None:
        blocksize = 1 << 20
        h = hashlib.sha256()
        length = 0
        with open(os.path.realpath(filename), 'rb') as f:
            for block in read_chunks(f, size=blocksize):
                length += len(block)
                h.update(block)

        hash_value = h.hexdigest()

    return hash_value

def get_faces(input_image):
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for i, face in enumerate(faces):
        # Calculate the central coordinates of the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculates the top-left and bottom-right coordinates of the crop box
        crop_size = 1024
        x1 = max(0, center_x - crop_size // 2)
        y1 = max(0, center_y - crop_size // 2)
        x2 = min(input_image.shape[1], x1 + crop_size)
        y2 = min(input_image.shape[0], y1 + crop_size)

        # Crop face
        cropped_face = input_image[y1:y2, x1:x2]
        cropped_face = input_image[max(0, y):min(input_image.shape[0], y+h), max(0, x):min(input_image.shape[1], x+w)]

        save_temp_image(cropped_face, f"face_{i}.png")

        return [x, y, w, h], cropped_face

    return None, None

def resample_image(im, width, height):
    im = Image.fromarray(im)
    im = im.resize((int(width), int(height)), resample=LANCZOS)
    return np.array(im)

def resize_image(im, width, height, resize_mode=1, crop_offset=(0.5, 0.33), use_hwc3=True, face_mode=True, size_step=1):
    """
    Resizes an image with the specified resize_mode, width, and height.

    Args:
        resize_mode: The mode to use when resizing the image.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
    """

    if im is None or im.shape[:2] == (height, width):
        return im

    im = Image.fromarray(HWC3(im) if use_hwc3 else im)

    def resize(im, w, h):
        return im.resize((w, h), resample=LANCZOS)

    width, height = int(width / size_step) * size_step, int(height / size_step) * size_step
    
    if resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        crop_offset_w, crop_offset_h = crop_offset
        crop_w = int(src_w * crop_offset_w)
        crop_h = int(src_h * crop_offset_h)

        if face_mode:
            faces, _ = get_faces(np.array(resized))

        if not face_mode or faces is None:
            x1 = max(min(0, width // 2 - crop_w), (width - src_w))
            y1 = max(min(0, height // 2 - crop_h), (height - src_h))
            res.paste(resized, box=(x1, y1))
        else:
            fx, fy, fw, fh = faces
            center_x = fx + fw // 2
            center_y = fy + fh // 2

            # x1 = max(min(0, center_x - width // 2), (width - src_w))
            x1 = max(min(0, int(width * crop_offset_w) - center_x), (width - src_w))
            # y1 = max(min(0, center_y - fh * 2), (height - src_h))
            y1 = max(min(0, int(height * crop_offset_h) - center_y), (height - src_h))

            res.paste(resized, box=(x1, y1))
    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return np.array(res)

Grid = namedtuple("Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])

def split_grid(image, tile_w=512, tile_h=512, overlap=64):
    w = image.width
    h = image.height

    non_overlap_width = tile_w - overlap
    non_overlap_height = tile_h - overlap

    cols = math.ceil((w - overlap) / non_overlap_width)
    rows = math.ceil((h - overlap) / non_overlap_height)

    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images = []

        y = int(row * dy)

        if y + tile_h >= h:
            y = h - tile_h

        for col in range(cols):
            x = int(col * dx)

            if x + tile_w >= w:
                x = w - tile_w

            tile = image.crop((x, y, x + tile_w, y + tile_h))

            row_images.append([x, tile_w, tile])

        grid.tiles.append([y, tile_h, row_images])

    return grid

def combine_grid(grid):
    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, 'L')

    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0))
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    save_temp_image(np.array(combined_image), "upscaler.png")

    return combined_image

def get_shape_ceil(h, w):
    return math.ceil(((h * w) ** 0.5) / 64.0) * 64.0


def get_image_shape_ceil(im):
    H, W, _ = im.shape
    return get_shape_ceil(H, W)

def set_image_shape_ceil(im, shape_ceil):
    shape_ceil = float(shape_ceil)

    H_origin, W_origin, _ = im.shape
    H, W = H_origin, W_origin
    
    for _ in range(256):
        current_shape_ceil = get_shape_ceil(H, W)
        if abs(current_shape_ceil - shape_ceil) < 0.1:
            break
        k = shape_ceil / current_shape_ceil
        H = int(round(float(H) * k / 64.0) * 64)
        W = int(round(float(W) * k / 64.0) * 64)

    if H == H_origin and W == W_origin:
        return im

    return resample_image(im, width=W, height=H)

def HWC3(x):
    if x is None:
        return None
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def norm255(x, low=4, high=96):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2 and x.dtype == np.float32

    v_min = np.percentile(x, low)
    v_max = np.percentile(x, high)

    x -= v_min
    x /= v_max - v_min

    return x * 255.0

def min_max_norm(x):
    x -= np.min(x)
    x /= np.maximum(np.max(x), 1e-5)
    return x

def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z

def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

#https://github.com/Mikubill/sd-webui-controlnet/blob/main/scripts/processor.py#L17
def image_pad(input_image):
    H_raw, W_raw, _ = input_image.shape
    H_pad, W_pad = pad64(H_raw), pad64(W_raw)
    img_padded = np.pad(input_image, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_raw, :W_raw, ...])

    return safer_memory(img_padded), remove_pad

def img2mask(img, H, W, low=10, high=90):
    assert img.ndim == 3 or img.ndim == 2
    assert img.dtype == np.uint8

    if img.ndim == 3:
        y = img[:, :, random.randrange(0, img.shape[2])]
    else:
        y = img

    y = cv2.resize(y, (W, H), interpolation=cv2.INTER_CUBIC)

    if random.uniform(0, 1) < 0.5:
        y = 255 - y

    return y < np.percentile(y, random.randrange(low, high))

def make_noise_disk(H, W, C, F):
    noise = np.random.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    noise = cv2.resize(noise, (W + 2 * F, H + 2 * F), interpolation=cv2.INTER_CUBIC)
    noise = noise[F: F + H, F: F + W]
    noise -= np.min(noise)
    noise /= np.max(noise)
    if C == 1:
        noise = noise[:, :, None]
    return noise

def repeat_to_batch_size(tensor, batch_size):
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]
    elif tensor.shape[0] < batch_size:
        return tensor.repeat([math.ceil(batch_size / tensor.shape[0])] + [1] * (len(tensor.shape) - 1))[:batch_size]
    return tensor

def convert_sd_to(state_dict, dtype):
    keys = list(state_dict.keys())
    for k in keys:
        state_dict[k] = state_dict[k].to(dtype)
    return state_dict

def copy_to_param(obj, attr, value):
    # inplace update tensor instead of replacing it
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)

@torch.no_grad()
@torch.inference_mode()
def pytorch_to_numpy(x):
    return [np.clip(255. * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]

@torch.no_grad()
@torch.inference_mode()
def numpy_to_pytorch(x):
    if x is None:
        return None
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y

def save_image_with_geninfo(image, geninfo, filename, pnginfo_section_name="parameters", quality=100):
    """
    Saves image to filename, including geninfo as text information for generation info.
    For PNG images, geninfo is added to existing pnginfo dictionary using the pnginfo_section_name argument as key.
    For JPG images, there's no dictionary and geninfo just replaces the EXIF description.
    """

    extension = os.path.splitext(filename)[1]

    image_format = Image.registered_extensions()[extension]

    if extension.lower() == '.png':
        pnginfo = {}
        pnginfo[pnginfo_section_name] = geninfo
        
        pnginfo_data = PngImagePlugin.PngInfo()
        for k, v in pnginfo.items():
            pnginfo_data.add_text(k, str(v))

        image.save(filename, format=image_format, quality=quality, pnginfo=pnginfo_data)

    elif extension.lower() in (".jpg", ".jpeg", ".webp"):
        if image.mode == 'RGBA':
            image = image.convert("RGB")
        elif image.mode == 'I;16':
            image = image.point(lambda p: p * 0.0038910505836576).convert("RGB" if extension.lower() == ".webp" else "L")

        image.save(filename, quality=quality, format=image_format)

        if geninfo is not None:
            exif_bytes = piexif.dump({
                "Exif": {
                    piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(geninfo or "", encoding="unicode")
                },
            })

            piexif.insert(exif_bytes, filename)
    else:
        image.save(filename, quality=quality, format=image_format)

def read_info_from_image(image: Image.Image) -> tuple[str | None, dict]:
    items = (image.info or {}).copy()

    geninfo = items.pop("parameters", "")

    if "exif" in items:
        exif = piexif.load(items["exif"])
        exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
        try:
            exif_comment = piexif.helper.UserComment.load(exif_comment)
        except ValueError:
            exif_comment = exif_comment.decode("utf8", errors="ignore")

        if exif_comment:
            items["exif comment"] = exif_comment
            geninfo = exif_comment

    return geninfo, items

def parse_generation_parameters(x: str):
    """parses generation parameters string, the one you see in text field under the picture:
    ```
    girl with an artist's beret, determined, blue eyes, desert scene, computer monitors, heavy makeup, by Alphonse Mucha and Charlie Bowater, ((eyeshadow)), (coquettish), detailed, intricate
    Negative prompt: ugly, fat, obese, chubby, (((deformed))), [blurry], bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), messy drawing
    Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 965400086, Size: 512x512, Model hash: 45dee52b
    ```
    """

    res = {}
    lines = x.strip().splitlines()
    prompts = []
    negative_prfix = "negative prompt:"
    params_prefix = ["steps:", "size:", "seed:", "model:", "sampler:", "cfg_scale:"]
    if lines:
        while lines:
            line = lines[0].strip().lower()
            if not line.startswith(negative_prfix) and not [x for x in params_prefix if line.startswith(x)]:
                prompts.append(lines.pop(0).strip())
            else:
                break
        res["prompt"] = "\n".join(prompts)
        
        if lines[0].lower().startswith(negative_prfix):
            res["negative"] = lines.pop(0)[len(negative_prfix):].strip()
        
        param_index = 1
        for line in lines:
            line = line.strip()
            params = line.split(",")

            for pi in params:
                p = pi.split(":", 1)
                if p[0].strip() != "":
                    k, v = p if len(p) > 1 else [f"param_{param_index}", p[0]]
                    k = k.strip().lower()
                    res[k] = v.strip()

                param_index += 1
    
    return res

def size_str(size: int) -> str:
    unit_index = int(math.log(size, 1000))
    unit = " KMGTPE"[unit_index]
    size = round(size / 1024 ** unit_index, 2)
    return f"{size}{unit}"

# from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        # return low * (1 - val) + high * val
        return high

    omega = torch.acos(dot.clamp(-1, 1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res

def diagonal_fov(focal_length, sensor_diagonal=43.27):
  return 2 * math.atan(sensor_diagonal / (2 * focal_length)) * (180 / math.pi)

def shuffle(input_image, f=256):
    h, w, c = input_image.shape
    x = make_noise_disk(h, w, 1, f) * float(w - 1)
    y = make_noise_disk(h, w, 1, f) * float(h - 1)
    flow = np.concatenate([x, y], axis=2).astype(np.float32)
    detected_map = cv2.remap(input_image, flow, None, cv2.INTER_LINEAR)

    return detected_map

def blur(image, k):
    image = Image.fromarray(image)
    image = image.filter(ImageFilter.BoxBlur(k))
    return np.array(image)

def random_blur(image, mask, num_pts=20, size=(64, 256), weight=(3, 20)):
    height, width = mask.shape[:2]
    indices = np.where(mask > 127)
    indices = np.array(list(zip(indices[0], indices[1])))
    random_indices = random.sample(range(len(indices)), num_pts)
    random_pts = indices[random_indices]

    for pt in random_pts:
        y, x = pt

        rect_size = random.randint(*size) 
        x1 = max(x-rect_size//2, 0)
        y1 = max(y-rect_size//2, 0)  
        x2 = min(x+rect_size//2, width-1) 
        y2 = min(y+rect_size//2 , height-1)
        
        roi = image[y1:y2, x1:x2].copy()
        blur_weight = random.randint(*weight)
        roi = cv2.blur(roi, (blur_weight, blur_weight))
        image[y1:y2, x1:x2] = roi

    return image

def max33(x):
    x = Image.fromarray(x)
    x = x.filter(ImageFilter.MaxFilter(3))
    return np.array(x)

def morphological_open(x):
    x_int32 = np.zeros_like(x).astype(np.int32)
    x_int32[x > 127] = 256
    for _ in range(32):
        maxed = max33(x_int32) - 8
        x_int32 = np.maximum(maxed, x_int32)
    return x_int32.clip(0, 255).astype(np.uint8)

def up255(x, t=127):
    y = np.zeros_like(x).astype(np.uint8)
    y[x > t] = 255
    return y

def face_mask(input_image, face_landmarks, blur_size=None):
    mask = np.zeros(input_image.shape, np.uint8)
    for landmark in face_landmarks:
        epoints = [[int(x), int(y)] for x, y in landmark]
        epoints_hull = cv2.convexHull(np.array(epoints))
        cv2.fillConvexPoly(mask, epoints_hull, color=[255, 255, 255])
    
        x1, y1 = epoints[1]
        x2, y2 = epoints[17]
        radius = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) // 2
        # center_point = epoints[72]
        center_point = [x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2]
        cv2.circle(mask, (center_point[0], center_point[1]), radius, [255, 255, 255], -1)

    if blur_size is None:
        max_y, max_x = 0, 0
        min_y, min_x = input_image.shape[:2]
        for x, y in face_landmarks[0]:
            max_y, max_x = max(y, max_y), max(x, max_x)
            min_y, min_x = min(y, min_y), min(x, min_x)
        blur_size = max(5, math.sqrt((max_x - min_x) * (max_y - min_y)) / 16)
    mask = blur(mask, blur_size)
    mask[mask > 16] = 255
    mask = blur(mask, blur_size)
    save_temp_image(mask, "face_mask.png")

    return mask

def concat_images(images, cols=None):
    image_count = len(images)
    cols = max(1, int(math.sqrt(image_count) if cols is None else cols))
    rows = math.ceil(image_count / cols)
    # print(cols, rows, image_count)
    h, w = images[0].shape[:2]
    target = Image.new("RGB", (w * cols, h * rows))
    for row in range(rows):
        for col in range(cols):
            img = images.pop(0)
            if img is not None:
                img = resize_image(img, w, h)
                target.paste(Image.fromarray(img), box=(w * col, h * row))

    save_temp_image(np.array(target), "concat.png")
    return target