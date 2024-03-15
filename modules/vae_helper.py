import math
import os
import einops
import numpy as np
import torch
import modules.paths
from modules.model import latent_formats, model_loader, model_helper
from modules import devices

VAE_approx_models = {}

class LatentPreviewer:
    def decode_latent_to_preview(self, x0):
        pass

    def decode_latent_to_preview_image(self, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return preview_image.cpu().numpy().clip(0, 255).astype(np.uint8)

class Latent2RGBPreviewer(LatentPreviewer):
    def __init__(self, latent_rgb_factors):
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu")

    def decode_latent_to_preview(self, x0):
        latent_image = x0[0].permute(1, 2, 0).cpu() @ self.latent_rgb_factors

        latents_ubyte = (((latent_image + 1) / 2)
                            .clamp(0, 1)  # change scale from -1..1 to 0..1
                            .mul(0xFF)  # to 0..255
                            .byte()).cpu()

        return latents_ubyte

def get_previewer(latent_format):
    previewer = None
    if latent_format.latent_rgb_factors is not None:
        previewer = Latent2RGBPreviewer(latent_format.latent_rgb_factors)

    return previewer

class VAEApprox(torch.nn.Module):
    def __init__(self):
        super(VAEApprox, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, (7, 7))
        self.conv2 = torch.nn.Conv2d(8, 16, (5, 5))
        self.conv3 = torch.nn.Conv2d(16, 32, (3, 3))
        self.conv4 = torch.nn.Conv2d(32, 64, (3, 3))
        self.conv5 = torch.nn.Conv2d(64, 32, (3, 3))
        self.conv6 = torch.nn.Conv2d(32, 16, (3, 3))
        self.conv7 = torch.nn.Conv2d(16, 8, (3, 3))
        self.conv8 = torch.nn.Conv2d(8, 3, (3, 3))
        self.current_type = None

    def forward(self, x):
        extra = 11
        x = torch.nn.functional.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        x = torch.nn.functional.pad(x, (extra, extra, extra, extra))
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(x, 0.1)
        return x
    
def get_vae_approx(model):
    is_sdxl = isinstance(model.model.latent_format, latent_formats.SDXL)
    vae_approx_filename = os.path.join(modules.paths.vae_approx_path, 'xlvaeapp.pth' if is_sdxl else 'vaeapp_sd15.pth')

    if vae_approx_filename in VAE_approx_models:
        VAE_approx_model = VAE_approx_models[vae_approx_filename]
    else:
        sd = model_helper.load_torch_file(vae_approx_filename)
        VAE_approx_model = VAEApprox()
        VAE_approx_model.load_state_dict(sd)
        del sd
        VAE_approx_model.eval()

        load_device = model_loader.run_device("unet")
        dtype = devices.dtype(load_device)

        if dtype == torch.float16:
            VAE_approx_model.half()
        else:
            VAE_approx_model.float()
            
        VAE_approx_model.current_type = dtype
        VAE_approx_model.to(load_device)
        VAE_approx_models[vae_approx_filename] = VAE_approx_model
    
    return VAE_approx_model

@torch.no_grad()
@torch.inference_mode()
def decode_vae(vae, latent_image, tiled=False, tile_size=512):
    if tiled:
        decoded_latent = vae.decode_tiled(latent_image, tile_x=tile_size // 8, tile_y=tile_size // 8, )
    else:
        decoded_latent = vae.decode(latent_image)
    devices.torch_gc()

    return decoded_latent

@torch.no_grad()
@torch.inference_mode()
def decode_vae_preview(unet_model, latent_image):
    VAE_approx_model = get_vae_approx(unet_model)
    with torch.no_grad():
        if latent_image.is_cpu:
            latent_image = latent_image.clone().cuda()
        preview_image = latent_image.to(VAE_approx_model.current_type)
        preview_image = VAE_approx_model(preview_image) * 127.5 + 127.5
        preview_image = einops.rearrange(preview_image, 'b c h w -> b h w c')[0]
        preview_image = preview_image.cpu().numpy().clip(0, 255).astype(np.uint8)
    
    return preview_image

def vae_encode_crop_pixels(pixels):
    x = (pixels.shape[1] // 8) * 8
    y = (pixels.shape[2] // 8) * 8
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % 8) // 2
        y_offset = (pixels.shape[2] % 8) // 2
        pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
    return pixels

@torch.no_grad()
@torch.inference_mode()
def encode_vae(vae, pixels, tiled=False, tile_size=512):
    if tiled:
        pixels = vae_encode_crop_pixels(pixels)
        t = vae.encode_tiled(pixels[:,:,:,:3], tile_x=tile_size, tile_y=tile_size, )
        return {"samples":t}
    else:
        pixels = vae_encode_crop_pixels(pixels)
        t = vae.encode(pixels[:,:,:,:3])
        return {"samples":t}


@torch.no_grad()
@torch.inference_mode()
def encode_vae_mask(vae, pixels, mask, grow_mask_by=0, tiled=False):
    assert mask.ndim == 3 and pixels.ndim == 4
    # assert mask.shape[-1] == pixels.shape[-2]
    # assert mask.shape[-2] == pixels.shape[-3]

    x = (pixels.shape[1] // 8) * 8
    y = (pixels.shape[2] // 8) * 8
    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

    pixels = pixels.clone()
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % 8) // 2
        y_offset = (pixels.shape[2] % 8) // 2
        pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
        mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

    #grow mask by a few pixels to keep things seamless in latent space
    if grow_mask_by == 0:
        mask_erosion = mask
    else:
        kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
        padding = math.ceil((grow_mask_by - 1) / 2)

        mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)

    # m = (1.0 - mask.round()).squeeze(1)
    # for i in range(3):
    #     pixels[:,:,:,i] -= 0.5
    #     pixels[:,:,:,i] *= m
    #     pixels[:,:,:,i] += 0.5
    t = encode_vae(vae, pixels, tiled=tiled)
    n = mask_erosion[:,:,:x,:y].round()
    t["noise_mask"] = None if (n == 1).all() else n

    devices.torch_gc()

    return t

def parse(x, model):
    vae_approx_model = get_vae_approx(model)
    x_origin = x.clone()

    model_loader.load_model_gpu(vae_approx_model)

    x = x_origin.to(device=vae_approx_model.load_device, dtype=vae_approx_model.dtype)
    x = vae_approx_model.model(x).to(x_origin)
    return x
