import os
from enum import Enum
import torch
import functools
import copy
from typing import Optional, List
from dataclasses import dataclass

from modules import devices, paths, util
from modules.model import conds, model_base, supported_models, supported_models_base, model_patcher, model_helper, model_loader, latent_formats
from .lib_layerdiffusion.utils import to_lora_patch_dict
from .lib_layerdiffusion.models import TransparentVAEDecoder
from .lib_layerdiffusion.attention_sharing import AttentionSharingPatcher
from .lib_layerdiffusion.enums import StableDiffusionVersion

class JoinImageWithAlpha:
    def join_image_with_alpha(self, image: torch.Tensor, alpha: torch.Tensor):
        def resize_mask(mask, shape):
            return torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[0], shape[1]), mode="bilinear").squeeze(1)

        batch_size = min(len(image), len(alpha))
        out_images = []

        alpha = 1.0 - resize_mask(alpha, image.shape[1:])
        for i in range(batch_size):
           out_images.append(torch.cat((image[i][:,:,:3], alpha[i].unsqueeze(2)), dim=2))

        result = (torch.stack(out_images),)
        return result

# ------------ Start patching ComfyUI ------------
def calculate_weight_adjust_channel(func):
    """Patches ComfyUI's LoRA weight application to accept multi-channel inputs."""

    @functools.wraps(func)
    def calculate_weight(
        self: model_patcher.ModelPatcher, patches, weight: torch.Tensor, key: str
    ) -> torch.Tensor:
        weight = func(self, patches, weight, key)

        for p in patches:
            alpha = p[0]
            v = p[1]

            # The recursion call should be handled in the main func call.
            if isinstance(v, list):
                continue

            if len(v) == 1:
                patch_type = "diff"
            elif len(v) == 2:
                patch_type = v[0]
                v = v[1]

            if patch_type == "diff":
                w1 = v[0]
                if all(
                    (
                        alpha != 0.0,
                        w1.shape != weight.shape,
                        w1.ndim == weight.ndim == 4,
                    )
                ):
                    new_shape = [max(n, m) for n, m in zip(weight.shape, w1.shape)]
                    print(
                        f"Merged with {key} channel changed from {weight.shape} to {new_shape}"
                    )
                    new_diff = alpha * devices.cast_to_device(
                        w1, weight.device, weight.dtype
                    )
                    new_weight = torch.zeros(size=new_shape).to(weight)
                    new_weight[
                        : weight.shape[0],
                        : weight.shape[1],
                        : weight.shape[2],
                        : weight.shape[3],
                    ] = weight
                    new_weight[
                        : new_diff.shape[0],
                        : new_diff.shape[1],
                        : new_diff.shape[2],
                        : new_diff.shape[3],
                    ] += new_diff
                    new_weight = new_weight.contiguous().clone()
                    weight = new_weight
        return weight

    return calculate_weight


model_patcher.ModelPatcher.calculate_weight = calculate_weight_adjust_channel(
    model_patcher.ModelPatcher.calculate_weight
)

# ------------ End patching ComfyUI ------------

class LayeredDiffusionDecode:
    """
    Decode alpha channel value from pixel value.
    [B, C=3, H, W] => [B, C=4, H, W]
    Outputs RGB image + Alpha mask.
    """
    def __init__(self, device=None, dtype=None) -> None:
        self.vae_transparent_decoder = {}

        if device is None:
            device = model_loader.run_device("vae")
        self.device = device
        
        self.load_device = model_loader.run_device("vae")
        self.offload_device = model_loader.offload_device("vae")

        if dtype is None:
            dtype = torch.float16
        self.vae_dtype = devices.dtype(device, want_use_dtype=dtype)

    def decode(self, samples, images, sd_version: latent_formats, sub_batch_size: int=16):
        """
        sub_batch_size: How many images to decode in a single pass.
        See https://github.com/huchenlei/ComfyUI-layerdiffuse/pull/4 for more
        context.
        """
        pixel = images.movedim(-1, 1)  # [B, H, W, C] => [B, C, H, W]
        
        # Decoder requires dimension to be 64-aligned.
        B, C, H, W = pixel.shape
        assert H % 64 == 0, f"Height({H}) is not multiple of 64."
        assert W % 64 == 0, f"Height({W}) is not multiple of 64."

        if not self.vae_transparent_decoder.get(sd_version):
            if isinstance(sd_version, latent_formats.SD15):
                file_name = "layer_sd15_vae_transparent_decoder.safetensors"
            elif isinstance(sd_version, latent_formats.SDXL):
                file_name = "vae_transparent_decoder.safetensors"
            else:
                print(f"layered diffusion not support {sd_version}")
                return images, None
            
            model_path = os.path.join(paths.layerd_path, file_name)
            model = TransparentVAEDecoder(
                model_helper.load_torch_file(model_path),
                device=self.device,
                dtype=self.vae_dtype,
            )

            self.vae_transparent_decoder[sd_version] = model
            self.vae_transparent_decoder[f"{sd_version}_wrap"] = model_patcher.ModelPatcher(model.model, self.load_device, self.offload_device)

        vae_model = self.vae_transparent_decoder[sd_version]
        vae_model_wrap = self.vae_transparent_decoder[f"{sd_version}_wrap"]
        model_loader.load_model_gpu(vae_model_wrap)

        decoded = []
        for start_idx in range(0, samples.shape[0], sub_batch_size):
            decoded.append(
                vae_model.decode_pixel(
                    pixel,
                    samples,
                )
            )
        pixel_with_alpha = torch.cat(decoded, dim=0)

        # [B, C, H, W] => [B, H, W, C]
        pixel_with_alpha = pixel_with_alpha.movedim(1, -1)
        image = pixel_with_alpha[..., 1:]
        alpha = pixel_with_alpha[..., 0]

        return (image, alpha)


class LayeredDiffusionDecodeRGBA(LayeredDiffusionDecode):
    """
    Decode alpha channel value from pixel value.
    [B, C=3, H, W] => [B, C=4, H, W]
    Outputs RGBA image.
    """
    def decode(self, samples, images, sd_version: str, sub_batch_size: int=16):
        image, mask = super().decode(samples, images, sd_version, sub_batch_size)
        alpha = 1.0 - mask if mask is not None else None
        return JoinImageWithAlpha().join_image_with_alpha(image, alpha)


class LayeredDiffusionDecodeSplit(LayeredDiffusionDecodeRGBA):
    """Decode RGBA every N images."""
    def decode(
        self,
        samples,
        images: torch.Tensor,
        frames: int,
        sd_version: str,
        sub_batch_size: int,
    ):
        sliced_samples = copy.copy(samples)
        sliced_samples["samples"] = sliced_samples["samples"][::frames]
        return tuple(
            (
                (
                    super(LayeredDiffusionDecodeSplit, self).decode(
                        sliced_samples, imgs, sd_version, sub_batch_size
                    )[0]
                    if i == 0
                    else imgs
                )
                for i in range(frames)
                for imgs in (images[i::frames],)
            )
        ) + (None,) * (self.MAX_FRAMES - frames)


class LayerMethod(Enum):
    ATTN = "Attention Injection"
    CONV = "Conv Injection"


class LayerType(Enum):
    FG = "Foreground"
    BG = "Background"


@dataclass
class LayeredDiffusionBase:
    model_file_name: str
    sd_version: latent_formats
    attn_sharing: bool = False
    injection_method: Optional[LayerMethod] = None
    cond_type: Optional[LayerType] = None
    # Number of output images per run.
    frames: int = 1

    @property
    def config_string(self) -> str:
        injection_method = self.injection_method.value if self.injection_method else ""
        cond_type = self.cond_type.value if self.cond_type else ""
        attn_sharing = "attn_sharing" if self.attn_sharing else ""
        frames = f"Batch size ({self.frames}N)" if self.frames != 1 else ""
        return ", ".join(
            x
            for x in (
                self.sd_version.value,
                injection_method,
                cond_type,
                attn_sharing,
                frames,
            )
            if x
        )

    def apply_c_concat(self, cond, uncond, c_concat):
        """Set foreground/background concat condition."""

        def write_c_concat(cond):
            new_cond = []
            for t in cond:
                n = [t[0], t[1].copy()]
                if "model_conds" not in n[1]:
                    n[1]["model_conds"] = {}
                n[1]["model_conds"]["c_concat"] = conds.CONDRegular(c_concat)
                new_cond.append(n)
            return new_cond

        return (write_c_concat(cond), write_c_concat(uncond))

    def apply_layered_diffusion(
        self,
        model: model_patcher.ModelPatcher,
        weight: float,
    ):
        """Patch model"""
        model_path = os.path.join(paths.layerd_path, self.model_file_name)
        layer_lora_state_dict = model_helper.load_torch_file(model_path)
        layer_lora_patch_dict = to_lora_patch_dict(layer_lora_state_dict)
        work_model = model.clone()
        work_model.add_patches(layer_lora_patch_dict, weight)
        return (work_model,)

    def apply_layered_diffusion_attn_sharing(
        self,
        model: model_patcher.ModelPatcher,
        control_img: Optional[torch.TensorType] = None,
    ):
        """Patch model with attn sharing"""
        model_path = os.path.join(paths.layerd_path, self.model_file_name)
        layer_lora_state_dict = model_helper.load_torch_file(model_path)
        work_model = model.clone()
        patcher = AttentionSharingPatcher(
            work_model, self.frames, use_control=control_img is not None
        )
        patcher.load_state_dict(layer_lora_state_dict, strict=True)
        if control_img is not None:
            patcher.set_control(control_img)
        return (work_model,)


def get_model_sd_version(model: model_patcher.ModelPatcher) -> latent_formats:
    """Get model's StableDiffusionVersion."""
    base: model_base.BaseModel = model.model
    # model_config: supported_models.supported_models_base.BASE = base.model_config
    if isinstance(base.latent_format, latent_formats.SDXL):
        return latent_formats.SDXL
    elif isinstance(base.latent_format, latent_formats.SD15):
        # SD15 and SD20 are compatible with each other.
        return latent_formats.SD15
    else:
        raise Exception(f"Unsupported SD Version: {type(base.latent_format)}.")


class LayeredDiffusionFG:
    """Generate foreground with transparent background."""

    MODELS = (
        LayeredDiffusionBase(
            model_file_name="layer_xl_transparent_attn.safetensors",
            sd_version=latent_formats.SDXL,
            injection_method=LayerMethod.ATTN,
        ),
        LayeredDiffusionBase(
            model_file_name="layer_xl_transparent_conv.safetensors",
            sd_version=latent_formats.SDXL,
            injection_method=LayerMethod.CONV,
        ),
        LayeredDiffusionBase(
            model_file_name="layer_sd15_transparent_attn.safetensors",
            sd_version=latent_formats.SD15,
            injection_method=LayerMethod.ATTN,
            attn_sharing=True,
        ),
    )

    def apply_layered_diffusion(
        self,
        model: model_patcher.ModelPatcher,
        lantent_format: latent_formats,
        injection_method: LayerMethod,
        # config: str,
        weight: float,
    ):
        # ld_model = [m for m in self.MODELS if m.config_string == config][0]
        ld_model = [m for m in self.MODELS if isinstance(lantent_format, m.sd_version) and m.injection_method==injection_method][0]
        assert get_model_sd_version(model) == ld_model.sd_version
        if ld_model.attn_sharing:
            return ld_model.apply_layered_diffusion_attn_sharing(model)
        else:
            return ld_model.apply_layered_diffusion(model, weight)


class LayeredDiffusionJoint:
    """Generate FG + BG + Blended in one inference batch. Batch size = 3N."""
    MODELS = (
        LayeredDiffusionBase(
            model_file_name="layer_sd15_joint.safetensors",
            sd_version=latent_formats.SD15,
            attn_sharing=True,
            frames=3,
        ),
    )

    def apply_layered_diffusion(
        self,
        model: model_patcher.ModelPatcher,
        config: str,
        fg_cond: Optional[List[List[torch.TensorType]]] = None,
        bg_cond: Optional[List[List[torch.TensorType]]] = None,
        blended_cond: Optional[List[List[torch.TensorType]]] = None,
    ):
        ld_model = [m for m in self.MODELS if m.config_string == config][0]
        assert get_model_sd_version(model) == ld_model.sd_version
        assert ld_model.attn_sharing
        work_model = ld_model.apply_layered_diffusion_attn_sharing(model)[0]
        work_model.model_options.setdefault("transformer_options", {})
        work_model.model_options["transformer_options"]["cond_overwrite"] = [
            cond[0][0] if cond is not None else None
            for cond in (
                fg_cond,
                bg_cond,
                blended_cond,
            )
        ]
        return (work_model,)


class LayeredDiffusionCond:
    """Generate foreground + background given background / foreground.
    - FG => Blended
    - BG => Blended
    """
    MODELS = (
        LayeredDiffusionBase(
            model_file_name="layer_xl_fg2ble.safetensors",
            sd_version=latent_formats.SDXL,
            cond_type=LayerType.FG,
        ),
        LayeredDiffusionBase(
            model_file_name="layer_xl_bg2ble.safetensors",
            sd_version=latent_formats.SDXL,
            cond_type=LayerType.BG,
        ),
    )

    def apply_layered_diffusion(
        self,
        model: model_patcher.ModelPatcher,
        cond,
        uncond,
        latent,
        config: str,
        weight: float,
    ):
        ld_model = [m for m in self.MODELS if m.config_string == config][0]
        assert get_model_sd_version(model) == ld_model.sd_version
        c_concat = model.model.latent_format.process_in(latent["samples"])
        return ld_model.apply_layered_diffusion(
            model, weight
        ) + ld_model.apply_c_concat(cond, uncond, c_concat)


class LayeredDiffusionCondJoint:
    """Generate fg/bg + blended given fg/bg.
    - FG => Blended + BG
    - BG => Blended + FG
    """
    MODELS = (
        LayeredDiffusionBase(
            model_file_name="layer_sd15_fg2bg.safetensors",
            sd_version=latent_formats.SD15,
            attn_sharing=True,
            frames=2,
            cond_type=LayerType.FG,
        ),
        LayeredDiffusionBase(
            model_file_name="layer_sd15_bg2fg.safetensors",
            sd_version=latent_formats.SD15,
            attn_sharing=True,
            frames=2,
            cond_type=LayerType.BG,
        ),
    )

    def apply_layered_diffusion(
        self,
        model: model_patcher.ModelPatcher,
        image,
        config: str,
        cond: Optional[List[List[torch.TensorType]]] = None,
        blended_cond: Optional[List[List[torch.TensorType]]] = None,
    ):
        ld_model = [m for m in self.MODELS if m.config_string == config][0]
        assert get_model_sd_version(model) == ld_model.sd_version
        assert ld_model.attn_sharing
        work_model = ld_model.apply_layered_diffusion_attn_sharing(
            model, control_img=image.movedim(-1, 1)
        )[0]
        work_model.model_options.setdefault("transformer_options", {})
        work_model.model_options["transformer_options"]["cond_overwrite"] = [
            cond[0][0] if cond is not None else None
            for cond in (
                cond,
                blended_cond,
            )
        ]
        return (work_model,)


class LayeredDiffusionDiff:
    """Extract FG/BG from blended image.
    - Blended + FG => BG
    - Blended + BG => FG
    """
    MODELS = (
        LayeredDiffusionBase(
            model_file_name="layer_xl_fgble2bg.safetensors",
            sd_version=latent_formats.SDXL,
            cond_type=LayerType.FG,
        ),
        LayeredDiffusionBase(
            model_file_name="layer_xl_bgble2fg.safetensors",
            sd_version=latent_formats.SDXL,
            cond_type=LayerType.BG,
        ),
    )

    def apply_layered_diffusion(
        self,
        model: model_patcher.ModelPatcher,
        cond,
        uncond,
        blended_latent,
        latent,
        config: str,
        weight: float,
    ):
        ld_model = [m for m in self.MODELS if m.config_string == config][0]
        assert get_model_sd_version(model) == ld_model.sd_version
        c_concat = model.model.latent_format.process_in(
            torch.cat([latent["samples"], blended_latent["samples"]], dim=1)
        )
        return ld_model.apply_layered_diffusion(
            model, weight
        ) + ld_model.apply_c_concat(cond, uncond, c_concat)
