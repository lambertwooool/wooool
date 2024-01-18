import os
import torch
import time
from modules import devices, constants
from modules.model import model_base
import modules.model.cldm.cldm
import modules.model.ldm.modules.diffusionmodules.openaimodel
import modules.model.samplers
import modules.model.k_diffusion.external
import modules.model.ldm.modules.attention
import modules.model.k_diffusion.sampling
import modules.model.sd1_clip
import modules.model.ldm.modules.diffusionmodules.model
import modules.model.sd
import modules.model.model_patcher
import warnings

from modules.model.k_diffusion import utils
from modules.model.ldm.modules.diffusionmodules.openaimodel import timestep_embedding, forward_timestep_embed

from .patch_manager import patch, undo

class FooocusPatches:
    def __init__(self):
        self.calculate_weight = patch(__name__, modules.model.model_patcher.ModelPatcher, 'calculate_weight', calculate_weight_patched)
        self.ControlNet_forword = patch(__name__, modules.model.cldm.cldm.ControlNet, 'forward', patched_cldm_forward)
        self.UNetModel_forword = patch(__name__, modules.model.ldm.modules.diffusionmodules.openaimodel.UNetModel, 'forward', patched_unet_forward)
        self.DDPMDenoiser_forword = patch(__name__, modules.model.k_diffusion.external.DiscreteEpsDDPMDenoiser, 'forward', patched_discrete_eps_ddpm_denoiser_forward)
        self.SDXL_encode_adm = patch(__name__, modules.model.model_base.SDXL, 'encode_adm', sdxl_encode_adm_patched)
        self.Clip_encode_token_weights = patch(__name__, modules.model.sd1_clip.ClipTokenWeightEncoder, 'encode_token_weights', encode_token_weights_patched_with_a1111_method)

        warnings.filterwarnings(action='ignore', module='torchsde')

    def undo(self):
        self.calculate_weight = undo(__name__, modules.model.model_patcher.ModelPatcher, 'calculate_weight')
        self.ControlNet_forword = undo(__name__, modules.model.cldm.cldm.ControlNet, 'forward')
        self.UNetModel_forword = undo(__name__, modules.model.ldm.modules.diffusionmodules.openaimodel.UNetModel, 'forward')
        self.DDPMDenoiser_forword = undo(__name__, modules.model.k_diffusion.external.DiscreteEpsDDPMDenoiser, 'forward')
        self.SDXL_encode_adm = undo(__name__, modules.model.model_base.SDXL, 'encode_adm')
        self.Clip_encode_token_weights = undo(__name__, modules.model.sd1_clip.ClipTokenWeightEncoder, 'encode_token_weights')

sharpness = 2.0

adm_scaler_end = 0.3
positive_adm_scale = 1.5
negative_adm_scale = 0.8

cfg_x0 = 0.0
cfg_s = 1.0
cfg_cin = 1.0
adaptive_cfg = 0.7
# adaptive_cfg = 0.


def calculate_weight_patched(self, patches, weight, key):
    for p in patches:
        alpha = p[0]
        v = p[1]
        strength_model = p[2]

        if strength_model != 1.0:
            weight *= strength_model

        if isinstance(v, list):
            v = (self.calculate_weight(v[1:], v[0].clone(), key),)

        if len(v) == 1:
            w1 = v[0]
            if alpha != 0.0:
                if w1.shape != weight.shape:
                    print("WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(key, w1.shape, weight.shape))
                else:
                    weight += alpha * devices.cast_to_device(w1, weight.device, weight.dtype)
        elif len(v) == 3:
            # fooocus
            w1 = devices.cast_to_device(v[0], weight.device, torch.float32)
            w_min = devices.cast_to_device(v[1], weight.device, torch.float32)
            w_max = devices.cast_to_device(v[2], weight.device, torch.float32)
            w1 = (w1 / 255.0) * (w_max - w_min) + w_min
            if alpha != 0.0:
                if w1.shape != weight.shape:
                    print("WARNING SHAPE MISMATCH {} FOOOCUS WEIGHT NOT MERGED {} != {}".format(key, w1.shape, weight.shape))
                else:
                    weight += alpha * devices.cast_to_device(w1, weight.device, weight.dtype)
        elif len(v) == 4:  # lora/locon
            mat1 = devices.cast_to_device(v[0], weight.device, torch.float32)
            mat2 = devices.cast_to_device(v[1], weight.device, torch.float32)
            if v[2] is not None:
                alpha *= v[2] / mat2.shape[0]
            if v[3] is not None:
                # locon mid weights, hopefully the math is fine because I didn't properly test it
                mat3 = devices.cast_to_device(v[3], weight.device, torch.float32)
                final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
                mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1),
                                mat3.transpose(0, 1).flatten(start_dim=1)).reshape(final_shape).transpose(0, 1)
            try:
                weight += (alpha * torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1))).reshape(
                    weight.shape).type(weight.dtype)
            except Exception as e:
                print("ERROR", key, e)
        elif len(v) == 8:  # lokr
            w1 = v[0]
            w2 = v[1]
            w1_a = v[3]
            w1_b = v[4]
            w2_a = v[5]
            w2_b = v[6]
            t2 = v[7]
            dim = None

            if w1 is None:
                dim = w1_b.shape[0]
                w1 = torch.mm(devices.cast_to_device(w1_a, weight.device, torch.float32),
                              devices.cast_to_device(w1_b, weight.device, torch.float32))
            else:
                w1 = devices.cast_to_device(w1, weight.device, torch.float32)

            if w2 is None:
                dim = w2_b.shape[0]
                if t2 is None:
                    w2 = torch.mm(devices.cast_to_device(w2_a, weight.device, torch.float32),
                                  devices.cast_to_device(w2_b, weight.device, torch.float32))
                else:
                    w2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                      devices.cast_to_device(t2, weight.device, torch.float32),
                                      devices.cast_to_device(w2_b, weight.device, torch.float32),
                                      devices.cast_to_device(w2_a, weight.device, torch.float32))
            else:
                w2 = devices.cast_to_device(w2, weight.device, torch.float32)

            if len(w2.shape) == 4:
                w1 = w1.unsqueeze(2).unsqueeze(2)
            if v[2] is not None and dim is not None:
                alpha *= v[2] / dim

            try:
                weight += alpha * torch.kron(w1, w2).reshape(weight.shape).type(weight.dtype)
            except Exception as e:
                print("ERROR", key, e)
        else:  # loha
            w1a = v[0]
            w1b = v[1]
            if v[2] is not None:
                alpha *= v[2] / w1b.shape[0]
            w2a = v[3]
            w2b = v[4]
            if v[5] is not None:  # cp decomposition
                t1 = v[5]
                t2 = v[6]
                m1 = torch.einsum('i j k l, j r, i p -> p r k l',
                                  devices.cast_to_device(t1, weight.device, torch.float32),
                                  devices.cast_to_device(w1b, weight.device, torch.float32),
                                  devices.cast_to_device(w1a, weight.device, torch.float32))

                m2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                  devices.cast_to_device(t2, weight.device, torch.float32),
                                  devices.cast_to_device(w2b, weight.device, torch.float32),
                                  devices.cast_to_device(w2a, weight.device, torch.float32))
            else:
                m1 = torch.mm(devices.cast_to_device(w1a, weight.device, torch.float32),
                              devices.cast_to_device(w1b, weight.device, torch.float32))
                m2 = torch.mm(devices.cast_to_device(w2a, weight.device, torch.float32),
                              devices.cast_to_device(w2b, weight.device, torch.float32))

            try:
                weight += (alpha * m1 * m2).reshape(weight.shape).type(weight.dtype)
            except Exception as e:
                print("ERROR", key, e)

    return weight


def patched_discrete_eps_ddpm_denoiser_forward(self, input, sigma, **kwargs):
    global cfg_x0, cfg_s, cfg_cin
    c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
    cfg_x0, cfg_s, cfg_cin = input, c_out, c_in
    eps = self.get_eps(input * c_in, self.sigma_to_t(sigma), **kwargs)
    return input + eps * c_out


def sdxl_encode_adm_patched(self, **kwargs):
    global positive_adm_scale, negative_adm_scale

    clip_pooled = model_base.sdxl_pooled(kwargs, self.noise_augmentor)
    width = kwargs.get("width", 768)
    height = kwargs.get("height", 768)
    target_width = width
    target_height = height

    if kwargs.get("prompt_type", "") == "negative":
        width = float(width) * negative_adm_scale
        height = float(height) * negative_adm_scale
    elif kwargs.get("prompt_type", "") == "positive":
        width = float(width) * positive_adm_scale
        height = float(height) * positive_adm_scale

    # Avoid artifacts
    width = int(width)
    height = int(height)
    crop_w = 0
    crop_h = 0
    target_width = int(target_width)
    target_height = int(target_height)

    out_a = [self.embedder(torch.Tensor([height])), self.embedder(torch.Tensor([width])),
             self.embedder(torch.Tensor([crop_h])), self.embedder(torch.Tensor([crop_w])),
             self.embedder(torch.Tensor([target_height])), self.embedder(torch.Tensor([target_width]))]
    flat_a = torch.flatten(torch.cat(out_a)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)

    out_b = [self.embedder(torch.Tensor([target_height])), self.embedder(torch.Tensor([target_width])),
             self.embedder(torch.Tensor([crop_h])), self.embedder(torch.Tensor([crop_w])),
             self.embedder(torch.Tensor([target_height])), self.embedder(torch.Tensor([target_width]))]
    flat_b = torch.flatten(torch.cat(out_b)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)

    return torch.cat((clip_pooled.to(flat_a.device), flat_a, clip_pooled.to(flat_b.device), flat_b), dim=1)


def encode_token_weights_patched_with_a1111_method(self, token_weight_pairs):
    to_encode = list(self.empty_tokens)
    for x in token_weight_pairs:
        tokens = list(map(lambda a: a[0], x))
        to_encode.append(tokens)

    out, pooled = self.encode(to_encode)

    z_empty = out[0:1]
    if pooled.shape[0] > 1:
        first_pooled = pooled[1:2]
    else:
        first_pooled = pooled[0:1]

    output = []
    for k in range(1, out.shape[0]):
        z = out[k:k + 1]
        original_mean = z.mean()

        for i in range(len(z)):
            for j in range(len(z[i])):
                weight = token_weight_pairs[k - 1][j][1]
                z[i][j] = (z[i][j] - z_empty[0][j]) * weight + z_empty[0][j]

        new_mean = z.mean()
        z = z * (original_mean / new_mean)
        output.append(z)

    if len(output) == 0:
        return z_empty.cpu(), first_pooled.cpu()
    return torch.cat(output, dim=-2).cpu(), first_pooled.cpu()

def patched_cldm_forward(self, x, hint, timesteps, context, y=None, **kwargs):
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(self.dtype)
    emb = self.time_embed(t_emb)

    guided_hint = self.input_hint_block(hint, emb, context)

    y = timed_adm(y, timesteps)

    outs = []

    hs = []
    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

    h = x.type(self.dtype)
    for module, zero_conv in zip(self.input_blocks, self.zero_convs):
        if guided_hint is not None:
            h = module(h, emb, context)
            h += guided_hint
            guided_hint = None
        else:
            h = module(h, emb, context)
        outs.append(zero_conv(h, emb, context))

    h = self.middle_block(h, emb, context)
    outs.append(self.middle_block_out(h, emb, context))

    # if advanced_parameters.controlnet_softness > 0:
    #     for i in range(10):
    #         k = 1.0 - float(i) / 9.0
    #         outs[i] = outs[i] * (1.0 - advanced_parameters.controlnet_softness * k)

    return outs

def patched_unet_forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
    self.current_step = 1.0 - timesteps.to(x) / 999.0

    inpaint_fix = None
    # if getattr(self, 'in_inpaint', False) and inpaint_worker.current_task is not None:
    #     inpaint_fix = inpaint_worker.current_task.inpaint_head_feature

    transformer_options["original_shape"] = list(x.shape)
    transformer_options["current_index"] = 0
    transformer_patches = transformer_options.get("patches", {})

    y = timed_adm(y, timesteps)

    hs = []
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(self.dtype)
    emb = self.time_embed(t_emb)

    if self.num_classes is not None and y.shape == emb.shape:
        # assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

    h = x.type(self.dtype)
    for id, module in enumerate(self.input_blocks):
        transformer_options["block"] = ("input", id)
        h = forward_timestep_embed(module, h, emb, context, transformer_options)

        if inpaint_fix is not None:
            if int(h.shape[1]) == int(inpaint_fix.shape[1]):
                h = h + inpaint_fix.to(h)
                inpaint_fix = None

        if control is not None and 'input' in control and len(control['input']) > 0:
            ctrl = control['input'].pop()
            if ctrl is not None:
                h += ctrl
        hs.append(h)
    transformer_options["block"] = ("middle", 0)
    h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options)
    if control is not None and 'middle' in control and len(control['middle']) > 0:
        ctrl = control['middle'].pop()
        if ctrl is not None:
            h += ctrl

    for id, module in enumerate(self.output_blocks):
        transformer_options["block"] = ("output", id)
        hsp = hs.pop()
        if control is not None and 'output' in control and len(control['output']) > 0:
            ctrl = control['output'].pop()
            if ctrl is not None:
                hsp += ctrl

        if "output_block_patch" in transformer_patches:
            patch = transformer_patches["output_block_patch"]
            for p in patch:
                h, hsp = p(h, hsp, transformer_options)

        h = torch.cat([h, hsp], dim=1)
        del hsp
        if len(hs) > 0:
            output_shape = hs[-1].shape
        else:
            output_shape = None
        h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape)
    h = h.type(x.dtype)
    if self.predict_codebook_ids:
        return self.id_predictor(h)
    else:
        return self.out(h)

def timed_adm(y, timesteps):
    if isinstance(y, torch.Tensor) and int(y.dim()) == 2 and int(y.shape[1]) == 5632:
        y_mask = (timesteps > 999.0 * (1.0 - float(adm_scaler_end))).to(y)[..., None]
        y_with_adm = y[..., :2816].clone()
        y_without_adm = y[..., 2816:].clone()
        return y_with_adm * y_mask + y_without_adm * (1.0 - y_mask)
    return y