import torch
from modules import devices
from modules.model import model_detection, model_helper, model_loader, model_patcher, clip_vision, diffusers_convert, lora
from modules.model.ldm.models.autoencoder import AutoencoderKL

def load_model_weights(model, sd):
    m, u = model.load_state_dict(sd, strict=False)
    m = set(m)
    unexpected_keys = set(u)

    k = list(sd.keys())
    for x in k:
        if x not in unexpected_keys:
            w = sd.pop(x)
            del w
    if len(m) > 0:
        print("missing", m)
    return model

def load_clip_weights(model, sd):
    k = list(sd.keys())
    for x in k:
        if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
            y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
            sd[y] = sd.pop(x)

    if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in sd:
        ids = sd['cond_stage_model.transformer.text_model.embeddings.position_ids']
        if ids.dtype == torch.float32:
            sd['cond_stage_model.transformer.text_model.embeddings.position_ids'] = ids.round()

    sd = model_helper.transformers_convert(sd, "cond_stage_model.model.", "cond_stage_model.transformer.text_model.", 24)
    return load_model_weights(model, sd)

def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True):
    sd = model_helper.load_torch_file(ckpt_path)

    patcher, clip, vae, clipvision = [None] * 4

    # unet_dtype = devices.dtype()
    load_device = model_loader.run_device("unet")
    unet_dtype = devices.dtype(load_device)

    class WeightsLoader(torch.nn.Module):
        pass

    parameters = model_helper.calculate_parameters(sd, "model.diffusion_model.")
    model_config = model_detection.model_config_from_unet(sd, "model.diffusion_model.", unet_dtype)
    if model_config is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(ckpt_path))

    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

    if output_model:
        # inital_load_device = model_loader.unet_inital_load_device(parameters, unet_dtype)
        inital_load_device = model_loader.init_device("unet")
        offload_device = model_loader.offload_device("unet")
        model = model_config.get_model(sd, "model.diffusion_model.", device=inital_load_device)
        model.load_model_weights(sd, "model.diffusion_model.")

    if output_vae:
        vae = VAE()
        w = WeightsLoader()
        w.first_stage_model = vae.first_stage_model
        load_model_weights(w, sd)

    if output_clip:
        w = WeightsLoader()
        clip_target = model_config.clip_target()
        clip = CLIP(clip_target, embedding_directory=embedding_directory)
        w.cond_stage_model = clip.cond_stage_model
        sd = model_config.process_clip_state_dict(sd)
        load_model_weights(w, sd)

    left_over = sd.keys()
    if len(left_over) > 0:
        print("left over keys:", left_over)

    if output_model:
        patcher = model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device, current_device=inital_load_device)
        if inital_load_device != torch.device("cpu"):
            print("loaded straight to GPU")
            model_loader.load_model_gpu(patcher)

    return (patcher, clip, vae, clipvision)

def load_lora_for_models(model, clip, lora_model, strength_model, strength_clip):
    key_map = lora.model_lora_keys_unet(model.model)
    key_map = lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    loaded = lora.load_lora(lora_model, key_map)
    new_modelpatcher = model.clone()
    k = new_modelpatcher.add_patches(loaded, strength_model)
    new_clip = clip.clone()
    k1 = new_clip.add_patches(loaded, strength_clip)
    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            print("NOT LOADED", x)

    return (new_modelpatcher, new_clip)

class CLIP:
    def __init__(self, target=None, embedding_directory=None, no_init=False):
        if no_init:
            return
        params = target.params.copy()
        clip = target.clip
        tokenizer = target.tokenizer

        load_device = model_loader.run_device("clip")
        offload_device = model_loader.offload_device("clip")
        params['device'] = offload_device
        params['dtype'] = devices.dtype(load_device)

        self.cond_stage_model = clip(**(params))

        self.tokenizer = tokenizer(embedding_directory=embedding_directory)
        self.patcher = model_patcher.ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)
        self.layer_idx = None

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        n.layer_idx = self.layer_idx
        return n

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_patches(patches, strength_patch, strength_model)

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def encode_from_tokens(self, tokens, return_pooled=False):
        if self.layer_idx is not None:
            self.cond_stage_model.clip_layer(self.layer_idx)
        else:
            self.cond_stage_model.reset_clip_layer()

        self.load_model()
        cond, pooled = self.cond_stage_model.encode_token_weights(tokens)
        if return_pooled:
            return cond, pooled
        return cond

    def encode(self, text):
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)

    def load_sd(self, sd):
        return self.cond_stage_model.load_sd(sd)

    def get_sd(self):
        return self.cond_stage_model.state_dict()

    def load_model(self):
        model_loader.load_model_gpu(self.patcher)
        return self.patcher

    def get_key_patches(self):
        return self.patcher.get_key_patches()

class VAE:
    def __init__(self, ckpt_path=None, device=None, config=None):
        if config is None:
            #default SD1.x/SD2.x VAE parameters
            ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
            self.first_stage_model = AutoencoderKL(ddconfig, {'target': 'torch.nn.Identity'}, 4, monitor="val/rec_loss")
        else:
            self.first_stage_model = AutoencoderKL(**(config['params']))
        self.first_stage_model = self.first_stage_model.eval()
        if ckpt_path is not None:
            sd = model_helper.load_torch_file(ckpt_path)
            if 'decoder.up_blocks.0.resnets.0.norm1.weight' in sd.keys(): #diffusers format
                sd = diffusers_convert.convert_vae_state_dict(sd)
            m, u = self.first_stage_model.load_state_dict(sd, strict=False)
            if len(m) > 0:
                print("Missing VAE keys", m)

        if device is None:
            device = model_loader.run_device("vae")
        self.device = device
        self.offload_device = model_loader.offload_device("vae")
        self.vae_dtype = devices.dtype(device, use_bf16=True)
        self.first_stage_model.to(self.vae_dtype)

    def decode(self, samples_in):
        self.first_stage_model = self.first_stage_model.to(self.device)
        try:
            memory_used = (2562 * samples_in.shape[2] * samples_in.shape[3] * 64) * 1.7
            model_loader.free_memory(memory_used, self.device)
            free_memory = devices.get_free_memory(self.device)
            batch_number = int(free_memory / memory_used)
            batch_number = max(1, batch_number)

            pixel_samples = torch.empty((samples_in.shape[0], 3, round(samples_in.shape[2] * 8), round(samples_in.shape[3] * 8)), device="cpu")
            for x in range(0, samples_in.shape[0], batch_number):
                samples = samples_in[x:x+batch_number].to(self.vae_dtype).to(self.device)
                pixel_samples[x:x+batch_number] = torch.clamp((self.first_stage_model.decode(samples).cpu().float() + 1.0) / 2.0, min=0.0, max=1.0)
        except devices.OOM_EXCEPTION as e:
            print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
            pixel_samples = self.decode_tiled_(samples_in)

        self.first_stage_model = self.first_stage_model.to(self.offload_device)
        pixel_samples = pixel_samples.cpu().movedim(1,-1)
        return pixel_samples

    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap = 16):
        self.first_stage_model = self.first_stage_model.to(self.device)
        output = self.decode_tiled_(samples, tile_x, tile_y, overlap)
        self.first_stage_model = self.first_stage_model.to(self.offload_device)
        return output.movedim(1,-1)

    def encode(self, pixel_samples):
        self.first_stage_model = self.first_stage_model.to(self.device)
        pixel_samples = pixel_samples.movedim(-1,1)
        try:
            memory_used = (2078 * pixel_samples.shape[2] * pixel_samples.shape[3]) * 1.7 #NOTE: this constant along with the one in the decode above are estimated from the mem usage for the VAE and could change.
            model_loader.free_memory(memory_used, self.device)
            free_memory = devices.get_free_memory(self.device)
            batch_number = int(free_memory / memory_used)
            batch_number = max(1, batch_number)
            samples = torch.empty((pixel_samples.shape[0], 4, round(pixel_samples.shape[2] // 8), round(pixel_samples.shape[3] // 8)), device="cpu")
            for x in range(0, pixel_samples.shape[0], batch_number):
                pixels_in = (2. * pixel_samples[x:x+batch_number] - 1.).to(self.vae_dtype).to(self.device)
                samples[x:x+batch_number] = self.first_stage_model.encode(pixels_in).sample().cpu().float()

        except devices.OOM_EXCEPTION as e:
            print("Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding.")
            samples = self.encode_tiled_(pixel_samples)

        self.first_stage_model = self.first_stage_model.to(self.offload_device)
        return samples

    def encode_tiled(self, pixel_samples, tile_x=512, tile_y=512, overlap = 64):
        self.first_stage_model = self.first_stage_model.to(self.device)
        pixel_samples = pixel_samples.movedim(-1,1)
        samples = self.encode_tiled_(pixel_samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap)
        self.first_stage_model = self.first_stage_model.to(self.offload_device)
        return samples

    def get_sd(self):
        return self.first_stage_model.state_dict()