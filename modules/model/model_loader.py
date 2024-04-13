import logging
import sys
import torch
from modules import devices

current_loaded_models = []

# True use GPU, value is (run, offload, dtype)
device_config = {
    "clip": (False, False),
    "clip_vision": (True, False),
    "annotator": (True, False),
    "unet": (True, False, torch.float16),
    "upscaler": (True, False),
    "controlnet": (True, False, torch.float8_e4m3fn),
    "ipadapter": (True, False, torch.float16),
    "image2text": (True, False, torch.float16),
    "vae": (True, False, torch.bfloat16),
    "face": (True, False),
}

def module_size(module):
    if module is None:
        return 0
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem

class LoadedModel:
    def __init__(self, model):
        self.model = model
        self.device = model.load_device
        self.weights_loaded = False
        self.real_model = None

    def model_memory(self):
        return self.model.model_size()

    def model_memory_required(self, device):
        if device == self.model.current_device:
            return 0
        else:
            return self.model_memory()

    def model_load(self, device_to=None, lowvram_model_memory=0):
        if device_to is None:
            device_to = self.device

        patch_model_to = device_to

        self.model.model_patches_to(patch_model_to)
        self.model.model_patches_to(devices.dtype(patch_model_to))

        load_weights = not self.weights_loaded

        try:
            if lowvram_model_memory > 0 and load_weights:
                self.real_model = self.model.patch_model_lowvram(device_to=patch_model_to, lowvram_model_memory=lowvram_model_memory)
            else:
                self.real_model = self.model.patch_model(device_to=patch_model_to, patch_weights=load_weights)
            # self.real_model = self.model.patch_model_lowvram(device_to=patch_model_to, lowvram_model_memory=lowvram_model_memory)
        except Exception as e:
            self.model.unpatch_model(self.model.offload_device)
            self.model_unload()
            raise e

        self.weights_loaded = True
        return self.real_model

    def model_unload(self, unpatch_weights=True):
        self.model.unpatch_model(self.model.offload_device, unpatch_weights=unpatch_weights)
        self.model.model_patches_to(self.model.offload_device)
        self.weights_loaded = self.weights_loaded and not unpatch_weights
        self.real_model = None

    def __eq__(self, other):
        return self.model is other.model
    
def minimum_inference_memory():
    return 1 * (1024 ** 3) # 1G

def load_model_gpu(model):
    return load_models_gpu([model])

def load_models_gpu(models, memory_required=0):
    global current_loaded_models

    # extra_mem = max(minimum_inference_memory(), memory_required)
    extra_mem = memory_required

    device_cpu = torch.device("cpu")
    models_already_loaded = []
    models_to_load = {}

    for x in models:
        loaded_model = LoadedModel(x)

        if loaded_model in current_loaded_models:
            models_already_loaded.append(loaded_model)

        if models_to_load.get(loaded_model.device) is None:
            models_to_load[loaded_model.device] = []
        models_to_load[loaded_model.device] \
            .append((loaded_model, loaded_model.model_memory_required(loaded_model.device)))
        unload_model_clones(loaded_model.model)

    for device in models_to_load:
        # Prioritize switch model over load model
        all_load_models = models_to_load[device]

        memory_all_load_modles = sum([x[1] for x in all_load_models])
        free_memory(memory_all_load_modles * 1.3 + extra_mem, device, models_already_loaded)

        for loaded_model, use_memory in all_load_models:
            is_loaded = loaded_model in current_loaded_models
            model = loaded_model.model

            free_mem = devices.get_free_memory(device=device)
            lowvram_model_memory = 0
            
            # when not enough vram to load model, try free ram to load
            keep_loaded = [x for x in models_already_loaded] + [x[0] for x in all_load_models]
            if use_memory > (free_mem - extra_mem) and not free_memory(use_memory + extra_mem, device, keep_loaded):
                # print("memory lower ...", [(x.real_model.__class__.__name__, sys.getrefcount(x)) for x in current_loaded_models])
                lowvram_model_memory = int(max(64 * (1024 * 1024), (free_mem - 1024 ** 3)))
                # lowvram_model_memory = int(max(64 * (1024 * 1024), free_mem ))

            cur_loaded_model = loaded_model.model_load(device_to=device, lowvram_model_memory=lowvram_model_memory)

            if hasattr(loaded_model, "model"):
                print(f"[Load Model] {cur_loaded_model.__class__.__name__} to {device}, model size {loaded_model.model_memory() / (1024 ** 3):.2f}G, use memory {use_memory / (1024 ** 3):.2f}G")

            if not is_loaded:
                current_loaded_models.insert(0, loaded_model)

    return

def free_memory(memory_required, device, keep_loaded=[], gc=True):
    global current_loaded_models

    unloaded_model = False

    for i in range(len(current_loaded_models) -1, -1, -1):
        if devices.get_free_memory(device) > memory_required:
            break
        shift_model = current_loaded_models[i]
        free_ram, free_vram = devices.get_free_memory(get_all=True)
        if shift_model not in keep_loaded:
            if shift_model.device == device:
                m = current_loaded_models.pop(i)
                if hasattr(m, "model"):
                    print(f"[Free memery] Unload {m.real_model.__class__.__name__} from {m.model.current_device}, free memory {m.model_memory() / (1024 ** 3):.2f}G")
                m.model_unload()
                del m

            unloaded_model = True

    if unloaded_model and gc:
        devices.torch_gc()

    return devices.get_free_memory(device) > memory_required

def unload_model_clones(model, unload_weights_only=True, force_unload=True):
    to_unload = []
    for i in range(len(current_loaded_models)):
        if model.is_clone(current_loaded_models[i].model):
            to_unload = [i] + to_unload

    if len(to_unload) == 0:
        return True

    same_weights = 0
    for i in to_unload:
        if model.clone_has_same_weights(current_loaded_models[i].model):
            same_weights += 1

    if same_weights == len(to_unload):
        unload_weight = False
    else:
        unload_weight = True

    if not force_unload:
        if unload_weights_only and unload_weight == False:
            return None

    for i in to_unload:
        logging.debug("unload clone {} {}".format(i, unload_weight))
        current_loaded_models.pop(i).model_unload(unpatch_weights=unload_weight)

    return unload_weight

def cleanup_models(keep_clone_weights_loaded=False):
    to_delete = []
    for i in range(len(current_loaded_models)):
        if sys.getrefcount(current_loaded_models[i].model) <= 2:
            if not keep_clone_weights_loaded:
                to_delete = [i] + to_delete
            #TODO: find a less fragile way to do this.
            elif sys.getrefcount(current_loaded_models[i].real_model) <= 3: #references from .real_model + the .model
                to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        x.model_unload()
        del x

def unload_all_models():
    free_memory(1e30, devices.get_torch_device())

def init_device(type_name):
    return torch.device("cpu")

def run_device(type_name):
    if device_config[type_name][0] and devices.should_use_fp16():
        return devices.get_torch_device()
    else:
        return torch.device("cpu")

def offload_device(type_name):
    if device_config[type_name][1] and devices.should_use_fp16():
        return devices.get_torch_device()
    else:
        return torch.device("cpu")

def dtype(type_name, want_use_dtype=None):
    device = run_device(type_name)
    if want_use_dtype is None:
        want_use_dtype = device_config[type_name][2] if len(device_config[type_name]) > 2 else None
    return devices.dtype(device, want_use_dtype)

def get_device_and_dtype(type_name, want_use_dtype=None):
    t_load_device = run_device(type_name)
    t_offload_device = offload_device(type_name)
    t_dtype = dtype(type_name, want_use_dtype=want_use_dtype)
    t_manual_cast_dtype = devices.unet_manual_cast(t_dtype, t_load_device) or t_dtype

    return t_load_device, t_offload_device, t_dtype, t_manual_cast_dtype