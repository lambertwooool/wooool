import sys
import torch
from modules import devices

current_loaded_models = []

# True use GPU, value is (run, offload)
device_config = {
    "clip": (False, False),
    "clip_vision": (True, False),
    "annotator": (False, False),
    "unet": (True, False),
    "upscaler": (True, False),
    "controlnet": (True, False),
    "vae": (False, False),
    "face": (True, False),
}

class LoadedModel:
    def __init__(self, model):
        self.model = model
        self.model_accelerated = False
        self.device = model.load_device

    def model_memory(self):
        return self.model.model_size()

    def model_memory_required(self, device):
        if device == self.model.current_device:
            return 0
        else:
            return self.model_memory()

    def model_load(self, device_to=None):
        if device_to is None:
            device_to = self.device

        # self.model.model_patches_to(self.device)
        # self.model.model_patches_to(self.model.model_dtype())
        self.model.model_patches_to(device_to)
        self.model.model_patches_to(devices.dtype(device_to))

        try:
            self.real_model = self.model.patch_model(device_to=device_to) #TODO: do something with loras and offloading to CPU
        except Exception as e:
            self.model.unpatch_model(self.model.offload_device)
            self.model_unload()
            raise e

        return self.real_model

    def model_unload(self):
        self.model.unpatch_model(self.model.offload_device)
        self.model.model_patches_to(self.model.offload_device)

    def __eq__(self, other):
        return self.model is other.model
    
def minimum_inference_memory():
    return 1 * (1024 ** 3) # 1G

def load_model_gpu(model):
    return load_models_gpu([model])

def load_models_gpu(models, memory_required=0):
    global current_loaded_models

    # extra_mem = max(minimum_inference_memory(), memory_required)
    extra_mem = 0

    device_cpu = torch.device("cpu")
    models_already_loaded = []
    models_to_load = {}

    for x in models:
        loaded_model = LoadedModel(x)
        load_mode = None

        if loaded_model in current_loaded_models:
            index = current_loaded_models.index(loaded_model)
            current_loaded_models.insert(0, current_loaded_models.pop(index))
            models_already_loaded.append(loaded_model)
            if loaded_model.model.current_device != loaded_model.model.load_device:
                load_mode = "switch"
        else:
            load_mode = "load"

        if load_mode is not None:
            if models_to_load.get(loaded_model.device) is None:
                models_to_load[loaded_model.device] = { "switch":[], "load":[] }
            models_to_load[loaded_model.device][load_mode] \
                .append((loaded_model, loaded_model.model_memory_required(loaded_model.device)))
            unload_model_clones(loaded_model.model)

    for device in models_to_load:
        # Prioritize switch model over load model
        all_load_models = models_to_load[device]["switch"] + models_to_load[device]["load"]
        is_loaded = loaded_model in current_loaded_models

        if device != device_cpu:
            memory_required = sum([x[1] for x in all_load_models])
            free_memory(memory_required * 1.3 + extra_mem, device, models_already_loaded)

        for loaded_model, use_memory in all_load_models:
            model = loaded_model.model
            device_to = device

            if device != device_cpu:
                free_ram, free_vram = devices.get_free_memory(get_all=True)
                # when not enough vram to load model, try free ram to load
                if use_memory > (free_vram - extra_mem) and not free_memory(use_memory + extra_mem, device, models_already_loaded):
                    # lowvram_model_memory = int(max(256 * (1024 ** 2), (current_free_mem - 1024 ** 3) / 1.3 ))
                    # device_to = device_cpu
                    print("memory lower ...", [(x.real_model.__class__.__name__, sys.getrefcount(x)) for x in current_loaded_models])

            cur_loaded_model = loaded_model.model_load(device_to)

            if hasattr(loaded_model, "model"):
                print(f"[Load Model] {load_mode} {cur_loaded_model.__class__.__name__} to {device_to}, model size {loaded_model.model_memory() / (1024 ** 3):.2f}G, use memory {use_memory / (1024 ** 3):.2f}G")

            if not is_loaded:
                current_loaded_models.insert(0, loaded_model)

    return

def free_memory(memory_required, device, keep_loaded=[], gc=True):
    global current_loaded_models

    unloaded_model = False
    device_cpu = torch.device("cpu")

    for i in range(len(current_loaded_models) -1, -1, -1):
        if devices.get_free_memory(device) > memory_required:
            break
        shift_model = current_loaded_models[i]
        free_ram, free_vram = devices.get_free_memory(get_all=True)
        if shift_model not in keep_loaded:
            if shift_model.device == device and device != device_cpu:
                # shift_model_ram = shift_model.model_memory_required(device_cpu)
                # if free_ram > shift_model_ram or free_memory(shift_model_ram, device_cpu, keep_loaded, False):
                    m = current_loaded_models.pop(i)
                    if hasattr(m, "model"):
                        print(f"[Free memery] Unload {m.real_model.__class__.__name__} from {m.model.current_device}, free memory {m.model_memory() / (1024 ** 3):.2f}G")
                    m.model_unload()
                    del m

            unloaded_model = True

    if unloaded_model and gc:
        devices.torch_gc()

    return devices.get_free_memory(device) > memory_required

def unload_model_clones(model):
    to_unload = []
    for i in range(len(current_loaded_models)):
        if model.is_clone(current_loaded_models[i].model):
            to_unload = [i] + to_unload

    for i in to_unload:
        print("unload clone", i)
        current_loaded_models.pop(i).model_unload()

def cleanup_models():
    to_delete = []
    for i in range(len(current_loaded_models)):
        if sys.getrefcount(current_loaded_models[i].model) <= 2:
            to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        x.model_unload()
        del x

def init_device(type):
    return torch.device("cpu")

def run_device(type):
    if device_config[type][0] and devices.should_use_fp16():
        return devices.get_torch_device()
    else:
        return torch.device("cpu")

def offload_device(type):
    if device_config[type][1] and devices.should_use_fp16():
        return devices.get_torch_device()
    else:
        return torch.device("cpu")