import os
import torch.nn as nn

import modules.paths
from modules.model import model_helper, model_loader, model_patcher

class Interposer(nn.Module):
	"""
		Basic NN layout, ported from:
		https://github.com/city96/SD-Latent-Interposer/blob/main/interposer.py
	"""
	version = 3.1 # network revision

	def __init__(self):
		super().__init__()
		self.chan = 4
		self.hid = 128

		self.head_join  = nn.ReLU()
		self.head_short = nn.Conv2d(self.chan, self.hid, kernel_size=3, stride=1, padding=1)
		self.head_long  = nn.Sequential(
			nn.Conv2d(self.chan, self.hid, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.1),
			nn.Conv2d(self.hid,  self.hid, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.1),
			nn.Conv2d(self.hid,  self.hid, kernel_size=3, stride=1, padding=1),
		)
		self.core = nn.Sequential(
			Block(self.hid),
			Block(self.hid),
			Block(self.hid),
		)
		self.tail = nn.Sequential(
			nn.ReLU(),
			nn.Conv2d(self.hid, self.chan, kernel_size=3, stride=1, padding=1)
		)

	def forward(self, x):
		y = self.head_join(
			self.head_long(x)+
			self.head_short(x)
		)
		z = self.core(y)
		return self.tail(z)

class Block(nn.Module):
	def __init__(self, size):
		super().__init__()
		self.join = nn.ReLU()
		self.long = nn.Sequential(
			nn.Conv2d(size, size, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.1),
			nn.Conv2d(size, size, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.1),
			nn.Conv2d(size, size, kernel_size=3, stride=1, padding=1),
		)
	def forward(self, x):
		y = self.long(x)
		z = self.join(y + x)
		return z


class LatentInterposer:
	def __init__(self, latent_src, latent_dst):
		if latent_src == latent_dst:
			return
		
		model = Interposer()
		model.eval()

		type_map = {
			"xl": "xl",
			"sdxl": "xl",
			"v1": "v1",
			"sd15": "v1",
		}

		filename = f"{type_map.get(latent_src)}-to-{type_map.get(latent_dst)}_interposer-v{model.version}.safetensors"
		model_path = os.path.join(modules.paths.vae_approx_path, filename)
		sd = model_helper.load_torch_file(model_path)

		model.load_state_dict(sd)

		load_device = model_loader.run_device("vae")
		offload_device = model_loader.offload_device("vae")
		self.model = model_patcher.ModelPatcher(model, load_device, offload_device)


	def __call__(self, latent):
		model_loader.load_model_gpu(self.model)
		device = self.model.load_device

		sample_latent = latent.clone()
		sample_latent = sample_latent.to(device)
		sample_latent = self.model.model(sample_latent).to(latent)

		return sample_latent
