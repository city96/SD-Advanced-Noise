import math
import torch
import numpy as np
from torchvision import transforms
from .latent_math_encoder import linear_encoder
from .latent_noise_generator import gaussian_latent_noise

class MathEncode:
	"""
		Encode latents without using a NN.
	"""
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"pixels": ("IMAGE",),
				"latent_ver": (["v1", "xl"],),
				"mode": ([
					"linear_encoder",
				],),
			}
		}
	RETURN_TYPES = ("LATENT",)
	FUNCTION = "encode"
	CATEGORY = "latent"
	TITLE = "Math Encoder"

	def encode(self, pixels, latent_ver, mode):
		out = []
		for batch, img in enumerate(pixels.numpy()):
			# target latent size
			lat_size = (round(img.shape[0]/8), round(img.shape[1]/8))
			img = img.transpose((2, 0, 1)) # [W,H,3]=>[3,W,H]
			img = torch.from_numpy(img)
			img = transforms.Resize(lat_size, antialias=True)(img)
			# encode
			lat = linear_encoder(img, latent_ver)
			out.append(lat)
		return ({"samples":torch.stack(out)},)


class LatentGaussianNoise:
	"""
		Create Gaussian noise directly in latent space.
	"""
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"latent_ver": (["v1", "xl"],),
				"width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
				"height": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
				"factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
				"null": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
				"batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
				"scale": ("INT", {"default": 1, "min": 1, "max": 8}),
				"random": (["shared", "per channel"],),
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
			}
		}
	RETURN_TYPES = ("LATENT",)
	FUNCTION = "generate"
	CATEGORY = "noise"
	TITLE = "Gaussian Noise (Latent)"

	def generate(self, latent_ver, width, height, factor, null, batch_size, scale, random, seed):
		out = []
		for b in range(batch_size):
			lat = gaussian_latent_noise(
				width = round(width/8/scale),
				height = round(height/8/scale),
				ver = latent_ver,
				seed = seed+b,
				fac = factor,
				nul = null,
				srnd = True if random == "shared" else False,
			)
			if scale > 1:
				target = (round(height/8),round(width/8))
				lat = transforms.Resize(target, antialias=True)(lat)
			out.append(lat)
		out = torch.stack(out)
		
		return ({"samples":out},)


NODE_CLASS_MAPPINGS = {
	"MathEncode": MathEncode,
	"LatentGaussianNoise": LatentGaussianNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MathEncode": MathEncode.TITLE,
    "LatentGaussianNoise": LatentGaussianNoise.TITLE,
}
