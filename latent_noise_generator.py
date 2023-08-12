import json
import torch
import numpy as np

def gaussian_latent_noise(width=64, height=64, ver="v1", seed=-1, fac=0.6, nul=0.0, srnd=True):
	limit = {
		"v1": {
			"min": {"A": -5.5618, "B":-17.1368, "C":-10.3445, "D": -8.6218},
			"max": {"A": 13.5369, "B": 11.1997, "C": 16.3043, "D": 10.6343},
			"nul": {"A": -5.3870, "B":-14.2931, "C":  6.2738, "D":  7.1220},
		},
		"xl": {
			"min": {"A":-22.2127, "B":-20.0131, "C":-17.7673, "D":-14.9434},
			"max": {"A": 17.9334, "B": 26.3043, "C": 33.1648, "D":  8.9380},
			"nul": {"A":-21.9287, "B":  3.8783, "C":  2.5879, "D":  2.5435},
		}
	}
	# seed
	if seed >= 0: torch.manual_seed(seed)

	limit = limit[ver]
	if srnd: # shared random
		rand = torch.rand([width,height])
		lat = torch.stack([
			(limit["min"]["A"] + torch.clone(rand)*(limit["max"]["A"]-limit["min"]["A"])),
			(limit["min"]["B"] + torch.clone(rand)*(limit["max"]["B"]-limit["min"]["B"])),
			(limit["min"]["C"] + torch.clone(rand)*(limit["max"]["C"]-limit["min"]["C"])),
			(limit["min"]["D"] + torch.clone(rand)*(limit["max"]["D"]-limit["min"]["D"])),
		])
	else: # separate random
		lat = torch.stack([
			(limit["min"]["A"] + torch.rand([width,height])*(limit["max"]["A"]-limit["min"]["A"])),
			(limit["min"]["B"] + torch.rand([width,height])*(limit["max"]["B"]-limit["min"]["B"])),
			(limit["min"]["C"] + torch.rand([width,height])*(limit["max"]["C"]-limit["min"]["C"])),
			(limit["min"]["D"] + torch.rand([width,height])*(limit["max"]["D"]-limit["min"]["D"])),
		])
	tnul = torch.stack([ # black image
		torch.ones([width, height])*limit["nul"]["A"],
		torch.ones([width, height])*limit["nul"]["B"],
		torch.ones([width, height])*limit["nul"]["C"],
		torch.ones([width, height])*limit["nul"]["D"],
	])
	out = ((lat*fac)*(1.0-nul) + tnul*nul)/2
	return out
