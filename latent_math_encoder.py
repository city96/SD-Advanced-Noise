#
# These are currently near-useless, but at least they're instant.
#
import json
import torch

def linear_encoder(img, ver="v1", weights="./linear_weights.json"):
	"""Encodes tensor RGB[3,W,H](0.0-1.0) into tensor LATENT[4,W,H]"""
	with open(weights) as f:
		w = json.load(f)
	w = w[ver]
	lat = torch.stack([
		( # A channel
			(img[0]*w["A"]["R"]) +  # R
			(img[1]*w["A"]["G"]) +  # G
			(img[2]*w["A"]["B"]) +  # B
			w["A"]["C"]             # Constant
		),( # B channel
			(img[0]*w["B"]["R"]) +  # R
			(img[1]*w["B"]["G"]) +  # G
			(img[2]*w["B"]["B"]) +  # B
			w["B"]["C"]             # Constant
		),( # C channel
			(img[0]*w["C"]["R"]) +  # R
			(img[1]*w["C"]["G"]) +  # G
			(img[2]*w["C"]["B"]) +  # B
			w["C"]["C"]             # Constant
		),( # D channel
			(img[0]*w["D"]["R"]) +  # R
			(img[1]*w["D"]["G"]) +  # G
			(img[2]*w["D"]["B"]) +  # B
			w["D"]["C"]             # Constant
		),
	])
	return lat
