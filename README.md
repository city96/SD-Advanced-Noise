# Colored Noise / Advanced Noise / Latent Noise Experiments

It is only fair that I shout out that all the other noise related repos such ash [ComfyUI_Noise](https://github.com/BlenderNeko/ComfyUI_Noise), [noise_latent_perlinpinpin](https://github.com/Extraltodeus/noise_latent_perlinpinpin) and [comfy-plasma](https://github.com/Jordach/comfy-plasma). I think the [WAS node suite](https://github.com/WASasquatch/was-node-suite-comfyui) also has quite a few noise-related nodes.

While messing around with the stable diffusion VAE, I noticed the latent space behaves mostly linearly. I figured it should be possible to generate latent noise directly if I map out the per channel limits and give the whole thing an offset.

This repo also has some decoders, but those are a proof of concept tier at best. [TAESD](https://github.com/madebyollin/taesd) is better in every way.

## LatentGaussianNoise

I'm not sure if the way I coded this even makes sense, or if it's even gaussian noise. It's just `torch.random` with a bunch of stuff like scaling/per channel random/etc.

![LATENT_SPACE_NOISE](https://github.com/city96/SD-Advanced-Noise/assets/125218114/a3b1d790-4632-4290-b450-ec0919a8265c)

## Linearity / linear_encoder

This is the simplest encoder. It uses the fact that a change in the RGB channels creates a (mostly?) linear change in the 4 latents channels (which I just called A/B/C/D since I couldn't find any info about them).

The next step would be to plot at a higher precision and fit them onto a polynomial. 

![LINEAR_ENCODER](https://github.com/city96/SD-Advanced-Noise/assets/125218114/f68b7e48-8def-480b-93a4-3f1843ba492c)

![NL](https://github.com/city96/SD-Advanced-Noise/assets/125218114/0fbffd6f-b062-441d-aa14-764249216926)
