# Denoising Diffusion Probabilistic Models (DDPM) — JAX/Flax Implementation

This repository contains a from-scratch implementation of the paper:

> ** Denoising Diffusion Probabilistic Models**  
> (https://arxiv.org/abs/2006.11239)

## 🏁 Training

```bash
python train.py config.py --diffusion-path path/to/diffusion.pkl
```

## 🎨 Inference

```bash
python inference.py config.py --diffusion-path path/to/diffusion.pkl
```

## 🖼 Sample Generated Images From LSUN/Bedrooms

![Generated Image](gen_images/generated_image0.png)
![Generated Image](gen_images/generated_image4.png)
![Generated Image](gen_images/generated_image10.png)
![Generated Image](gen_images/generated_image24.png)
