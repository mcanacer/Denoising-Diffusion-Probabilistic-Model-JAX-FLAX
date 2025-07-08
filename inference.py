import sys
import importlib

import torch
import numpy as np

import jax
import jax.numpy as jnp
from flax import serialization
from torchvision.utils import save_image


def get_everything(config_path, args):
    module_path = config_path.replace('/', '.').replace('.py', '')
    module = importlib.import_module(module_path, package=None)
    return module.everything(args)


def generate_samples(rng, model, model_params, sampler, shape):
    def make_predict_fn(*, apply_fn, sampler):
        def predict_fn(params, rng, xt, timesteps):
            predicted_noise = apply_fn(params, xt, timesteps)
            xt_prev = sampler.remove_noise(rng, xt, predicted_noise, timesteps)
            return xt_prev

        return jax.pmap(predict_fn, axis_name='batch', donate_argnums=())

    def shard(x):
        n, *s = x.shape
        return x.reshape((num_devices, n // num_devices, *s))

    def unshard(x):
        d, b, *s = x.shape
        return x.reshape((d * b, *s))

    devices = jax.local_devices()
    num_devices = len(devices)
    replicate = lambda tree: jax.device_put_replicated(tree, devices)
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    predict_fn = make_predict_fn(apply_fn=model.apply, sampler=sampler)

    params_repl = replicate(model_params)
    rng, sample_rng = jax.random.split(rng, 2)

    xt = jax.random.normal(sample_rng, shape=shape)

    for t in reversed(range(sampler.total_timesteps)):
        rng, sample_rng = jax.random.split(rng, 2)
        timesteps = jnp.full(shape=(xt.shape[0],), fill_value=t, dtype=jnp.int32)

        xt = jax.tree_util.tree_map(lambda x: shard(x), xt)
        timesteps = jax.tree_util.tree_map(lambda x: shard(x), timesteps)
        rng_shard = jax.random.split(sample_rng, num_devices)

        xt = predict_fn(params_repl, rng_shard, xt, timesteps)
        xt = jax.tree_util.tree_map(lambda x: unshard(x), xt)

    x0 = xt  # [N, H, W, 3]

    # Rescale from [-1, 1] to [0, 1]
    x0 = (x0 + 1) / 2

    return jnp.clip(x0, 0.0, 1.0)


def main(config_path, args):
    evy = get_everything(config_path, args)

    seed = evy['seed']
    key = jax.random.PRNGKey(seed)

    image_size = evy['image_size']

    model = evy['model']
    sampler = evy['sampler']

    checkpoint_path = evy['diffusion_path']
    num_samples = evy['num_samples']

    shape = (num_samples, image_size, image_size, 3)

    with open(checkpoint_path, 'rb') as f:
        model_params = serialization.from_bytes(None, f.read())

    x_gen = generate_samples(key, model, model_params, sampler, shape)

    for i in range(x_gen.shape[0]):
        img = np.array(x_gen[i])

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        save_image(img, f'/content/drive/MyDrive/DDPM/gen_images/generated_image{i}.png')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1], sys.argv[2:])
