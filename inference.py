import sys
import importlib
import os

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


def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(None, f.read())


def generate_samples(rng, model, model_params, sampler, shape, num_steps):
    def make_predict_fn(*, apply_fn, sampler):
        def predict_fn(params, rng, xt, t, t_prev):
            predicted_noise = apply_fn(params, xt, t)
            xt_prev = sampler.remove_noise(rng, xt, predicted_noise, t, t_prev)
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
    timesteps = jnp.arange(0, sampler.total_timesteps, sampler.total_timesteps // num_steps)[::-1]
    timesteps_prev = jnp.concatenate([timesteps[1:], jnp.array([0], dtype=jnp.int32)], axis=0)

    for i in range(len(timesteps)):
        rng, sample_rng = jax.random.split(rng, 2)
        t = jnp.full((shape[0],), timesteps[i], dtype=jnp.int32)
        t_prev = jnp.full((shape[0],), timesteps_prev[i], dtype=jnp.int32)

        xt = jax.tree_util.tree_map(lambda x: shard(x), xt)
        t = jax.tree_util.tree_map(lambda x: shard(x), t)
        t_prev = jax.tree_util.tree_map(lambda x: shard(x), t_prev)
        rng_shard = jax.random.split(sample_rng, num_devices)

        xt = predict_fn(params_repl, rng_shard, xt, t, t_prev)
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
    num_steps = evy['num_steps']
    num_samples = evy['num_samples']

    shape = (num_samples, image_size, image_size, 3)

    loaded_state = load_checkpoint(checkpoint_path)
    params = loaded_state["params"]
    ema_params = loaded_state["ema_params"]

    x_gen = generate_samples(key, model, ema_params, sampler, shape, num_steps)

    for i in range(x_gen.shape[0]):
        img = np.array(x_gen[i])

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        save_image(img, f'/content/drive/MyDrive/DDPM/gen_images/generated_image{i}.png')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1], sys.argv[2:])
