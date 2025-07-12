import sys
import importlib
import os

import jax
import jax.numpy as jnp
import optax
from flax import serialization

import numpy as np


def get_everything(config_path, args):
    module_path = config_path.replace('/', '.').replace('.py', '')
    module = importlib.import_module(module_path, package=None)
    return module.everything(args)


def save_checkpoint(path, state):
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())


def ema_update(ema_params, new_params, decay):
    return jax.tree_util.tree_map(
        lambda e, p: decay * e + (1.0 - decay) * p,
        ema_params,
        new_params
    )


def make_update_fn(*, apply_fn, optimizer, sampler, ema_decay):
    def update_fn(params, opt_state, images, timesteps, rng, ema_params):
        def loss_fn(params):
            noisy_image, noise = sampler.add_noise(rng, images, timesteps)
            predicted_noise = apply_fn(params, noisy_image, timesteps)

            loss = jnp.sum((predicted_noise - noise) ** 2)

            return loss

        loss, grad = jax.value_and_grad(loss_fn)(params)

        loss, grad = jax.tree_util.tree_map(
            lambda x: jax.lax.pmean(x, axis_name='batch'),
            (loss, grad),
        )

        updates, opt_state = optimizer.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        new_ema_params = ema_update(ema_params, new_params, decay=ema_decay)

        return new_params, opt_state, new_ema_params, loss

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=())


def main(config_path, args):
    evy = get_everything(config_path, args)

    seed = evy['seed']

    train_loader = evy['train_loader']

    loader_iter = iter(train_loader)
    inputs = next(loader_iter)

    model = evy['model']
    optimizer = evy['optimizer']
    sampler = evy['sampler']
    epochs = evy['epochs']

    run = evy['run']

    checkpoint_path = evy['diffusion_path']

    key = jax.random.PRNGKey(seed)
    key, sub_key, other_key = jax.random.split(key, 3)

    fake_timesteps = jax.random.randint(other_key, minval=0, maxval=sampler.total_timesteps, shape=(inputs.shape[0],))
    params = model.init(sub_key, inputs, fake_timesteps)

    opt_state = optimizer.init(params)

    devices = jax.local_devices()
    replicate = lambda tree: jax.device_put_replicated(tree, devices)
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    ema_decay = 0.9999
    ema_params = params
    ema_params_repl = replicate(ema_params)

    update_fn = make_update_fn(apply_fn=model.apply, optimizer=optimizer, sampler=sampler, ema_decay=ema_decay)

    params_repl = replicate(params)
    opt_state_repl = replicate(opt_state)

    del params
    del opt_state

    num_devices = jax.local_device_count()

    state_template = {
        "params": unreplicate(params_repl),
        "opt_state": unreplicate(opt_state_repl),
        "ema_params": unreplicate(ema_params_repl),
        "epoch": 0,
        "rng": key,
    }

    loaded_state = load_checkpoint(checkpoint_path, state_template)
    if loaded_state is not None:
        print("Resuming from checkpoint...")
        params_repl = replicate(loaded_state["params"])
        opt_state_repl = replicate(loaded_state["opt_state"])
        ema_params_repl = replicate(loaded_state["ema_params"])
        key = loaded_state["rng"]
        start_epoch = loaded_state["epoch"] + 1
    else:
        start_epoch = 0

    def shard(x):
        n, *s = x.shape
        return np.reshape(x, (num_devices, n // num_devices, *s))

    def unshard(inputs):
        num_devices, batch_size, *shape = inputs.shape
        return jnp.reshape(inputs, (num_devices * batch_size, *shape))

    for epoch in range(start_epoch, epochs):
        for step, images in enumerate(train_loader):
            key, time_rng, sample_rng = jax.random.split(key, 3)
            timesteps = jax.random.randint(time_rng, minval=0, maxval=sampler.total_timesteps, shape=(images.shape[0],))

            images = jax.tree_util.tree_map(lambda x: shard(np.array(x)), images)
            timesteps = jax.tree_util.tree_map(lambda x: shard(np.array(x)), timesteps)
            rng_shard = jax.random.split(sample_rng, num_devices)

            (
                params_repl,
                opt_state_repl,
                ema_params_repl,
                loss,
            ) = update_fn(
                params_repl,
                opt_state_repl,
                images,
                timesteps,
                rng_shard,
                ema_params_repl,
            )

            loss = unreplicate(loss)

            run.log({
                "total_loss": loss,
                "epoch": epoch})

        checkpoint_state = {
            "params": unreplicate(params_repl),
            "opt_state": unreplicate(opt_state_repl),
            "ema_params": unreplicate(ema_params_repl),
            "epoch": epoch,
            "rng": key,
        }
        save_checkpoint(checkpoint_path, checkpoint_state)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1], sys.argv[2:])
