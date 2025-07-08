import jax
import jax.numpy as jnp


class Sampler:
    def __init__(self, total_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.total_timesteps = total_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Linear beta schedule
        self.beta = jnp.linspace(beta_start, beta_end, total_timesteps, dtype=jnp.float32)  # [T]
        self.alpha = 1.0 - self.beta  # [T]
        self.alpha_cum_prod = jnp.cumprod(self.alpha)  # [T]

        self.alpha_cum_prod_prev = jnp.concatenate([
            jnp.array([1.0], dtype=jnp.float32),
            self.alpha_cum_prod[:-1]
        ])  # [T]

    def add_noise(self, rng, x0, timesteps):
        """q(x_t | x_0) forward process"""
        alpha_bar = self.alpha_cum_prod[timesteps]  # [N]
        noise = jax.random.normal(rng, shape=x0.shape)  # [N, H, W, C]

        alpha_bar = jnp.expand_dims(alpha_bar, axis=(1, 2, 3))  # [N, 1, 1, 1]
        noisy_image = jnp.sqrt(alpha_bar) * x0 + jnp.sqrt(1.0 - alpha_bar) * noise

        return noisy_image, noise

    def remove_noise(self, rng, xt, predicted_noise, timesteps):
        """p(x_{t-1} | x_t) reverse process"""

        beta_t = jnp.expand_dims(self.beta[timesteps], axis=(1, 2, 3))  # [N, 1, 1, 1]
        alpha_t = jnp.expand_dims(self.alpha[timesteps], axis=(1, 2, 3))  # [N, 1, 1, 1]
        alpha_bar_t = jnp.expand_dims(self.alpha_cum_prod[timesteps], axis=(1, 2, 3))  # [N, 1, 1, 1]
        alpha_bar_prev_t = jnp.expand_dims(self.alpha_cum_prod_prev[timesteps], axis=(1, 2, 3))  # [N, 1, 1, 1]

        # Compute posterior mean
        coef1 = 1.0 / jnp.sqrt(alpha_t)
        coef2 = beta_t / jnp.sqrt(1.0 - alpha_bar_t)
        mean = coef1 * (xt - coef2 * predicted_noise)

        # Compute posterior variance
        var = beta_t * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)
        log_var = jnp.log(jnp.clip(var, a_min=1e-20))

        # Sample noise
        noise = jax.random.normal(rng, shape=xt.shape)

        # If t == 0, just return mean
        denoised = mean + jnp.exp(0.5 * log_var) * noise
        denoised = jnp.where(jnp.expand_dims(timesteps, axis=(1, 2, 3)) == 0, mean, denoised)

        return denoised
