import argparse
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from datasets import load_dataset
from dataset import HuggingFace
import wandb
from model import UNet
from sampler import Sampler
import optax


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)

    # Dataset
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)

    # Model
    parser.add_argument('--channel', type=int, default=128)
    parser.add_argument('--num_res_block', type=int, default=2)
    parser.add_argument('--attn_heads', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--num-epochs', type=int, default=100)

    # Sampler
    parser.add_argument('--total-timesteps', type=int, default=250)
    parser.add_argument('--beta-start', type=float, default=0.0001)
    parser.add_argument('--beta-end', type=float, default=0.02)

    # Wandb
    parser.add_argument('--project', type=str, default='DDPM')
    parser.add_argument('--name', type=str, default='run_standard')

    # Save
    parser.add_argument('--diffusion-path', type=str, required=True)

    # Inference
    parser.add_argument('--num-samples', type=int, default=8)

    return parser.parse_args(args)


def everything(args):
    args = parse_args(args)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),  # Normalize [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale [-1, 1]
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # Convert [C, H, W] to [H, W, C]
        transforms.Lambda(lambda x: x.numpy()),
    ])

    train_dataset = HuggingFace(
      dataset=load_dataset("bitmind/ffhq-256", split='train'),
      transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = UNet(
        channel_multiplier=[1, 1, 2, 2, 4, 4],
        attn_strides=[16],
        channel=args.channel,
        n_res_block=args.num_res_block,
        attn_heads=args.attn_heads,
    )

    optimizer = optax.chain(optax.adam(
        learning_rate=args.learning_rate,
        b1=0.9,
        b2=0.99,
        eps=1e-8,
    ))

    sampler = Sampler(
        total_timesteps=args.total_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )

    epochs = args.num_epochs

    run = wandb.init(
        project=args.project,
        name=args.name,
        reinit=True,
        config=vars(args)
    )

    return {
        'seed': args.seed,
        'image_size': args.image_size,
        'train_loader': train_loader,
        'model': model,
        'optimizer': optimizer,
        'sampler': sampler,
        'epochs': epochs,
        'run': run,
        'diffusion_path': args.diffusion_path,
        'num_samples': args.num_samples,
    }
