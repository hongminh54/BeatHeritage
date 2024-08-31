"""
A minimal training script for DiT using PyTorch DDP.
"""
import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LinearLR, CosineAnnealingLR, SequentialLR

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from collections import OrderedDict
from copy import deepcopy
from time import time

from models import DiT_models
from diffusion import create_diffusion

from data_loading import (
    get_data_loader,
    feature_size,
    window_and_relative_time,
    load_and_process_beatmap, BeatmapDatasetIterableFactory,
)


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def requires_grad_non_embed(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model except the embedding table weights.
    """
    for name, param in model.named_parameters():
        if name == "y_embedder.embedding_table.weight":
            continue
        param.requires_grad = flag


def get_scheduler(optimizer: Optimizer, args: DictConfig, accelerator) -> LRScheduler:
    scheduler_p1 = LinearLR(
        optimizer,
        start_factor=0.5,
        end_factor=1,
        total_iters=args.optim.warmup_steps * accelerator.num_processes,
        last_epoch=-1,
    )

    scheduler_p2 = CosineAnnealingLR(
        optimizer,
        T_max=args.optim.total_steps * accelerator.num_processes - args.optim.warmup_steps * accelerator.num_processes,
        eta_min=args.optim.final_cosine,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_p1, scheduler_p2],
        milestones=[args.optim.warmup_steps * accelerator.num_processes],
    )

    return scheduler


#################################################################################
#                                  Training Loop                                #
#################################################################################


@hydra.main(config_path="../configs/diffusion", config_name="v1", version_base="1.1")
def main(args):
    """
    Trains a new DiT model.
    """
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=args.precision,
        gradient_accumulation_steps=args.optim.grad_acc,
        log_with=args.logging.log_with,
        project_config=ProjectConfiguration(
            project_dir="..", logging_dir="tensorboard_logs"
        ),
    )
    accelerator.init_trackers(
        "osu-diffusion",
        init_kwargs={
            "wandb": {
                "entity": "mappingtools",
                "job_type": "training",
                "config": dict(args),
                "mode": args.logging.mode,
            }
        }
    )

    device = accelerator.device
    set_seed(args.seed)

    # Create model:
    model = DiT_models[args.model](
        num_classes=args.data.num_classes,
        context_size=feature_size - 3 + 128,
        class_dropout_prob=0.2,
    ).to(device)
    # Note that parameter initialization is done within the DiT constructor
    ema: torch.nn.Module = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    diffusion = create_diffusion(
        timestep_respacing="",
        noise_schedule=args.noise_schedule,
        use_l1=args.l1_loss,
        diffusion_steps=args.diffusion_steps,
    )  # default: 1000 steps, linear noise schedule
    print(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim.base_lr, weight_decay=args.optim.weight_decay)
    scheduler = get_scheduler(optimizer, args, accelerator)

    # Setup data:
    batch_size = args.optim.batch_size // args.optim.grad_acc // accelerator.num_processes
    loader = get_data_loader(
        dataset_path=args.data.train_dataset_path,
        start=args.data.start,
        end=args.data.end,
        iterable_factory=BeatmapDatasetIterableFactory(
            args.data.seq_len,
            args.data.stride,
            load_and_process_beatmap,
            window_and_relative_time,
        ),
        cycle_length=batch_size // 2,
        batch_size=batch_size,
        num_workers=args.dataloader.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    # Prepare models for training:
    update_ema(
        ema,
        model,
        decay=0,
    )  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # noinspection PyTypeChecker
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    accelerator.register_for_checkpointing(ema)

    # Load checkpoint
    if args.checkpoint_path:
        accelerator.load_state(args.checkpoint_path)

    if args.compile:
        model = torch.compile(model)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    avg_loss = 0
    epoch = 0
    start_time = time()

    print(f"Training for {args.optim.total_steps} steps...")
    while train_steps < args.optim.total_steps:
        print(f"Beginning epoch {epoch}...")
        optimizer.zero_grad(set_to_none=True)

        for (x, o, c), y in loader:
            with accelerator.accumulate(model):
                if train_steps > args.optim.total_steps:
                    break

                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                model_kwargs = dict(o=o, c=c, y=y)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                update_ema(ema, model)

                # Log loss values:
                running_loss += loss.item()

                if accelerator.sync_gradients:
                    log_steps += 1
                    train_steps += 1
                    if train_steps % args.logging.every_steps == 0:
                        # Measure training speed:
                        end_time = time()
                        steps_per_sec = log_steps / (end_time - start_time)
                        avg_loss = running_loss / log_steps
                        print(
                            f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}",
                        )
                        accelerator.log({"train_loss": avg_loss, "steps_per_sec": steps_per_sec}, train_steps)
                        # Reset monitoring variables:
                        running_loss = 0
                        log_steps = 0
                        start_time = time()

                    # Save DiT checkpoint:
                    if train_steps % args.checkpoint.every_steps == 0 and train_steps > 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            output_dir = f"checkpoint-{train_steps}"
                            accelerator.save_state(output_dir=output_dir, safe_serialization=True)
                            print(f"Saved checkpoint to {output_dir}")

        epoch += 1

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    print("Done!")


if __name__ == "__main__":
    main()
