import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D
from trainer1D import Trainer1D #custom trainer1D for better dataset
from omegaconf import OmegaConf
import os, sys

def build_dataset(config):
    dataset_config = config.dataset.params
    if dataset_config.name == 'breath':
        from data.breath.lazy_loader import make_dataset
        dataset = make_dataset(
            split="train",
            data_dir=dataset_config.data_dir,
            return_labels=False
        )
        return dataset

    elif dataset_config.name == 'defense':
        from data.defense import data_loading
        from torch.utils.data import DataLoader
        train_dataset = data_loading.build_dataset(
                    dataset_dir_path=dataset_config.data_dir,
                    input_dim=88300,
                    train_val_or_test='train',
                    return_labels=False
        ) 
        train_dataloader = DataLoader(
                                train_dataset, 
                                batch_size=config.training.per_gpu_batch_size,
                                num_workers=dataset_config.num_workers_per_gpu,
                                shuffle=True,
                                pin_memory=True,
                                persistent_workers=True
        )
        return train_dataloader
    else:
        raise ValueError(f"Unknown dataset name. Please choose a valid dataset")
    


def get_config():
    cli_conf = OmegaConf.from_cli()
    if not hasattr(cli_conf, 'run') or cli_conf.run is None:
        print("Error: Missing required argument --run")
        sys.exit(1)
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

def main():

    # ------- CONFIG ------- #
    config = get_config()
    print(config.keys())


    # ------- UNET ------- #
    model = Unet1D(
        dim = config.model.unet.dim,
        dim_mults = config.model.unet.dim_mults,
        channels = config.model.unet.channels,
    )

    # ------- DIFFUSION ------- #
    diffusion = GaussianDiffusion1D(
        model,
        seq_length = config.model.diffusion.seq_length,
        timesteps = config.model.diffusion.timesteps,
        objective = config.model.diffusion.objective
    )
    trainable_params = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
    print(f'diffusion parameters: {trainable_params:,}')

    # ------- DATASET ------- #
    dataset = build_dataset(config)

    # ------- TRAINER1D ------- #
    trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = config.training.per_gpu_batch_size,
    num_workers=config.dataset.params.num_workers_per_gpu,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    )

    # ------- TRAIN ------- #
    #trainer.train()



if __name__ == "__main__":
    main()
