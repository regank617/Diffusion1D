import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# ---- Helpers for distributed awareness ----
def is_dist_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def get_rank() -> int:
    return torch.distributed.get_rank() if is_dist_initialized() else 0

def get_world_size() -> int:
    return torch.distributed.get_world_size() if is_dist_initialized() else 1

# ---- Worker seeding ----
def make_worker_init_fn(base_seed: int, epoch: int = 0):
    def worker_init_fn(worker_id: int):
        rank = get_rank()
        seed = base_seed + epoch * 100000 + rank * 1000 + worker_id
        random.seed(seed)
        torch.manual_seed(seed)
    return worker_init_fn

class NumpyRowDataset(Dataset):
    def __init__(self, data_dir, return_labels=True):
        # Memory-map the numpy arrays
        self.data = np.load(os.path.join(data_dir,'data.npy'), mmap_mode='r')
        self.classes = np.load(os.path.join(data_dir,'class.npy'), mmap_mode='r')
        self.ids = np.load(os.path.join(data_dir,'id.npy'), mmap_mode='r')
        self.clusters = np.load(os.path.join(data_dir,'cluster.npy'), mmap_mode='r')
        self.return_labels = return_labels

        assert len(self.data) == len(self.classes) == len(self.ids) == len(self.clusters)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Copy the row to make it writable
        x = torch.from_numpy(self.data[idx].copy()).float()
        if self.return_labels:
            y_class = torch.tensor(self.classes[idx]).long()
            y_id = torch.tensor(self.ids[idx]).long()
            y_cluster = torch.tensor(self.clusters[idx]).long()
            return x, (y_class, y_id, y_cluster)
        else:
            return x




def make_dataset(
    split: str,
    data_dir: str = "shards",
    return_labels: bool = True,
):
    data_path = os.path.join(data_dir,split)
    dataset = NumpyRowDataset(data_path,return_labels)
    return dataset

from data.breath.collate import NumpyCollator
def get_train_dataloader(
    batch_size: int = 256,
    num_workers: int = 0,
    data_dir: str = "shards",
    multi_gpu: bool = False,
    return_labels: bool = True
) -> DataLoader:

    dataset = make_dataset(
        split="train",
        data_dir=data_dir,
        return_labels=return_labels
    )

    collator = NumpyCollator()#collator_config, mask_generator)
    #worker_init = make_worker_init_fn(base_seed, epoch=epoch)
    if multi_gpu:
        sampler = DistributedSampler(dataset, shuffle=True)  # ensures each GPU gets a unique slice
        shuffle=False
    else:
        sampler = None
        shuffle=True
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,  # DistributedSampler for multi-GPU
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
        shuffle=shuffle
    
    )
    return loader

def get_val_dataloader(
    batch_size: int = 256,
    num_workers: int = 0,
    data_dir: str = "shards",
    multi_gpu: bool = False,
    return_labels: bool = True
) -> DataLoader:

    dataset = make_dataset(
        split="val",
        data_dir=data_dir,
        return_labels=return_labels
    )

    collator = NumpyCollator()#collator_config, mask_generator)
    #worker_init = make_worker_init_fn(base_seed, epoch=epoch)
    if multi_gpu:
        sampler = DistributedSampler(dataset, shuffle=False)  # ensures each GPU gets a unique slice
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,  # DistributedSampler for multi-GPU
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator  
    
    )
    return loader

def get_test_dataloader(
    batch_size: int = 256,
    num_workers: int = 0,
    data_dir: str = "shards",
    multi_gpu: bool = False,
    return_labels: bool = True
) -> DataLoader:

    dataset = make_dataset(
        split="test",
        data_dir=data_dir,
        return_labels=return_labels
    )

    collator = NumpyCollator()#collator_config, mask_generator)
    #worker_init = make_worker_init_fn(base_seed, epoch=epoch)
    if multi_gpu:
        sampler = DistributedSampler(dataset, shuffle=False)  # ensures each GPU gets a unique slice
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,  # DistributedSampler for multi-GPU
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator  
    
    )
    return loader
