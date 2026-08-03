"""Manage random seeds for reproducible experiments."""

import os
import random

import numpy as np
import torch


def set_random_seed(
    random_seed: int,
    deterministic: bool = False,
) -> None:
    """Set Python, NumPy and PyTorch random seeds."""

    os.environ["PYTHONHASHSEED"] = str(random_seed)

    random.seed(random_seed)
    np.random.seed(random_seed)

    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    if deterministic:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def seed_data_loader_worker(worker_id: int) -> None:
    """Set NumPy and Python seeds inside each DataLoader worker."""

    del worker_id

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_data_loader_generator(
    random_seed: int,
) -> torch.Generator:
    """Create a seeded generator used for DataLoader shuffling."""

    generator = torch.Generator()
    generator.manual_seed(random_seed)
    return generator