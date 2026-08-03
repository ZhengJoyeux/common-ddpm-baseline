import numpy as np
import torch
from torch.utils.data import DataLoader

from src.checkpoint_manager import (
    CheckpointManager,
)
from src.ddpm_trainer import DdpmTrainer
from src.model_builder import (
    build_diffusion_model,
)
from src.spectrum_dataset import SpectrumDataset
from src.training_logger import TrainingLogger


def test_two_step_training(tmp_path):
    random_generator = np.random.default_rng(42)

    training_spectra = random_generator.random(
        (8, 16),
        dtype=np.float32,
    )
    validation_spectra = random_generator.random(
        (4, 16),
        dtype=np.float32,
    )

    training_loader = DataLoader(
        SpectrumDataset(training_spectra),
        batch_size=4,
        shuffle=False,
    )
    validation_loader = DataLoader(
        SpectrumDataset(validation_spectra),
        batch_size=4,
        shuffle=False,
    )

    model_configuration = {
        "channels": 1,
        "base_dimension": 8,
        "dimension_multipliers": [1, 2],
        "self_condition": False,
        "dropout": 0.0,
        "diffusion_timesteps": 10,
        "sampling_timesteps": 5,
        "objective": "pred_noise",
        "beta_schedule": "cosine",
        "ddim_sampling_eta": 0.0,
        "auto_normalize": True,
    }

    configuration = {
        "model": model_configuration,
        "training": {
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "total_training_steps": 2,
            "gradient_accumulation_steps": 1,
            "maximum_gradient_norm": 1.0,
            "use_mixed_precision": False,
            "ema_decay": 0.995,
            "ema_update_every": 1,
            "log_every_steps": 1,
            "validate_every_steps": 1,
            "checkpoint_every_steps": 1,
            "maximum_validation_batches": 0,
        },
    }

    _, diffusion = build_diffusion_model(
        model_configuration=model_configuration,
        sequence_length=16,
    )

    checkpoint_manager = CheckpointManager(
        tmp_path / "checkpoints"
    )
    logger = TrainingLogger(
        tmp_path / "training.csv"
    )

    trainer = DdpmTrainer(
        diffusion=diffusion,
        training_loader=training_loader,
        validation_loader=validation_loader,
        device=torch.device("cpu"),
        configuration=configuration,
        metadata={
            "original_length": 16,
            "required_multiple": 2,
            "padded_length": 16,
            "padding_size": 0,
            "padding_mode": "edge",
            "raman_shift": list(range(16)),
        },
        checkpoint_manager=checkpoint_manager,
        logger=logger,
    )

    trainer.train()

    assert (
        tmp_path / "checkpoints" / "latest.pt"
    ).exists()
    assert (
        tmp_path / "checkpoints" / "best.pt"
    ).exists()