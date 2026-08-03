import torch

from src.model_builder import (
    build_diffusion_model,
)


def test_ddpm_forward_loss():
    model_configuration = {
        "channels": 1,
        "base_dimension": 8,
        "dimension_multipliers": [1, 2],
        "self_condition": False,
        "dropout": 0.0,
        "diffusion_timesteps": 10,
        "sampling_timesteps": 10,
        "objective": "pred_noise",
        "beta_schedule": "cosine",
        "ddim_sampling_eta": 0.0,
        "auto_normalize": True,
    }

    _, diffusion = build_diffusion_model(
        model_configuration=model_configuration,
        sequence_length=16,
    )

    batch = torch.rand(2, 1, 16)
    loss = diffusion(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)