import torch

from src.model_builder import (
    build_diffusion_model,
)


def test_spectrum_generation_shape():
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

    _, diffusion = build_diffusion_model(
        model_configuration=model_configuration,
        sequence_length=16,
    )

    with torch.inference_mode():
        generated = diffusion.sample(batch_size=2)

    assert generated.shape == (2, 1, 16)
    assert torch.isfinite(generated).all()