"""Interface to the installed one-dimensional DDPM implementation."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
    from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import (
        GaussianDiffusion1D,
        Unet1D,
    )
except ImportError as error:
    raise ImportError(
        "无法导入Unet1D和GaussianDiffusion1D。"
        "请在sers_ddpm环境中安装"
        "denoising-diffusion-pytorch==2.2.6。"
    ) from error


EXPECTED_BACKEND_VERSION = "2.2.6"


def get_backend_version() -> str:
    """Return the installed diffusion package version."""

    try:
        return version("denoising-diffusion-pytorch")
    except PackageNotFoundError:
        return "not-installed"


def check_backend_version() -> None:
    """Raise an error if the installed version is not the expected version."""

    installed_version = get_backend_version()

    if installed_version != EXPECTED_BACKEND_VERSION:
        raise RuntimeError(
            "第三方扩散包版本不一致："
            f"当前为{installed_version}，"
            f"项目要求{EXPECTED_BACKEND_VERSION}。"
        )


__all__ = [
    "Unet1D",
    "GaussianDiffusion1D",
    "get_backend_version",
    "check_backend_version",
]