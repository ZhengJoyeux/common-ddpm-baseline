"""
SERS无条件DDPM项目的统一运行入口。

项目结构：

project/
├── run.py
├── configs/
│   └── ddpm_training.yaml
├── scripts/
│   ├── inspect_spectrum_data.py
│   ├── train_ddpm.py
│   └── generate_spectra.py
├── src/
│   ├── configuration_loader.py
│   ├── spectrum_file_reader.py
│   ├── spectrum_dataset.py
│   ├── dataset_splitter.py
│   ├── spectrum_length_adapter.py
│   ├── one_dimensional_ddpm.py
│   ├── model_builder.py
│   ├── random_seed_manager.py
│   ├── training_logger.py
│   ├── checkpoint_manager.py
│   ├── ddpm_trainer.py
│   ├── spectrum_generator.py
│   └── spectrum_exporter.py
└── tests/

常用命令：

1. 可选：检查输入数据
    python run.py inspect

2. 训练模型
    python run.py train

3. 断点续训
    python run.py train
        --resume outputs/checkpoints/latest.pt

4. 生成光谱
    python run.py generate

5. 指定生成50条光谱
    python run.py generate --number 50

6. 运行测试
    python run.py test
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


# ============================================================
# 项目路径
# ============================================================

# run.py所在目录就是项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parent

# src下面直接存放项目核心代码文件。
SOURCE_DIRECTORY = PROJECT_ROOT / "src"

# scripts下面存放检查、训练和生成入口脚本。
SCRIPT_DIRECTORY = PROJECT_ROOT / "scripts"

# 默认配置文件。
DEFAULT_CONFIG_PATH = Path(
    "configs/ddpm_training.yaml"
)

# 默认模型检查点。
DEFAULT_CHECKPOINT_PATH = Path(
    "outputs/checkpoints/latest.pt"
)


# ============================================================
# 路径处理
# ============================================================

def resolve_project_path(
    path_value: str | Path,
) -> Path:
    """
    将输入路径转换成绝对路径。

    如果输入的是相对路径，就将其解释为相对于
    项目根目录的路径。

    例如：

        configs/ddpm_training.yaml

    会转换成：

        项目根目录/configs/ddpm_training.yaml
    """
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def ensure_file_exists(
    file_path: Path,
    file_description: str,
) -> None:
    """
    检查指定文件是否存在。

    Parameters
    ----------
    file_path
        需要检查的文件路径。

    file_description
        用于错误提示的文件名称。
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_description}不存在：{file_path}"
        )

    if not file_path.is_file():
        raise FileNotFoundError(
            f"{file_description}不是一个文件：{file_path}"
        )


def ensure_directory_exists(
    directory_path: Path,
    directory_description: str,
) -> None:
    """
    检查指定目录是否存在。
    """
    if not directory_path.exists():
        raise FileNotFoundError(
            f"{directory_description}不存在：{directory_path}"
        )

    if not directory_path.is_dir():
        raise FileNotFoundError(
            f"{directory_description}不是一个目录："
            f"{directory_path}"
        )


# ============================================================
# 子进程环境
# ============================================================

def build_subprocess_environment() -> dict[str, str]:
    """
    为被调用的脚本准备运行环境。

    你的项目结构是src下面直接存放代码文件，例如：

        src/configuration_loader.py
        src/model_builder.py
        src/ddpm_trainer.py

    因此需要把src目录加入PYTHONPATH。

    加入以后，scripts中的程序就可以这样导入：

        from configuration_loader import load_configuration
        from model_builder import build_diffusion_model
        from ddpm_trainer import DDPMTrainer

    不需要写：

        from sers_ddpm.model_builder import ...

    也不需要创建src/sers_ddpm目录。
    """
    environment = os.environ.copy()

    source_path = str(SOURCE_DIRECTORY)

    existing_python_path = environment.get(
        "PYTHONPATH",
        "",
    )

    if existing_python_path:
        environment["PYTHONPATH"] = (
            source_path
            + os.pathsep
            + existing_python_path
        )
    else:
        environment["PYTHONPATH"] = source_path

    # 让训练日志立即显示在VS Code终端中，
    # 避免日志长时间停留在缓存里。
    environment["PYTHONUNBUFFERED"] = "1"

    return environment


# ============================================================
# 命令执行
# ============================================================

def execute_command(
    command: Sequence[str],
) -> int:
    """
    在项目根目录中执行指定命令。

    使用sys.executable可以保证调用的是当前已激活
    sers_ddpm环境中的Python解释器。
    """
    print()
    print("准备执行命令：")
    print(shlex.join(command))
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"Python解释器：{sys.executable}")
    print()

    completed_process = subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        env=build_subprocess_environment(),
        check=False,
    )

    return int(completed_process.returncode)


# ============================================================
# 检查输入数据
# ============================================================

def run_inspection(
    config_path: Path,
) -> int:
    """
    调用scripts/inspect_spectrum_data.py。

    这个命令只检查输入文件，不修改原始光谱数据。
    """
    script_path = (
        SCRIPT_DIRECTORY
        / "inspect_spectrum_data.py"
    )

    ensure_file_exists(
        script_path,
        "输入数据检查脚本",
    )

    ensure_file_exists(
        config_path,
        "配置文件",
    )

    command = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
    ]

    return execute_command(command)


# ============================================================
# 训练模型
# ============================================================

def run_training(
    config_path: Path,
    resume_path: Path | None,
) -> int:
    """
    调用scripts/train_ddpm.py训练DDPM模型。

    如果resume_path为None，就从头开始训练。

    如果提供resume_path，就从指定检查点继续训练。
    """
    script_path = (
        SCRIPT_DIRECTORY
        / "train_ddpm.py"
    )

    ensure_file_exists(
        script_path,
        "模型训练脚本",
    )

    ensure_file_exists(
        config_path,
        "配置文件",
    )

    command = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
    ]

    if resume_path is not None:
        ensure_file_exists(
            resume_path,
            "断点续训检查点",
        )

        command.extend(
            [
                "--resume",
                str(resume_path),
            ]
        )

    return execute_command(command)


# ============================================================
# 生成光谱
# ============================================================

def run_generation(
    config_path: Path,
    checkpoint_path: Path,
    number_of_spectra: int | None,
) -> int:
    """
    调用scripts/generate_spectra.py生成光谱。

    如果没有通过--number指定数量，就使用
    ddpm_training.yaml中设置的生成数量。
    """
    script_path = (
        SCRIPT_DIRECTORY
        / "generate_spectra.py"
    )

    ensure_file_exists(
        script_path,
        "光谱生成脚本",
    )

    ensure_file_exists(
        config_path,
        "配置文件",
    )

    ensure_file_exists(
        checkpoint_path,
        "模型检查点",
    )

    command = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
    ]

    if number_of_spectra is not None:
        if number_of_spectra <= 0:
            raise ValueError(
                "生成光谱的数量必须大于0。"
            )

        command.extend(
            [
                "--number",
                str(number_of_spectra),
            ]
        )

    return execute_command(command)


# ============================================================
# 运行测试
# ============================================================

def run_tests(
    pytest_arguments: Sequence[str],
) -> int:
    """
    使用当前Python环境运行pytest。

    默认命令：

        python -m pytest -q

    也可以继续向pytest传递参数，例如：

        python run.py test -k mini_training
    """
    command = [
        sys.executable,
        "-m",
        "pytest",
    ]

    if pytest_arguments:
        arguments = list(pytest_arguments)

        # 同时允许下面这种写法：
        #
        # python run.py test -- -q
        if arguments[0] == "--":
            arguments = arguments[1:]

        command.extend(arguments)

    else:
        command.append("-q")

    return execute_command(command)


# ============================================================
# 命令行参数
# ============================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """
    创建run.py的命令行参数解析器。
    """
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "SERS无条件DDPM项目统一运行入口"
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="可用命令",
    )

    # --------------------------------------------------------
    # inspect命令
    # --------------------------------------------------------

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="可选：检查输入光谱，不修改原始数据",
    )

    inspect_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=(
            "配置文件路径。默认："
            "configs/ddpm_training.yaml"
        ),
    )

    # --------------------------------------------------------
    # train命令
    # --------------------------------------------------------

    train_parser = subparsers.add_parser(
        "train",
        help="训练无条件DDPM模型",
    )

    train_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=(
            "配置文件路径。默认："
            "configs/ddpm_training.yaml"
        ),
    )

    train_parser.add_argument(
        "--resume",
        default=None,
        help=(
            "断点续训使用的检查点路径。"
            "不填写时从头开始训练"
        ),
    )

    # --------------------------------------------------------
    # generate命令
    # --------------------------------------------------------

    generate_parser = subparsers.add_parser(
        "generate",
        help="使用训练好的DDPM模型生成光谱",
    )

    generate_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=(
            "配置文件路径。默认："
            "configs/ddpm_training.yaml"
        ),
    )

    generate_parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help=(
            "模型检查点路径。默认："
            "outputs/checkpoints/latest.pt"
        ),
    )

    generate_parser.add_argument(
        "--number",
        type=int,
        default=None,
        help=(
            "需要生成的光谱数量。"
            "不填写时使用YAML配置中的数量"
        ),
    )

    # --------------------------------------------------------
    # test命令
    # --------------------------------------------------------

    subparsers.add_parser(
        "test",
        help="运行项目测试",
    )

    return parser


# ============================================================
# 主函数
# ============================================================

def main() -> None:
    """
    run.py主函数。
    """
    parser = create_argument_parser()

    # parse_known_args允许test命令继续向pytest传递参数。
    arguments, extra_arguments = parser.parse_known_args()

    try:
        # 在正式执行前检查项目的基础目录。
        ensure_directory_exists(
            SOURCE_DIRECTORY,
            "src核心代码目录",
        )

        ensure_directory_exists(
            SCRIPT_DIRECTORY,
            "scripts入口脚本目录",
        )

        if arguments.command == "inspect":
            if extra_arguments:
                parser.error(
                    "inspect命令存在无法识别的参数："
                    + " ".join(extra_arguments)
                )

            config_path = resolve_project_path(
                arguments.config
            )

            exit_code = run_inspection(
                config_path=config_path,
            )

        elif arguments.command == "train":
            if extra_arguments:
                parser.error(
                    "train命令存在无法识别的参数："
                    + " ".join(extra_arguments)
                )

            config_path = resolve_project_path(
                arguments.config
            )

            if arguments.resume:
                resume_path = resolve_project_path(
                    arguments.resume
                )
            else:
                resume_path = None

            exit_code = run_training(
                config_path=config_path,
                resume_path=resume_path,
            )

        elif arguments.command == "generate":
            if extra_arguments:
                parser.error(
                    "generate命令存在无法识别的参数："
                    + " ".join(extra_arguments)
                )

            config_path = resolve_project_path(
                arguments.config
            )

            checkpoint_path = resolve_project_path(
                arguments.checkpoint
            )

            exit_code = run_generation(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                number_of_spectra=arguments.number,
            )

        elif arguments.command == "test":
            exit_code = run_tests(
                pytest_arguments=extra_arguments,
            )

        else:
            parser.error(
                f"未知命令：{arguments.command}"
            )
            return

    except FileNotFoundError as error:
        print(
            f"\n文件或目录错误：{error}",
            file=sys.stderr,
        )
        sys.exit(2)

    except ValueError as error:
        print(
            f"\n参数错误：{error}",
            file=sys.stderr,
        )
        sys.exit(2)

    except KeyboardInterrupt:
        print(
            "\n程序已由用户手动终止。",
            file=sys.stderr,
        )
        sys.exit(130)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()