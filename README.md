# SERS Ordinary 1D DDPM

本项目用于训练一个普通、无条件的一维去噪扩散概率模型
（Denoising Diffusion Probabilistic Model，DDPM），生成预处理后的
SERS 拉曼光谱。

该模型仅作为后续条件扩散模型、先验残差扩散模型和物理规则约束扩散模型
的基线对照。

## 1. 模型范围

当前项目包含：

- Raman 数据读取；
- SG 平滑；
- ALS 基线校正；
- 基于训练集统计量的全局 Min-Max 归一化；
- 光谱长度自动补齐；
- 一维 U-Net；
- 普通 Gaussian DDPM；
- EMA 模型；
- 混合精度训练；
- 单卡或双卡训练；
- 断点续训；
- 光谱批量生成；
- CSV、Excel 和 NPZ 输出。

当前项目不包含：

- 农药种类条件；
- 农药浓度条件；
- 水体和土壤基质条件；
- 先验光谱；
- 残差扩散；
- 物理规则损失；
- CNN 或 SERSformer 识别和定量模型。

因此，该模型会把所有训练光谱看成来自同一个总体分布。数据文件中的
列注释会被保存，但不会输入扩散模型。

## 2. 输入文件格式

推荐使用宽表格式的 Excel 或 CSV 文件。

第一列是 Raman shift，后续每列是一条光谱，第一行是列名或光谱注释：

| Raman_shift_cm-1 | SA0001_R01 | SA0001_R02 | SA0002_R01 |
| ---------------: | ---------: | ---------: | ---------: |
|              400 |      125.3 |      118.7 |       96.2 |
|              401 |      128.4 |      121.6 |       98.5 |
|              402 |      130.1 |      123.2 |      101.4 |

支持以下格式：

- `.xlsx`
- `.xls`
- `.csv`
- `.npz`

对于 NPZ 文件，推荐包含：

```python
raman_shift  # [L]
spectra      # [N, L] 或 [L, N]
annotations  # 可选
```

如果 `data.path` 指向一个文件，则只读取该文件。

如果 `data.path` 指向一个文件夹，则读取文件夹中所有受支持的数据文件。

不同文件的拉曼位移轴不完全相同时，程序会把后续文件插值到第一个文件
的拉曼位移轴上。插值不会超出原始光谱的位移范围。

## 3. 数据预处理

默认预处理顺序为：

```text
原始光谱
→ SG 平滑
→ ALS 基线校正
→ 全局 Min-Max 归一化
→ 补齐到 U-Net 要求的长度
→ DDPM
```

全局 Min-Max 参数只根据训练集计算，验证集不参与参数拟合。

这里不采用每条光谱单独 Min-Max 归一化，因为逐条归一化会将每条光谱
都强制缩放到相同强度范围，从而破坏与农药浓度相关的整体强度差异。

生成光谱导出到 `preprocessed` 域时，程序会撤销全局 Min-Max 缩放，
但不能恢复已经去掉的基线，也不能撤销 SG 平滑。因此输出表示的是：

```text
SG 平滑 + ALS 基线校正后的强度
```

## 4. 环境

服务器个人环境：

```text
/home/wqzheng/.conda/envs/sers_ddpm
```

每次重新连接服务器后执行：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /home/wqzheng/.conda/envs/sers_ddpm
cd /home/wqzheng/projects/sers_ddpm
```

检查环境：

```bash
which python
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

正确结果应包括：

```text
/home/wqzheng/.conda/envs/sers_ddpm/bin/python
2.11.0+cu128
True
```

## 5. 安装项目

进入项目根目录：

```bash
cd /home/wqzheng/projects/sers_ddpm
```

安装项目本身：

```bash
python -m pip install -e .
```

检查第三方依赖：

```bash
python -m pip check
```

## 6. 检查输入数据

先在 `configs/default.yaml` 中设置：

```yaml
data:
  path: data/raw
```

然后执行：

```bash
python run.py inspect --config configs/default.yaml
```

程序会输出：

- 数据文件数量；
- 光谱数量；
- 拉曼位移点数；
- 拉曼位移范围；
- 强度范围；
- 模型补齐后的长度；
- 部分光谱注释。

## 7. 运行测试

快速测试：

```bash
pytest -q -m "not slow"
```

迷你训练测试：

```bash
pytest -q -m slow
```

全部测试：

```bash
pytest -q
```

## 8. 单张 GPU 训练

```bash
python run.py train --config configs/default.yaml
```

## 9. 两张 GPU 训练

第一次使用 Accelerate 时可以运行：

```bash
accelerate config
```

也可以直接启动两张显卡：

```bash
accelerate launch \
  --num_processes 2 \
  run.py train \
  --config configs/default.yaml
```

`batch_size` 默认表示整个训练任务的总批量大小。因为配置中：

```yaml
split_batches: true
```

所以总批量大小为16时，两张 GPU 通常各处理8条光谱。

## 10. 断点续训

指定检查点：

```bash
python run.py train \
  --config configs/default.yaml \
  --resume outputs/ordinary_ddpm/checkpoints/step_00005000.pt
```

从输出目录中最新的检查点恢复：

```bash
python run.py train \
  --config configs/default.yaml \
  --resume latest
```

双卡断点续训：

```bash
accelerate launch \
  --num_processes 2 \
  run.py train \
  --config configs/default.yaml \
  --resume latest
```

## 11. 生成光谱

指定模型检查点生成50条光谱：

```bash
python run.py sample \
  --config configs/default.yaml \
  --checkpoint outputs/ordinary_ddpm/checkpoints/step_00100000.pt \
  --num-samples 50 \
  --output outputs/ordinary_ddpm/generated/ddpm_generated_50.xlsx
```

使用最新检查点：

```bash
python run.py sample \
  --config configs/default.yaml \
  --checkpoint latest \
  --num-samples 50 \
  --output outputs/ordinary_ddpm/generated/ddpm_generated_50.xlsx
```

生成结果包括：

```text
ddpm_generated_50.xlsx
ddpm_generated_50.npz
ddpm_generated_50.png
```

Excel 第一列仍为 Raman shift，后面每列为一条生成光谱。

## 12. 输出目录

训练后目录大致为：

```text
outputs/ordinary_ddpm/
├── resolved_config.yaml
├── checkpoints/
│   ├── latest.txt
│   ├── step_00005000.pt
│   └── step_00100000.pt
├── previews/
│   ├── step_00005000.npz
│   ├── step_00005000.png
│   └── ...
└── generated/
    ├── ddpm_generated_50.xlsx
    ├── ddpm_generated_50.npz
    └── ddpm_generated_50.png
```

## 13. 关于 lucidrains 代码

本项目固定使用：

```text
denoising-diffusion-pytorch==2.2.6
```

项目中的 `src/sers_ddpm/lucidrains_1d.py` 只负责导入并检查：

```python
Unet1D
GaussianDiffusion1D
```

不要将 GitHub `main` 分支的不同版本代码直接覆盖到当前环境中。
