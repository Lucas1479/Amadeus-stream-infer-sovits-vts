# 弃用文件清单

本文件记录了所有已标记为弃用（`.deprecated` 后缀）的文件和目录。
这些内容均为**训练相关代码**，当前推理管线不依赖，可整体移出项目目录存档。

> 建议移至项目外的备份目录，例如 `GPT-SoVITS-v3lora-20250401_training_archive/`

---

## 文件清单（共 36 项）

### 训练主脚本（GPT_SoVITS/ 根目录）

| 当前位置 | 说明 |
|---------|------|
| `GPT_SoVITS/s1_train.py.deprecated` | S1（GPT/AR）阶段训练脚本 |
| `GPT_SoVITS/s2_train.py.deprecated` | S2（SoVITS）阶段训练脚本 |
| `GPT_SoVITS/s2_train_v3.py.deprecated` | S2 v3 训练脚本 |
| `GPT_SoVITS/s2_train_v3_lora.py.deprecated` | S2 v3 LoRA 微调训练脚本 |

### 模型导出脚本（GPT_SoVITS/ 根目录）

| 当前位置 | 说明 |
|---------|------|
| `GPT_SoVITS/onnx_export.py.deprecated` | 导出 ONNX 格式 |
| `GPT_SoVITS/export_torch_script.py.deprecated` | 导出 TorchScript v1 |
| `GPT_SoVITS/export_torch_script_v3.py.deprecated` | 导出 TorchScript v3 |

### 原版官方推理 UI（GPT_SoVITS/ 根目录）

| 当前位置 | 说明 |
|---------|------|
| `GPT_SoVITS/inference_webui.py.deprecated` | 原版 Gradio WebUI |
| `GPT_SoVITS/inference_webui_fast.py.deprecated` | 原版 Gradio 快速版 WebUI |
| `GPT_SoVITS/inference_gui.py.deprecated` | 原版 PyQt GUI（已由 chatGui.py 替代） |

### 数据集准备目录（GPT_SoVITS/ 根目录）

| 当前位置 | 说明 |
|---------|------|
| `GPT_SoVITS/prepare_datasets.deprecated/` | 训练数据预处理脚本（文本/HuBERT/语义） |

### BigVGAN 训练组件（GPT_SoVITS/BigVGAN/）

| 当前位置 | 说明 |
|---------|------|
| `GPT_SoVITS/BigVGAN/train.py.deprecated` | BigVGAN 训练主脚本 |
| `GPT_SoVITS/BigVGAN/discriminators.py.deprecated` | 训练用判别器 |
| `GPT_SoVITS/BigVGAN/loss.py.deprecated` | 训练损失函数 |
| `GPT_SoVITS/BigVGAN/meldataset.py.deprecated` | 训练数据集加载（`MAX_WAV_VALUE` 已内联到 `utils0.py`） |
| `GPT_SoVITS/BigVGAN/env.py.deprecated` | 训练环境工具（`AttrDict` 已内联到 `bigvgan.py`） |
| `GPT_SoVITS/BigVGAN/inference.py.deprecated` | BigVGAN 独立推理 demo（非 TTS 调用路径） |
| `GPT_SoVITS/BigVGAN/inference_e2e.py.deprecated` | BigVGAN 端到端推理 demo |
| `GPT_SoVITS/BigVGAN/filelists.deprecated/` | 训练数据文件列表 |
| `GPT_SoVITS/BigVGAN/demo.deprecated/` | 官方演示素材 |
| `GPT_SoVITS/BigVGAN/tests.deprecated/` | 单元测试 |
| `GPT_SoVITS/BigVGAN/incl_licenses.deprecated/` | 依赖库授权说明 |
| `GPT_SoVITS/BigVGAN/nv-modelcard++.deprecated/` | NVIDIA 模型卡片 |

### 训练配置文件（GPT_SoVITS/configs/）

| 当前位置 | 说明 |
|---------|------|
| `GPT_SoVITS/configs/s1.yaml.deprecated` | S1 标准训练配置 |
| `GPT_SoVITS/configs/s1big.yaml.deprecated` | S1 大模型训练配置 |
| `GPT_SoVITS/configs/s1big2.yaml.deprecated` | S1 大模型 v2 训练配置 |
| `GPT_SoVITS/configs/s1longer.yaml.deprecated` | S1 长文本训练配置 |
| `GPT_SoVITS/configs/s1longer-v2.yaml.deprecated` | S1 长文本 v2 训练配置 |
| `GPT_SoVITS/configs/s1mq.yaml.deprecated` | S1 多量化训练配置 |
| `GPT_SoVITS/configs/s2.json.deprecated` | S2 训练配置 |
| `GPT_SoVITS/configs/train.yaml.deprecated` | 通用训练配置 |

### module 目录训练专用文件（GPT_SoVITS/module/）

| 当前位置 | 说明 |
|---------|------|
| `GPT_SoVITS/module/data_utils.py.deprecated` | 训练数据加载工具 |
| `GPT_SoVITS/module/losses.py.deprecated` | 训练损失函数 |
| `GPT_SoVITS/module/models_onnx.py.deprecated` | ONNX 导出版模型定义 |
| `GPT_SoVITS/module/attentions_onnx.py.deprecated` | ONNX 导出版注意力机制 |

### AR 训练数据加载目录（GPT_SoVITS/AR/）

| 当前位置 | 说明 |
|---------|------|
| `GPT_SoVITS/AR/data.deprecated/` | AR 模型训练数据集类 |

---

## 注意事项

- **`GPT_SoVITS/configs/tts_infer.yaml`** 未弃用，推理时仍需使用
- **`GPT_SoVITS/BigVGAN/bigvgan.py`** 中已将 `AttrDict` 内联，不再依赖 `env.py`
- **`GPT_SoVITS/BigVGAN/utils0.py`** 中已将 `MAX_WAV_VALUE` 内联，不再依赖 `meldataset.py`
- 移出后若发现有新的 `ModuleNotFoundError`，说明还有遗漏的内联依赖，按相同方式处理即可
