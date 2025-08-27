## 📖 简介
本仓库基于 [MMOCR](https://github.com/open-mmlab/mmocr) 框架，复现论文：  
**SFBNet: Wavelet-Based Spatial-Frequency and Differentiable Binarization Network for Scene Text Detection**。

SFBNet 提出了一种结合 **空间特征 (Spatial Features)** 与 **频域特征 (Frequency Features)** 的场景文字检测方法，利用：
- **WFDM** (Wavelet Feature Decomposition Module) 小波分解增强细节表达  
- **MCAF** (Multiscale Cross-Domain Alignment Filter) 跨域特征对齐  
- **可微分二值化 (Differentiable Binarization)** 生成鲁棒分割结果  
在 ICDAR2015、Total-Text 和 DRRD 数据集上均取得了 SOTA 性能。  
<img width="937" height="648" alt="image" src="https://github.com/user-attachments/assets/91dee8e9-7bd9-4939-a1ac-f1772d981033" />

---

## 📂 项目结构
```
mmocr/
├── .circleci/             # 持续集成配置
├── .github/               # GitHub 配置：actions、issue 模板等
├── configs/               # 模型配置文件
│   ├── textdet/           # 文本检测配置
│   ├── textrecog/         # 文本识别配置
│   ├── kie/               # KIE (关键信息抽取) 配置
│   └── pipelines/         # 数据流水线配置
├── projects/              # 示例项目（如 ABCNet 等）:contentReference[oaicite:1]{index=1}
├── tools/                 # 训练、测试、推理等命令行脚本
├── demo/                  # 示例图片、结果展示、demo 脚本
├── docs/                  # 文档目录
│   ├── en/                # English 文档
│   │   ├── get_started/
│   │   ├── basic_concepts/
│   │   ├── user_guides/
│   │   └── api/
│   └── zh_cn/             # 中文文档（README_zh-CN.md 等）:contentReference[oaicite:2]{index=2}
├── dataset_zoo.py         # 数据集注册脚本
├── project_zoo.py         # 项目示例注册脚本
├── model-index.yml        # 模型目录与简介
├── setup.py               # 项目安装入口
├── requirements.txt       # Python 依赖表
├── setup.cfg              # 配置文件
├── LICENSE                # 授权协议
├── CITATION.cff           # 引用格式
├── README.md              # 主 README 文件
├── README_zh-CN.md        # 中文 README
├── MANIFEST.in            # 打包清单
├── stats.py               # 项目统计脚本
├── makefile / make.bat    # 构建辅助脚本
├── switch_language.md     # 文档语言切换指南
├── weight_list.py         # 权重管理脚本
└── tests/                 # 测试脚本与测试用例


````
---

## ⚙️ 环境安装
```bash
conda create -n mmocr python=3.8 -y
conda activate mmocr

# 安装 PyTorch (根据 CUDA 版本选择)
pip install torch torchvision

# 安装 mmcv & mmocr
pip install openmim
mim install "mmcv>=2.0.0"
mim install mmengine
mim install mmocr

# 安装依赖
pip install -r requirements.txt
````

---

## 🚀 使用方法

### 数据准备

使用以下数据集（需自行下载）：
* **ICDAR2015** (1000 train / 500 test)
* **Total-Text** (1255 train / 300 test)
* **DRRD** (24k 图像，辐射仪读数)

目录示例：

```
data/icdar2015
├── textdet_imgs
│   ├── test
│   └── train
├── textdet_test.json
└── textdet_train.json
```

### 训练

```bash
# 通过调用 tools/train.py 来训练指定的 MMOCR 模型
CUDA_VISIBLE_DEVICES= python tools/train.py ${CONFIG_FILE} [PY_ARGS]

# 训练
# 示例 1：使用 CPU 训练 
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py

# 示例 2：指定使用 gpu:0 训练 ，指定工作目录为 dbnet/，并打开混合精度（amp）训练
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py --work-dir dbnet/ --amp
```
### 测试

```bash
# 通过调用 tools/test.py 来测试指定的 MMOCR 模型
CUDA_VISIBLE_DEVICES= python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [PY_ARGS]

# 测试
# 示例 1：使用 CPU 测试 
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py dbnet_r50.pth
# 示例 2：使用 gpu:0 测试
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py dbnet_r50.pth
```
### 推理

```bash
python tools/infer.py \
    configs/textdet/sfbnet/sfbnet_r50_icdar2015.py \
    work_dirs/sfbnet/best.pth \
    demo/demo.jpg
```
---

## 📊 实验结果

在论文中，SFBNet 在多个数据集上的表现优异：

| 数据集        | Precision | Recall | F-measure |
| ---------- | --------- | ------ | --------- |
| ICDAR2015  | 92.4%     | 86.9%  | **89.6%** |
| Total-Text | 89.8%     | 84.2%  | **86.9%** |
| DRRD       | 93.4%     | 87.4%  | **90.3%** |


## 🤝 贡献

欢迎提出 Issue，一起改进该项目！
