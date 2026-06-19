# A unified foundation model framework for pancancer lymph-node metastasis detection


## Pre-requisites

All experiments are run on a machine with
- NVIDIA GPU (recommended)
- Python 3.10+ (required for Blackwell GPUs, e.g. RTX 50-series / RTX PRO 6000)
- PyTorch 2.7+ with CUDA 12.8+ (required for Blackwell `sm_120`)

> **GPU compatibility note**
>
> If you see `CUDA error: no kernel image is available for execution on the device`, your PyTorch build does not support your GPU architecture.
>
> - Older builds such as `torch==2.3.1+cu121` only support up to `sm_90`
> - Blackwell GPUs (`sm_120`) require **PyTorch >= 2.7** built with **CUDA >= 12.8**
>
> Quick check:
> ```shell
> python3 -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"
> ```
> Make sure `sm_120` appears in the arch list.

## Installation

1. Install [Anaconda](https://www.anaconda.com/distribution/)

2. Clone this repository and cd into the directory:
```shell
git clone https://github.com/mimiyuyu/LymphPath.git
cd LymphPath
```

3. Create a new environment and install dependencies:

**Option A: Blackwell / RTX 50-series / RTX PRO 6000 (recommended)**
```shell
conda create -n LymphPath python=3.10 -y
conda activate LymphPath
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**Option B: Older GPUs (Ampere / Ada, sm_80 / sm_86 / sm_90)**
```shell
conda create -n LymphPath python=3.8 -y
conda activate LymphPath
pip install --upgrade pip
pip install -r requirements.txt
```

## Repository Structure

```text
LymphPath/
├── train.py                         # training entry point
├── eval.py                          # evaluation entry point
├── checkpoints/                     # put downloaded or trained weights here
├── csv/                             # dataset split csv files
├── datasets/
│   └── ThreeStreamBagDataset.py     # 3-stream MIL dataset loader
├── models/stable_models/
│   └── LymphPath.py                 # LymphPath model
├── projects/configs/                # yaml configs for train / eval
│   └── ablation/                    # ablation study configs
├── training_methods/
│   └── lymphpath.py                 # train / validation / evaluation
└── utils/                           # config / dataloader / optimizer helpers
    └── ablation.py                  # single-stream ablation helpers
```

## Data Preparation

LymphPath expects **pre-extracted patch-level features** from three foundation models:
- GigaPath
- UNI
- Virchow2

### 1. CSV format

Each split csv should contain at least:

```text
slide_id,label
patient_000_node_0,0
patient_000_node_1,1
```

For training with multi-source cohort balancing, you may optionally add a `dataset` column.

### 2. Feature layout

For each slide, save one `.pt` file per foundation model. A typical layout is:

```text
LymphPath_feature/
└── 17-BRCA/
    ├── Gigapath/
    │   └── patient_000_node_0.pt
    ├── UNI/
    │   └── patient_000_node_0.pt
    └── Virchow2/
        └── patient_000_node_0.pt
```

Update the following fields in the yaml config:
- `Data.data_dir1`
- `Data.data_dir2`
- `Data.data_dir3`
- csv paths under `Data.dataset` (e.g. `train_set`, `val_set`, `test_set`)

## Image Processing Pipeline

### Extract Tiles from Whole Slide Images
Preprocess the slides following [CLAM](https://github.com/mahmoodlab/CLAM), including foreground tissue segmentation and stitching.

### Extract Image Feature Embeddings
1. Download the pretrained
   - [GigaPath model weights](https://huggingface.co/prov-gigapath/prov-gigapath)
   - [UNI model weights](https://huggingface.co/MahmoodLab/UNI)
   - [Virchow2 model weights](https://huggingface.co/paige-ai/Virchow2)

2. Use GigaPath, UNI and Virchow2 to extract image embeddings and save them as `.pt` files.

## Model Download

The pretrained LymphPath checkpoint can be accessed from [here](https://drive.google.com/file/d/1kGgYxj0dwkDT_EUqLGmu0v5BcXjTACNX/view).

Put it to:
```text
./checkpoints/LymphPath.pt
```

## Training

### Step 1. Prepare train / val csv files

Create your own csv files, for example:
- `./csv/train.csv`
- `./csv/val.csv`

### Step 2. Edit the training config

Open `projects/configs/cfg_LymphPath_train.yaml` and update:
- `Data.dataset.train_set.csv_path`
- `Data.dataset.val_set.csv_path`
- `Data.data_dir1/2/3`

### Step 3. Start training

cd your_path/LymphPath
```shell
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --config_path projects/configs/cfg_LymphPath_train.yaml \
  --set_seed \
  --begin 0 \
  --end 1
```

Training outputs will be saved to:
```text
./projects/result/cfg_LymphPath_train.yaml_seed326/
├── best_checkpoints/s_0_checkpoint.pt
├── branch_weights.csv
└── log/
```

The best checkpoint selected by early stopping will be stored at:
```text
./projects/result/cfg_LymphPath_train.yaml_seed326/best_checkpoints/s_0_checkpoint.pt
```

## Ablation Studies

We provide two ablation settings:

1. **Single-foundation-model ablation**: train with only GigaPath, only UNI, or only Virchow2.
2. **No-KAN ablation**: replace KAN attention with standard gated attention (`ifkan: False`).

All ablation experiments reuse the same `train.py` entry point. The only differences are in the yaml config.

### Config files

| Experiment | Config |
|---|---|
| GigaPath only | `projects/configs/ablation/cfg_ablation_gigapath_only.yaml` |
| UNI only | `projects/configs/ablation/cfg_ablation_uni_only.yaml` |
| Virchow2 only | `projects/configs/ablation/cfg_ablation_virchow2_only.yaml` |
| No KAN | `projects/configs/ablation/cfg_ablation_linear.yaml` |

### How single-stream ablation works

Set `Data.active_streams` in the config:

- GigaPath only: `[1, 0, 0]`
- UNI only: `[0, 1, 0]`
- Virchow2 only: `[0, 0, 1]`

For single-stream training, we also set:

```yaml
Model:
    loss1: 1.0
    mergeloss: 0.0
```

Inactive streams are zeroed inside the model, and only the active branch contributes to bag loss, instance loss, validation metrics, and evaluation predictions.

### Run ablation training

Before training, update csv paths and feature directories in the ablation config, same as the main training config.

**GigaPath only**
```shell
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --config_path projects/configs/ablation/cfg_ablation_gigapath_only.yaml \
  --set_seed \
  --begin 0 \
  --end 1
```

**UNI only**
```shell
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --config_path projects/configs/ablation/cfg_ablation_uni_only.yaml \
  --set_seed \
  --begin 0 \
  --end 1
```

**Virchow2 only**
```shell
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --config_path projects/configs/ablation/cfg_ablation_virchow2_only.yaml \
  --set_seed \
  --begin 0 \
  --end 1
```

**Linear**
```shell
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --config_path projects/configs/ablation/cfg_ablation_linear.yaml \
  --set_seed \
  --begin 0 \
  --end 1
```

Training outputs are saved to:

```text
./projects/result/<config_name>_seed326/
```

For example:

```text
./projects/result/cfg_ablation_gigapath_only.yaml_seed326/best_checkpoints/s_0_checkpoint.pt
./projects/result/cfg_ablation_linear.yaml_seed326/best_checkpoints/s_0_checkpoint.pt
```

### Evaluate ablation models

Use the **same ablation config** for evaluation so that `active_streams` / `ifkan` match the trained checkpoint.

For single-stream models, add a test csv to the ablation config first, for example:

```yaml
Data:
    dataset:
        test_set:
            csv_path: ./csv/17-BRCA.csv
    data_dir1: ./LymphPath_feature/17-BRCA/Gigapath/
    data_dir2: ./LymphPath_feature/17-BRCA/UNI/
    data_dir3: ./LymphPath_feature/17-BRCA/Virchow2/
```

Then run:

```shell
python3 eval.py \
  --config_path projects/configs/ablation/cfg_ablation_gigapath_only.yaml \
  --checkpoint ./projects/result/cfg_ablation_gigapath_only.yaml_seed326/best_checkpoints/s_0_checkpoint.pt
```

For the no-KAN model:

```shell
python3 eval.py \
  --config_path projects/configs/ablation/cfg_ablation_linear.yaml \
  --checkpoint ./projects/result/cfg_ablation_linear.yaml_seed326/best_checkpoints/s_0_checkpoint.pt
```

## Evaluation

To reproduce the results in our paper, we provide reproducible evaluation on [17-BRCA](https://camelyon17.grand-challenge.org/Data/) and [SLN-BRCA](https://www.cancerimagingarchive.net/collection/sln-breast).

### Step 1. Download processed frozen features

- 17-BRCA features: [Google Drive](https://drive.google.com/drive/u/0/folders/1MTUuzkNtXHfa2OK09qRW_Fr4N6-MYOGZ)
- SLN-BRCA features: [Google Drive](https://drive.google.com/drive/u/1/folders/1tIaBJNV1HXWPsAwWKNg25mdk7aifaA3s)

Put them under:
```text
./LymphPath_feature/
```

### Step 2. Put the checkpoint in place

```text
./checkpoints/LymphPath.pt
```

### Step 3. Run evaluation

```shell
python3 eval.py --config_path projects/configs/cfg_LymphPath_17-BRCA.yaml
python3 eval.py --config_path projects/configs/cfg_LymphPath_SLN-BRCA.yaml
```

If you want to evaluate your own trained checkpoint:
```shell
python3 eval.py \
  --config_path projects/configs/cfg_LymphPath_17-BRCA.yaml \
  --checkpoint ./projects/result/cfg_LymphPath_train.yaml_seed326/best_checkpoints/s_0_checkpoint.pt
```

### Step 4. Check results

Metrics and predictions will be saved to:
```text
./projects/result/17-BRCA/
./projects/result/SLN-BRCA/
```

Each folder contains:
- `preds_0.csv`
- `metrics.csv`

Reference results:
```text
17-BRCA: AUC 0.9905, AUPRC 0.9777
SLN-BRCA: AUC 0.974, AUPRC 0.9148
```

## Basic Usage: Lymph-Node Metastasis Detection with LymphPath

1. Load the LymphPath model
```python
from utils.utils import read_yaml
from utils.model_factory import load_model
import torch

cfg = read_yaml('projects/configs/cfg_LymphPath_17-BRCA.yaml')
model = load_model(cfg)
model.load_state_dict(torch.load('checkpoints/LymphPath.pt'), strict=True)
```

2. Lymph-Node Metastasis Detection
```python
import pandas as pd
import random
import torch.nn.functional as F
from datasets.ThreeStreamBagDataset import ThreeChannelBagDataset

random.seed(1)
your_csv = pd.read_csv('csv/17-BRCA.csv')

wsi_id = random.choice(list(your_csv['slide_id']))
wsi_data = your_csv[your_csv['slide_id'] == wsi_id]

data_dir1 = "./LymphPath_feature/17-BRCA/Gigapath/"
data_dir2 = "./LymphPath_feature/17-BRCA/UNI/"
data_dir3 = "./LymphPath_feature/17-BRCA/Virchow2/"

dataset = ThreeChannelBagDataset(
    df=wsi_data,
    data_dir1=data_dir1,
    data_dir2=data_dir2,
    data_dir3=data_dir3,
    label_field='label'
)

sample = dataset[0]
model.eval()
model.to('cuda')
feature1 = sample['features1'].to('cuda')
feature2 = sample['features2'].to('cuda')
feature3 = sample['features3'].to('cuda')
y = sample['label'].to('cuda')

res = model(feature1, feature2, feature3, label=y, instance_eval=True)
y_prob = torch.softmax(res['merge_logits'], dim=1)
print('slide id:', str(wsi_id), 'if LNM:', int(y_prob[:, 1] > 0.5))
```

## Acknowledgements

The project was built on many amazing repositories: [GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath), [UNI](https://huggingface.co/MahmoodLab/UNI), [Virchow2](https://huggingface.co/paige-ai/Virchow2), and [CLAM](https://github.com/mahmoodlab/CLAM). We thank the authors and developers for their contributions.

## License

LymphPath is made available under the CC BY-NC-SA 4.0 License and is available for non-commercial academic purposes.
