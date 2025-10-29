#  A unified foundation model framework for pancancer lymph-node metastasis detection


## Pre-requisites

All experiments are run on a machine with
- 4 NVIDIA RTX A6000 GPUs
- Python (Python 3.8) and Pyotrch (torch\==2.3.1)

## Installation
1. Install [Anaconda](https://www.anaconda.com/distribution/)

2. Clone this reposity and cd into the directory:
```shell
git clone https://github.com/mimiyuyu/LymphPath.git
cd LymphPath
```

3. Create a new environment and install dependencies:
```shell
conda create -n LymphPath python=3.8 -y --no-default-packages
conda activate LymphPath
pip install --upgrade pip
pip install -r requirements.txt
```

## Image Processing Pipeline

### Extract Tiles from Whole Slide Images
Preprocess the slides following [CLAM](https://github.com/mahmoodlab/CLAM), including foreground tissue segmentation and stitching. 

### Extract Image Feature Embeddings
1. Download the pretrained
   - [GigaPath model weights](https://huggingface.co/prov-gigapath/prov-gigapath), put it to *./weights/* and load the model
   - [UNI model weights](https://huggingface.co/MahmoodLab/UNI), put it to *./weights/* and load the model
   - [Virchow2 model weights](https://huggingface.co/paige-ai/Virchow2), put it to *./weights/* and load the model
```python
import torch
from PIL import Image
from timm.data import resolve_data_config, create_transform
from timm.layers import SwiGLUPacked
import timm

model_names = {
    "gigapath": "hf-hub:prov-gigapath/prov-gigapath",
    "uni": "hf-hub:MahmoodLab/UNI",
    "virchow2": "hf-hub:paige-ai/Virchow2"
}

models = {}
transforms = {}
for name, hub_path in model_names.items():
    # Create model and set to evaluation mode
    model = timm.create_model(
        hub_path,
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU
    ).eval()
    models[name] = model

    transforms[name] = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
```

2. Use Gigapath, UNI and Virchow2 to extract image embeddings
```python
image = Image.open("/path/to/your/image.png")

embeddings = {}

for name, model in models.items():
    transform = transforms[name]
    x = transform(image).unsqueeze(0)  # shape: 1 x 3 x 224 x 224

    with torch.no_grad():
        output = model(x)  # e.g., shape: 1 x 256 x 1280 (for Virchow2)

    class_token = output[:, 0]  # shape: 1 x 1280
    patch_tokens = output[:, 5:]  # shape: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

    # Concatenate the class token with the mean-pooled patch tokens
    embedding = torch.cat(
        [class_token, patch_tokens.mean(1)], dim=-1)  # shape: 1 x 2560

    embeddings[name] = embedding
```

## Basic Usage: Lymph-Node Metastasis Detection with LymphPath

Please refer to `demo.ipynb` for a demonstration. 

1. Load the LymphPath model
```python
from utils.utils import read_yaml
from utils.model_factory import load_model
import torch

cfg = read_yaml('projects/configs/cfg_LymphPath.yaml')
model = load_model(cfg)
model.load_state_dict(torch.load('weights/LymphPath.pt'), strict=True)
```

2. Load your image
```python
import pandas as pd
import random
from datasets.ThreeStreamBagDataset import ThreeChannelBagDataset

random.seed(1)
your_csv = pd.read_csv('csv/17-BRCA.csv')

wsi_id = random.choice(list(your_csv['slide_id']))
wsi_data = your_csv[your_csv['slide_id'] == wsi_id]

data_dir1 = '/feature/gigapath/'
data_dir2 = '/feature/UNI/'
data_dir3 = '/feature/virchow2/'

dataset = ThreeChannelBagDataset(
    df=wsi_data,
    data_dir1=data_dir1,
    data_dir2=data_dir2,
    data_dir3=data_dir3,
    label_field='label'
)

sample = dataset[0]
feature1 = sample['features1']  
feature2 = sample['features2']
feature3 = sample['features3']
y = sample['label']
```

3. Lymph-Node Metastasis Detection
```python
model.eval()
model.to('cuda')
feature1 = feature1.to('cuda')
feature2 = feature2.to('cuda')
feature3 = feature3.to('cuda')
y =y .to('cuda')
res = model(feature1, feature2, feature3, label=y, instance_eval=True)

logits1 = res['logits1']
logits2 = res['logits2']
logits3 = res['logits3']
bag_logits_merge = res['merge_logits']

y_prob1 = torch.softmax(logits1, dim=-1)
y_prob2 = torch.softmax(logits2, dim=-1)
y_prob3 = torch.softmax(logits3, dim=-1)
y_prob_merge = torch.softmax(bag_logits_merge, dim=-1)
y_prob = 0.4 * (y_prob1 + y_prob2 + y_prob3) / 3 + 0.6 * y_prob_merge
print('slide id:', str(wsi_id), 'if LNM:', int(y_prob[:,1] > 0.5))
```

## Evaluation 

To reproduce the results in our paper, we provide a reproducible result on [17-BRCA](https://camelyon17.grand-challenge.org/Data/) dataset.
Please refer to `demo.ipynb` for a demonstration. 
* First download our processed 17-BRCA frozen features [here](xxx)
* Put the extracted features to *./features/* 
* Run the following command:
```shell
python3 eval.py --config_path projects/configs/cfg_LymphPath.yaml
```
The AUC and AUPRC metrics for this cohort will be stored at `./projects/result/17-BRCA/`.
```
AUC:0.9905     AUPRC:0.977659
```

## Acknowledgements
The project was built on many amazing repositories: [GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath), [UNI](https://huggingface.co/MahmoodLab/UNI), [Virchow2](https://huggingface.co/paige-ai/Virchow2), and [CLAM](https://github.com/mahmoodlab/CLAM). We thank the authors and developers for their contributions.
 
## Issue
Please open new threads or address questions to zhang_jiayu@stu.scu.edu.cn or xiyue.wang.scu@gmail.com

## License

LymphPath is made available under the CC BY-NC-SA 4.0 License and is available for non-commercial academic purposes.



