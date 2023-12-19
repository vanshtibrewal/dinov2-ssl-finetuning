This is a slightly adapted version of the DINOv2 GitHub repository [`Paper`](https://arxiv.org/abs/2304.07193), which was used to finetune DINOv2 with histopathological data.

For the finetuning process, we utilized histopathological data from two primary datasets:
- **TCGA (The Cancer Genome Atlas):** Specifically, colorectal cancer (CRC) data from the cohorts COAD and READ were used. This dataset includes annotations of microsatellite instability (MSI).
  - TCGA Dataset: [The Cancer Genome Atlas Pan-Cancer analysis project](https://doi.org/10.1038/ng.2764)

- **NCT-CRC-100K:** This dataset consists of 100,000 histological images of human colorectal cancer and healthy tissue.
  - NCT-CRC-100K Dataset: [100,000 histological images of human colorectal cancer and healthy tissue](https://doi.org/10.5281/zenodo.1214456)

For testing purposes, we incorporated two additional external datasets:
- **CPTAC (Clinical Proteomic Tumor Analysis Consortium):** For more details, visit the [CPTAC Data Portal](https://cptac-data-portal.georgetown.edu/). (Accessed: 10.11.2023)
  - CPTAC Dataset: [CPTAC Data Portal](https://cptac-data-portal.georgetown.edu/)

- **CRC-VAL-HE-7K:** This dataset, similar to NCT-CRC-100K, was employed for testing purposes.
  - CRC-VAL-HE-7K Dataset: [7180 histological images of human colorectal cancer and healthy tissue](https://doi.org/10.5281/zenodo.1214456)

We used the following testing pipeline for TCGA and CPTAC:
- **Testing Pipeline:** [HistoBistro](https://github.com/peng-lab/HistoBistro)



# DINOv2: finetuning

Benedikt Roth,
Valentin Koch,
Sophia J. Wagner,
Julia A. Schnabel,
Carsten Marr,
Tingying Peng


## Pretrained models finetuned on NCT-CRC-100K

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />params</th>
      <th># of<br />iterations</th>
      <th>CRC-VAL-HE-7K<br />20-NN balanced acc</th>
      <th>CRC-VAL-HE-7K<br />linear balanced acc</th>
      <th>teacher backbone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14</td>
      <td align="right">21 M</td>
      <td align="right">2k</td>
      <td align="right">93.8%</td>
      <td align="right">92.7%</td>
      <td><a href="">teacher weights</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="right">1,100 M</td>
      <td align="right">10k</td>
      <td align="right">93.4%</td>
      <td align="right">93.7%</td>
      <td><a href="">teacher weights</a></td>
    </tr>
  </tbody>
</table>

## Pretrained models finetuned on TCGA

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />params</th>
      <th># of<br />iterations</th>
      <th>TCGA<br />AUROC</th>
      <th>CPTAC<br />AUROC</th>
      <th>teacher backbone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14</td>
      <td align="right">21 M</td>
      <td align="right">30k</td>
      <td align="right">89%</td>
      <td align="right">85%</td>
      <td><a href="">teacher weights</a></td>
    </tr>
    <tr>
      <td>ViT-S/14</td>
      <td align="right">21 M</td>
      <td align="right">40k</td>
      <td align="right">86%</td>
      <td align="right">89%</td>
      <td><a href="">teacher weights</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="right">1,100 M</td>
      <td align="right">60k</td>
      <td align="right">84%</td>
      <td align="right">79%</td>
      <td><a href="">teacher weights</a></td>
    </tr>
  </tbody>
</table>

## Load pretrained model 


```python
import torch
import torch.nn as nn

DINO_PATH_FINETUNED_DOWNLOADED=''

def get_dino_finetuned_downloaded():
    # load the original DINOv2 model with the correct architecture and parameters. The positional embedding is too large.
    # load vits or vitg
    model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    #model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    # load finetuned weights
    pretrained = torch.load(DINO_PATH_FINETUNED_DOWNLOADED, map_location=torch.device('cpu'))
    # make correct state dict for loading
    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key:
            print('not used')
        else:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value
    #change shape of pos_embed, shape depending on vits or vitg
    pos_embed = nn.Parameter(torch.zeros(1, 257, 384))
    #pos_embed = nn.Parameter(torch.zeros(1, 257, 1536))
    model.pos_embed = pos_embed
    # load state dict
    model.load_state_dict(new_state_dict, strict=True)
    return model

model=get_dino_finetuned_downloaded()
```
## Installation

This requires the same prerequisites as the original DINOv2 implementation.

The training and evaluation code requires PyTorch 2.0 and xFormers 0.0.18 as well as a number of other 3rd party packages. Note that the code has only been tested with the specified versions and also expects a Linux environment. To setup all the required dependencies for training and evaluation, please follow the instructions below:

conda (Recommended) - Clone the repository and then create and activate a dinov2 conda environment using the provided environment definition:

```python
conda env create -f conda.yaml
conda activate dinov2
```

pip - Clone the repository and then use the provided requirements.txt to install the dependencies:

```python
pip install -r requirements.txt
```


## Continue finetuning

If you want to continue finetuning or use the DINO heads, the remaining weights can be found here:

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th>dataset</th>
      <th># of<br />iterations</th>
      <th>student backbone</th>
      <th>student DINO head</th>
      <th>teacher DINO head</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14</td>
      <td>NCT-CRC-100K</td>
      <td align="right">2k</td>
      <td><a href="">student backbone</a></td>
      <td><a href="">student DINO head</a></td>
      <td><a href="">teacher DINO head</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td>NCT-CRC-100K</td>
      <td align="right">10k</td>
      <td><a href="">student backbone</a></td>
      <td><a href="">student DINO head</a></td>
      <td><a href="">teacher DINO head</a></td>
    </tr>
    <tr>
      <td>ViT-S/14</td>
      <td>TCGA</td>
      <td align="right">30k</td>
      <td><a href="">student backbone</a></td>
      <td><a href="">student DINO head</a></td>
      <td><a href="">teacher DINO head</a></td>
    </tr>
    <tr>
      <td>ViT-S/14</td>
      <td>TCGA</td>
      <td align="right">40k</td>
      <td><a href="">student backbone</a></td>
      <td><a href="">student DINO head</a></td>
      <td><a href="">teacher DINO head</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td>TCGA</td>
      <td align="right">60k</td>
      <td><a href="">student backbone</a></td>
      <td><a href="">student DINO head</a></td>
      <td><a href="">teacher DINO head</a></td>
    </tr>
  </tbody>
</table>


## Citation

If you use the original code or the adapted DINOv2 repository finetuned with histopathological data, please cite the following:

- DINOv2 Repository:
  - [Original Paper](https://arxiv.org/abs/2304.07193)
  - [Original DINOv2 Repository](https://github.com/facebookresearch/dinov2/tree/main/dinov2)

- This Repository:
  - [Paper]()
