This is a slightly adapted version of the DINOv2 GitHub repository [`Paper`](https://arxiv.org/abs/2304.07193), which was used to finetune DINOv2 with histopathological data.

For the finetuning process, we utilized histopathological data from two datasets:
- **TCGA (The Cancer Genome Atlas):** Specifically, colorectal cancer (CRC) data from the cohorts COAD and READ were used. This dataset includes annotations of microsatellite instability (MSI).
  - Original TCGA Dataset: [The Cancer Genome Atlas Pan-Cancer analysis project](https://doi.org/10.1038/ng.2764)

- **NCT-CRC-100K:** This dataset consists of 100,000 histological images of human colorectal cancer and healthy tissue.
  - Original NCT-CRC-100K Dataset: [100,000 histological images of human colorectal cancer and healthy tissue](https://doi.org/10.5281/zenodo.1214456)


# DINOv2: finetuning

Benedikt Roth,
Valentin Koch,
Sophia J. Wagner,
Julia A. Schnabel,
Carsten Marr,
Tingying Peng


## Pretrained models with histopathology Data

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />params</th>
      <th>ImageNet<br />k-NN</th>
      <th>ImageNet<br />linear</th>
      <th>download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14 distilled</td>
      <td align="right">21 M</td>
      <td align="right">79.0%</td>
      <td align="right">81.1%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth">backbone only</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="right">1,100 M</td>
      <td align="right">83.5%</td>
      <td align="right">86.5%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth">backbone only</a></td>
    </tr>
  </tbody>
</table>

### Load pretrained model 


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


## Citation

If you use this code or the adapted DINOv2 repository with histopathological data and finetuned with specific datasets, please cite the following:

- DINOv2 Repository:
  - [Original Paper](https://arxiv.org/abs/2304.07193)
  - [Original DINOv2 Repository](https://github.com/facebookresearch/dinov2/tree/main/dinov2)

- TCGA Dataset:
  - The Cancer Genome Atlas Research Network. (2013). The Cancer Genome Atlas Pan-Cancer analysis project. *Nature Genetics*, 45(10), 1113–1120. [DOI: 10.1038/ng.2764](https://doi.org/10.1038/ng.2764)

- NCT-CRC-100K Dataset:
  - Kather, J. N., Halama, N., & Marx, A. (2018). 100,000 histological images of human colorectal cancer and healthy tissue. Zenodo. [DOI: 10.5281/zenodo.1214456](https://doi.org/10.5281/zenodo.1214456)
