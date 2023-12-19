This is a slightly adapted version of the DINOv2 github repository [`Paper`](https://arxiv.org/abs/2304.07193), which was used to finetune DINOv2 with histopathological data.
For the finetuning, the datasets TCGA (with colorectal cancer (CRC) from the cohorts COAD and READ with annotations of microsatellite instability (MSI)) and NCT-CRC-100K were used. 

# DINOv2: finetuning

Benedikt Roth,
Valentin Koch,
Sophia J. Wagner,
Julia A. Schnabel,
Carsten Marr,
Tingying Peng


## Pretrained models

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
      <td>ViT-B/14 distilled</td>
      <td align="right">86 M</td>
      <td align="right">82.1%</td>
      <td align="right">84.5%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth">backbone only</a></td>
    </tr>
    <tr>
      <td>ViT-L/14 distilled</td>
      <td align="right">300 M</td>
      <td align="right">83.5%</td>
      <td align="right">86.3%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth">backbone only</a></td>
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

### Pretrained backbones (via PyTorch Hub)

Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch (the only required dependency for loading the model). Installing PyTorch with CUDA support is strongly recommended.

A corresponding [model card](MODEL_CARD.md) is included in the repository.

```python
import torch

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
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
