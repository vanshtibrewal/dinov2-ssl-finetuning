# DINOv2: Learning Robust Visual Features without Supervision

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

Benedikt Roth,
Valentin Koch,
Sophia J. Wagner,
Julia A. Schnabel,
Carsten Marr,
Tingying Peng

[[`Paper`](https://arxiv.org/abs/2304.07193)] [[`Blog`](https://ai.facebook.com/blog/dino-v2-computer-vision-self-supervised-learning/)] [[`Demo`](https://dinov2.metademolab.com)] [[`BibTeX`](#citing-dinov2)]



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


## Citing DINOv2

If you find this repository useful, please consider giving a star :star: and citation :t-rex::

```
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```
