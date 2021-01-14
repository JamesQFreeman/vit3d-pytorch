# vit3d-pytorch
3D Vision Transformer, in PyTorch. Modified from [lucidrains' vit-pytorch](https://github.com/lucidrains/vit-pytorch).


## Install
```
$ pip install vit-pytorch
```

## Usage
```
import torch
from vit3d_pytorch import ViT3D

v3d = ViT3D(
        image_size=(256, 256, 64),
        patch_size=32,
        num_classes=10,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
img3d = torch.randn(1, 1, 256, 256, 64)
preds = v3d(img3d)
print("ViT3D output size:", preds.shape)
```
