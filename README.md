# pytorch_clip_interrogator: Image-To-Promt.
[![Downloads](https://pepy.tech/badge/pytorch_clip_interrogator)](https://pepy.tech/project/pytorch_clip_interrogator)
[![Downloads](https://pepy.tech/badge/pytorch_clip_interrogator/month)](https://pepy.tech/project/pytorch_clip_interrogator)
[![Downloads](https://pepy.tech/badge/pytorch_clip_interrogator/week)](https://pepy.tech/project/pytorch_clip_interrogator)


## Install package

```bash
pip install pytorch_clip_interrogator
```

## Install the latest version

```bash
pip install --upgrade git+https://github.com/bes-dev/pytorch_clip_interrogator.git
```

## Features
- Fully compatible with models from Huggingface.
- Supports BLIP 1/2 model.
- Support batch processing.

## Usage

### Simple code

```python
import torch
import requests
from PIL import Image
from pytorch_clip_interrogator import PromptEngineer

# build pipeline
pipe = PromptEngineer(
    blip_model="Salesforce/blip2-opt-2.7b",
    clip_model="openai/clip-vit-base-patch32",
    device="cuda",
    torch_dtype=torch.float16
)

# load image
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')


# generate caption
print(pipe(image))
```
