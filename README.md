# Fast ViT Rollout 🚀

A robust, plug-and-play Attention Rollout extractor for Vision Transformers. Works seamlessly out of the box with `timm` and Hugging Face backbones.

### Why use this?
Most existing Attention Rollout scripts break on modern architectures. This package natively handles:
* **Flash Attention (SDPA):** Bypasses PyTorch 2.0+ fused attention issues.
* **Register Tokens:** Flawlessly parses DINOv2 and DeiT models without shape mismatch errors.
* **Auto-Detection:** Automatically hooks the correct layers without manual indexing.
* **Native Heatmaps:** Generates OpenCV-based overlays with built-in intensity colorbars.

### Installation
```bash
pip install fast-vit-rollout
```

### Quick Start
```
import torch
import timm
import cv2
import urllib.request
import numpy as np
from PIL import Image
from torchvision import transforms

# 1. This proves the package installed correctly!
from fast_vit_rollout import ViTAttentionRollout

print("Package imported successfully! Testing model...")

# 2. Setup DINOv2
import timm.layers
timm.layers.set_fused_attn(False) 
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# 3. Get Image
img_size = 224 #Setup Image size according to model's input size defined.
filename="image.png" # setup the image directory
img_pil = Image.open(filename).convert('RGB')
original_rgb = cv2.resize(np.array(img_pil), (img_size, img_size))

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img_pil).unsqueeze(0)

# 4. Run the newly installed package
rollout = ViTAttentionRollout(model, discard_ratio=0.9)
matrix, heatmap = rollout(input_tensor, original_image=original_rgb, return_vis=True)

cv2.imwrite("final_test.jpg", cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
print("Test passed! Heatmap saved as final_test.jpg")
```

###Github
🚀 This is an open-source codebase.
We welcome contributions from the community!

If you encounter any bugs, have suggestions for improvements, or would like to add new features, feel free to open an issue or submit a pull request.

Contribute here: https://github.com/Souradeep2233/fast-vit-rollout
