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
img_size = 224 #518 for dinov2
filename="image.png"
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