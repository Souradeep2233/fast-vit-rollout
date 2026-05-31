import torch
import timm
import urllib.request
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# Import the class we just built
from vit_rollout import ViTAttentionRollout

def main():
    # 1. Load a pre-trained Vision Transformer from timm
    print("Loading ViT model...")
    # vit_base_patch16_224 is a standard, widely used backbone
    import timm.layers
    timm.layers.set_fused_attn(False)
    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=True)
    
    # 2. Download and prepare a sample image
    # print("Downloading sample image...")
    # url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    # filename = "dog.jpg"
    # urllib.request.urlretrieve(url, filename)
    filename = "image.png" # Assuming you have the image locally, otherwise uncomment above lines to download

    # Load image and resize for the model
    img_size = 518
    img_pil = Image.open(filename).convert('RGB')
    original_rgb = cv2.resize(np.array(img_pil), (img_size, img_size))

    # Standard ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img_pil).unsqueeze(0) # Add batch dimension: (1, 3, 224, 224)

    # 3. Initialize your custom package module
    print("Initializing Attention Rollout...")
    rollout = ViTAttentionRollout(
        model=model, 
        head_fusion="mean", 
        discard_ratio=0.9 # Discard the lowest 90% of attention weights to remove noise
    )

    # 4. Execute the model and generate the heatmap
    print("Executing forward pass and generating heatmap...")
    # return_vis=True tells our module to return both the raw matrix and the overlay image
    rollout_matrix, heatmap_vis = rollout(
        input_tensor=input_tensor, 
        original_image=original_rgb, 
        return_vis=True,
        legend=True
    )

    # 5. Save the resulting heatmap
    output_path = "attention_heatmap.jpg"
    # Convert RGB to BGR because OpenCV expects BGR formatting for saving
    heatmap_bgr = cv2.cvtColor(heatmap_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, heatmap_bgr)

    print(f"Success! Open '{output_path}' to see what the ViT was looking at.")

if __name__ == "__main__":
    main()