import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
from PIL import Image
from torchvision import transforms
from typing import Union, Optional, Tuple, List

class ViTAttentionRollout:
    """
    A unified Attention Rollout extractor for Vision Transformers.
    Works seamlessly with both timm and Hugging Face architectures.
    """
    def __init__(
        self, 
        model: nn.Module, 
        head_fusion: str = "mean", 
        discard_ratio: float = 0.9,
        custom_target_layer: Optional[str] = None
    ):
        """
        Args:
            model: The PyTorch Vision Transformer model.
            head_fusion: Strategy to fuse multi-head attention ('mean', 'max', 'min').
            discard_ratio: Float [0, 1). Percentage of lowest attention weights to zero out (denoising).
            custom_target_layer: Explicit layer string to hook into (bypasses auto-detection if provided).
        """
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.custom_target_layer = custom_target_layer
        self.attentions: List[torch.Tensor] = []
        
        self.model.eval()
        self._register_hooks()

    def _get_hook(self):
        def hook(module, input, output):
            # Input to the dropout layer is the clean, softmaxed attention matrix
            # Shape expected: (batch_size, num_heads, seq_len, seq_len)
            self.attentions.append(input[0].detach().cpu())
        return hook

    def _register_hooks(self):
        hooked = False
        for name, module in self.model.named_modules():
            # If user provided a specific layer name, only hook that
            if self.custom_target_layer:
                if self.custom_target_layer in name:
                    module.register_forward_hook(self._get_hook())
                    hooked = True
            # Otherwise, auto-detect standard timm and Hugging Face attention drop layers
            elif name.endswith('attn_drop') or name.endswith('attention.self.dropout'):
                module.register_forward_hook(self._get_hook())
                hooked = True
                
        if not hooked:
            raise ValueError(
                f"Could not hook attention layers. Target layer '{self.custom_target_layer}' not found, "
                "or auto-detection failed. Please pass a valid custom_target_layer."
            )

    def _compute_rollout(self) -> torch.Tensor:
        if not self.attentions:
            raise RuntimeError("No attention maps were captured. Ensure the model executed a forward pass.")

        batch_size, _, num_tokens, _ = self.attentions[0].shape
        result = torch.eye(num_tokens).unsqueeze(0).repeat(batch_size, 1, 1)
        
        for attn in self.attentions:
            # 1. Fuse the multi-head attention
            if self.head_fusion == "mean":
                attn_fused = attn.mean(dim=1)
            elif self.head_fusion == "max":
                attn_fused = attn.max(dim=1)[0]
            elif self.head_fusion == "min":
                attn_fused = attn.min(dim=1)[0]
            else:
                raise ValueError("head_fusion must be 'mean', 'max', or 'min'")
                
            # 2. Add residual connection
            attn_fused = attn_fused + torch.eye(num_tokens).unsqueeze(0)
            
            # 3. Normalize
            attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)
            
            # 4. Apply discard ratio (denoising)
            if self.discard_ratio > 0:
                flat_attn = attn_fused.view(batch_size, -1)
                for b in range(batch_size):
                    threshold = torch.quantile(flat_attn[b], self.discard_ratio)
                    attn_fused[b][attn_fused[b] < threshold] = 0.0
                # Renormalize after thresholding
                attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)

            # 5. Rollout Matrix Multiplication
            result = torch.matmul(attn_fused, result)
            
        return result

    def _add_legend(self, image: np.ndarray, colormap: int) -> np.ndarray:
        """Appends a vertical colorbar legend to the right side of the image."""
        h, w, _ = image.shape
        
        # 1. Create a vertical gradient (255 at top, 0 at bottom)
        gradient = np.linspace(255, 0, h, dtype=np.uint8)
        gradient = np.tile(gradient, (30, 1)).T  # Make it 30 pixels wide
        
        # 2. Apply the same colormap to the gradient
        colorbar = cv2.applyColorMap(gradient, colormap)
        # Draw a thin black border around the colorbar
        cv2.rectangle(colorbar, (0, 0), (29, h - 1), (0, 0, 0), 1)
        
        # 3. Create a white canvas for the text labels
        text_bg = np.ones((h, 60, 3), dtype=np.uint8) * 255
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        color = (0, 0, 0)  # Black text
        
        # Add High/Low intensity labels
        cv2.putText(text_bg, "1.0", (5, 20), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(text_bg, "High", (5, 40), font, font_scale, color, thickness, cv2.LINE_AA)
        
        cv2.putText(text_bg, "0.5", (5, h // 2 + 5), font, font_scale, color, thickness, cv2.LINE_AA)
        
        cv2.putText(text_bg, "0.0", (5, h - 25), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(text_bg, "Low", (5, h - 5), font, font_scale, color, thickness, cv2.LINE_AA)
        
        # 4. Stitch them all together horizontally: [Heatmap] + [Colorbar] + [Text]
        return np.hstack((image, colorbar, text_bg))

    def _generate_heatmap(
        self, rollout_matrix: torch.Tensor,
        original_image: Union[np.ndarray, torch.Tensor],
        colormap: int = cv2.COLORMAP_JET,
        show_legend: bool = False
    ):
        """Projects the rollout matrix onto the original image as a heatmap."""
        # 1. Dynamically calculate the spatial grid size (ignoring CLS and Registers)
        total_tokens = rollout_matrix.size(-1)
        # int(np.sqrt()) naturally truncates the small addition of register tokens
        grid_size = int(np.sqrt(total_tokens - 1))
        spatial_tokens_count = grid_size * grid_size
        
        # 2. Extract ONLY the spatial patches (always the last tokens in the sequence)
        spatial_attention = rollout_matrix[0, 0, -spatial_tokens_count:]
        spatial_attention = spatial_attention.reshape(grid_size, grid_size).numpy()
        
        # Min-Max Normalization
        spatial_attention = (spatial_attention - spatial_attention.min()) / \
                        (spatial_attention.max() - spatial_attention.min() + 1e-8)
        
        # Format original image
        if isinstance(original_image, torch.Tensor):
            original_image = original_image[0].permute(1, 2, 0).cpu().numpy()
            original_image = np.clip(original_image, 0, 1)
        else:
            original_image = original_image.astype(np.float32) / 255.0
            
        h, w, _ = original_image.shape
        
        # Resize attention and apply colormap
        spatial_attention = cv2.resize(spatial_attention, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * spatial_attention), colormap)
        heatmap = np.float32(heatmap) / 255
        
        # Blend
        cam = heatmap + original_image
        cam = cam / np.max(cam)
        final_image = np.uint8(255 * cam)
        
        # Append the legend if requested
        if show_legend:
            final_image = self._add_legend(final_image, colormap)
            
        return final_image

    def __call__(
        self, 
        input_tensor: torch.Tensor, 
        original_image: Optional[Union[np.ndarray, torch.Tensor]] = None,
        return_vis: bool = False,
        legend: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray]]:
        """
        Executes the forward pass and computes the rollout.
        
        Args:
            input_tensor: Normalized tensor ready for the model (B, C, H, W).
            original_image: Unnormalized image for visualization overlay (required if return_vis=True).
            return_vis: If True, returns a tuple of (Rollout Matrix, Heatmap Numpy Array).
        """
        self.attentions = []
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        rollout_matrix = self._compute_rollout()
        
        if return_vis:
            if original_image is None:
                raise ValueError("original_image must be provided if return_vis is True.")
            heatmap = self._generate_heatmap(rollout_matrix, original_image, show_legend=legend)
            return rollout_matrix, heatmap
            
        return rollout_matrix


# ==========================================
# CLI / Standalone Execution Setup
# ==========================================

def load_and_preprocess(image_path: str, img_size: int = 224) -> Tuple[torch.Tensor, np.ndarray]:
    """Loads an image, returns the normalized tensor and the original RGB numpy array."""
    img_pil = Image.open(image_path).convert('RGB')
    img_np = cv2.resize(np.array(img_pil), (img_size, img_size))
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_pil).unsqueeze(0)
    return img_tensor, img_np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT Attention Rollout Extractor")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224", help="timm model name")
    parser.add_argument("--fusion", type=str, default="mean", choices=["mean", "max", "min"], help="Head fusion method")
    parser.add_argument("--discard", type=float, default=0.9, help="Discard ratio for denoising (0.0 to 1.0)")
    parser.add_argument("--out", type=str, default="rollout_output.jpg", help="Output path for the heatmap")
    args = parser.parse_args()

    try:
        import timm
    except ImportError:
        print("Please install timm to run the CLI: pip install timm")
        exit(1)

    print(f"Loading model '{args.model}'...")
    model = timm.create_model(args.model, pretrained=True)
    
    print(f"Preparing image '{args.image}'...")
    input_tensor, original_rgb = load_and_preprocess(args.image)
    
    print(f"Initializing Rollout (Fusion: {args.fusion}, Discard: {args.discard})...")
    # Initialize the parameterized module
    rollout = ViTAttentionRollout(
        model=model, 
        head_fusion=args.fusion, 
        discard_ratio=args.discard
    )
    
    print("Computing attention rollout...")
    # Execute with the visualization parameter turned on
    _, heatmap_vis = rollout(
        input_tensor=input_tensor, 
        original_image=original_rgb, 
        return_vis=True
    )
    
    # BGR conversion for OpenCV saving
    heatmap_bgr = cv2.cvtColor(heatmap_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.out, heatmap_bgr)
    print(f"Success! Heatmap saved to {args.out}")