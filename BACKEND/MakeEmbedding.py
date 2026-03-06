import numpy as np
import torch
from PIL import Image

def image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return t

@torch.no_grad()
def make_embedding(model: torch.nn.Module, img: Image.Image, device="cpu") -> np.ndarray:
    model.eval()
    t = image_to_tensor(img).to(device)
    emb = model(t).cpu().numpy()[0]
    return emb
