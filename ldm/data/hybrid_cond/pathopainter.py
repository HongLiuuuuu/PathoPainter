from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random

class TCGADataset(Dataset):
    """Dataset with tumor presence labels in text"""

    def __init__(self, config=None):
        self.root = Path(config.get("root"))

        self.image_root = self.root

        self.p_uncond = config.get("p_uncond", 0)

        self.mag = config.get("magnification")

        with open(self.root, 'r') as file:
            self.data = file.readlines()    
            self.data = [x.strip().split(', ')[0] for x in self.data if x.strip().split(', ')[1] != 'non_tumor']

    def __len__(self):
        return len(self.data)   

    def __getitem__(self, idx):

        mask_path = self.data[idx]  
        mask = np.array(Image.open(mask_path)) / 255
        
        mask = np.array(mask, dtype=np.float32)
        mask = np.expand_dims(mask, axis=2)

        img_path = mask_path.replace('mask', 'image')
        tile = np.array(Image.open(img_path))
        image = (tile / 127.5 - 1.0).astype(np.float32)  

        # Random horizontal and vertical flips
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # load SSL features
        ## for training, we use embeddings from the same image
        feat_path = mask_path.replace('mask', 'ssl_features').replace('.png', '.npy')
        feat_patch = np.load(feat_path)

        # ## for sampling, we find a random embedding from a different image within the same tumor group
        # feat_path = random.choice(self.data).replace('mask', 'ssl_features').replace('.png', '.npy')
        # feat_patch = np.load(feat_path)

        
        # replace patch level emb with all zeros
        if np.random.rand() < self.p_uncond:
            feat_patch = np.zeros_like(feat_patch)

        return {
            "image": image,
            "mask": mask,
            "feat_patch": feat_patch,
            # "image_name": img_name,
            "mask_path": mask_path,
            "human_label": ""
        }