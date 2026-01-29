import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import os
import json
import numpy as np
from tqdm import tqdm
import torchvision.models as models

PATCH_H, PATCH_W = 64, 48
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ErrorBarDatasetResNet(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        
        files = sorted([f for f in os.listdir(label_dir) if f.endswith('.json')])
        print(f" [ResNet] Pre-loading data into RAM from {len(files)} files...")
        
        for f in tqdm(files):
            label_path = os.path.join(label_dir, f)
            img_path = os.path.join(image_dir, f.replace(".json", ".png"))
            
            if not os.path.exists(img_path): continue

            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                # Padding
                pad_h, pad_w = PATCH_H // 2, PATCH_W // 2
                padded_img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=255)
                
                with open(label_path, 'r') as jf:
                    data = json.load(jf)
                
                series_list = data if isinstance(data, list) else data.get('error_bars', [])
                
                for series in series_list:
                    for pt in series.get('points', []):
                        cx = float(pt['x'])
                        cy = float(pt['y'])
                        top = float(pt.get('topBarPixelDistance', 0))
                        bot = float(pt.get('bottomBarPixelDistance', 0))
                        
                        cx_pad, cy_pad = cx + pad_w, cy + pad_h
                        x1 = int(cx_pad - PATCH_W // 2)
                        y1 = int(cy_pad - PATCH_H // 2)
                        
                        patch = padded_img[y1:y1+PATCH_H, x1:x1+PATCH_W]
                        
                        # Safety Resize
                        if patch.shape != (PATCH_H, PATCH_W):
                            try: patch = cv2.resize(patch, (PATCH_W, PATCH_H))
                            except: continue
                        
                        self.samples.append(patch)
                        self.targets.append([top, bot])
                        
            except Exception: continue
                
        self.samples = np.array(self.samples, dtype=np.uint8)
        self.targets = np.array(self.targets, dtype=np.float32)
        print(f" [ResNet] RAM Loaded: {len(self.samples)} patches ready!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch = self.samples[idx]
        target = self.targets[idx]
        
        
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).float() / 255.0
        
        
        if self.transform:
            patch_tensor = self.transform(patch_tensor)

        return patch_tensor, torch.tensor(target, dtype=torch.float32)

class ErrorBarResNet(nn.Module):
    def __init__(self):
        super(ErrorBarResNet, self).__init__()
        
        self.model = models.resnet18(weights=None) 
        
        
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, 2) # [top, bot]
        )
        
    def forward(self, x):
        return self.model(x)