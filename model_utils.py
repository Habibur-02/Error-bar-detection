import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import os
import json
import numpy as np
from tqdm import tqdm 

PATCH_H, PATCH_W = 64, 48
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ErrorBarDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.samples = []
        self.targets = []
        
        files = sorted([f for f in os.listdir(label_dir) if f.endswith('.json')])
        print(f" Pre-loading data into RAM from {len(files)} files... Please wait.")
        
        
        for f in tqdm(files):
            label_path = os.path.join(label_dir, f)
            img_path = os.path.join(image_dir, f.replace(".json", ".png"))
            
            if not os.path.exists(img_path): continue

            try:
                
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                
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
                        
                        # --- PATCH CUTTING HERE ---
                        cx_pad, cy_pad = cx + pad_w, cy + pad_h
                        x1 = int(cx_pad - PATCH_W // 2)
                        y1 = int(cy_pad - PATCH_H // 2)
                        
                        patch = padded_img[y1:y1+PATCH_H, x1:x1+PATCH_W]
                        
                        
                        if patch.shape != (PATCH_H, PATCH_W):
                            try:
                                patch = cv2.resize(patch, (PATCH_W, PATCH_H))
                            except: continue
                        
                        
                        patch_norm = patch.astype(np.float32) / 255.0
                        self.samples.append(patch_norm)
                        self.targets.append([top, bot])
                        
            except Exception:
                continue
                
        
        self.samples = np.array(self.samples)
        self.targets = np.array(self.targets)
        
        print(f" RAM Loaded: {len(self.samples)} patches ready for GPU!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        patch = torch.from_numpy(self.samples[idx]).unsqueeze(0) # (1, H, W)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return patch, target

class ErrorBarCNN(nn.Module):
    def __init__(self):
        super(ErrorBarCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (PATCH_H//8) * (PATCH_W//8), 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        return self.regressor(self.features(x))