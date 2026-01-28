import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.utils.data import Dataset
import json
import os


PATCH_H, PATCH_W = 256, 64  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ErrorBarCNN(nn.Module):
    def __init__(self):
        super(ErrorBarCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(256, 2) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

class ErrorBarDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.samples = []
        
        files = os.listdir(label_dir)
        for f in files:
            if f.endswith('_output.json'): 
                path = os.path.join(label_dir, f)
                try:
                    with open(path, 'r') as jf:
                        data = json.load(jf)
                        img_name = data['image_file']
                        
                        
                        if not os.path.exists(os.path.join(image_dir, img_name)):
                            continue

                        for line in data['error_bars']:
                            for pt in line['points']:
                                cx, cy = pt['data_point']['x'], pt['data_point']['y']
                                uy = pt['upper_error_bar']['y']
                                ly = pt['lower_error_bar']['y']
                                
                                target_up = cy - uy  
                                target_down = ly - cy 
                                
                                self.samples.append({
                                    'img_path': os.path.join(image_dir, img_name),
                                    'center': (cx, cy),
                                    'target': (target_up, target_down)
                                })
                except Exception as e:
                    print(f"Skipping broken file {f}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        
        img = cv2.imread(item['img_path'], cv2.IMREAD_GRAYSCALE)
        
        
        if img is None:
            return torch.zeros((1, PATCH_H, PATCH_W)), torch.tensor([0.0, 0.0])

        cx, cy = item['center']
        h, w = img.shape
        

        y1 = int(cy - PATCH_H // 2)
        y2 = y1 + PATCH_H  
        
        x1 = int(cx - PATCH_W // 2)
        x2 = x1 + PATCH_W  
        
        
        img_y1, img_y2 = max(0, y1), min(h, y2)
        img_x1, img_x2 = max(0, x1), min(w, x2)
        
        
        if img_y1 >= img_y2 or img_x1 >= img_x2:
            patch = np.ones((PATCH_H, PATCH_W), dtype=np.uint8) * 255
        else:
            
            crop_part = img[img_y1:img_y2, img_x1:img_x2]
            
            
            patch = np.ones((PATCH_H, PATCH_W), dtype=np.uint8) * 255
            
            
            out_y1 = max(0, -y1)
            out_x1 = max(0, -x1)
            
            part_h, part_w = crop_part.shape
            patch[out_y1 : out_y1 + part_h, out_x1 : out_x1 + part_w] = crop_part
        
       
        patch = patch.astype(np.float32) / 255.0
        patch_tensor = torch.from_numpy(patch).unsqueeze(0) # (1, H, W)
        
        target = torch.tensor(item['target'], dtype=torch.float32)
        
        return patch_tensor, target  