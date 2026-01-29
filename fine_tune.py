import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import json
import os
import numpy as np
from tqdm import tqdm
from model_utils import ErrorBarCNN, PATCH_H, PATCH_W, DEVICE

REAL_IMAGE_DIR = "company_dataset/images"
REAL_LABEL_DIR = "company_dataset/labels"
PRETRAINED_MODEL = "best_model.pth"
OUTPUT_MODEL = "final_model.pth"

class RealDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.samples = []
        
        files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
        
        for f in files:
            label_path = os.path.join(label_dir, f)
            
            base_name = os.path.splitext(f)[0]
            img_path = os.path.join(image_dir, base_name + ".png")
            if not os.path.exists(img_path):
                img_path = os.path.join(image_dir, base_name + ".jpg")
            
            if not os.path.exists(img_path):
                continue
                
            with open(label_path, 'r') as jf:
                data = json.load(jf)
                
            raw_lines = data if isinstance(data, list) else data.get('error_bars', [])
            
            for line in raw_lines:
                for pt in line.get('points', []):
                    try:
                        # Center Point
                        cx = float(pt['x'])
                        cy = float(pt['y'])
                        

                        top_dist = float(pt.get('topBarPixelDistance', 0))
                        bot_dist = float(pt.get('bottomBarPixelDistance', 0))
                        
                        self.samples.append({
                            'img_path': img_path,
                            'center': (cx, cy),
                            'target': (top_dist, bot_dist) 
                        })
                    except Exception:
                        continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        img = cv2.imread(item['img_path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
             return torch.zeros((1, PATCH_H, PATCH_W)), torch.tensor([0.0, 0.0])

        cx, cy = item['center']
        h, w = img.shape
        
        # Robust Cropping (Canvas Method - Same as before)
        y1 = int(cy - PATCH_H // 2)
        y2 = y1 + PATCH_H
        x1 = int(cx - PATCH_W // 2)
        x2 = x1 + PATCH_W
        
        img_y1, img_y2 = max(0, y1), min(h, y2)
        img_x1, img_x2 = max(0, x1), min(w, x2)
        
        patch = np.ones((PATCH_H, PATCH_W), dtype=np.uint8) * 255
        
        if img_y1 < img_y2 and img_x1 < img_x2:
            crop_part = img[img_y1:img_y2, img_x1:img_x2]
            out_y1 = max(0, -y1)
            out_x1 = max(0, -x1)
            part_h, part_w = crop_part.shape
            patch[out_y1 : out_y1 + part_h, out_x1 : out_x1 + part_w] = crop_part
        
        patch = patch.astype(np.float32) / 255.0
        patch_tensor = torch.from_numpy(patch).unsqueeze(0)
        target = torch.tensor(item['target'], dtype=torch.float32)
        
        return patch_tensor, target

def fine_tune():
    print("Loading Real Dataset for Fine-Tuning...")
    dataset = RealDataset(REAL_IMAGE_DIR, REAL_LABEL_DIR)
    print(f"Found {len(dataset)} real samples from company data.")
    
    if len(dataset) == 0:
        print("Error: No data found! Check 'company_dataset' folder structure.")
        return

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    
    model = ErrorBarCNN().to(DEVICE)
    if os.path.exists(PRETRAINED_MODEL):
        print(f"Loading pretrained weights from {PRETRAINED_MODEL}")
        try:
            model.load_state_dict(torch.load(PRETRAINED_MODEL, weights_only=False))
        except:
            model.load_state_dict(torch.load(PRETRAINED_MODEL))
    else:
        print("Warning: Pretrained model not found! Training from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    criterion = torch.nn.MSELoss()
    
    EPOCHS = 40  
    
    print("Starting Fine-Tuning on Real Data...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"Fine-Tuning Complete! New model saved as '{OUTPUT_MODEL}' 🚀")

if __name__ == "__main__":
    fine_tune()










