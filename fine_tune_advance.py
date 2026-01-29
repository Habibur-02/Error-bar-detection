import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import json
import os
import numpy as np
import random
from model_utils import ErrorBarCNN, PATCH_H, PATCH_W, DEVICE

REAL_IMAGE_DIR = "company_dataset/images"
REAL_LABEL_DIR = "company_dataset/labels"

PREVIOUS_MODEL = "final_model.pth"       # Phase 2 এর মডেল
OUTPUT_MODEL = "final_model_advanced.pth" # Phase 3 এর মডেল (Final)

def apply_augmentation(img, center):
    """
    ইমেজকে র‍্যান্ডমলি রোটেট এবং নয়েজ দেওয়া হবে।
    এটি মডেলকে 'মুখস্থ' বিদ্যা থেকে বের করে 'বুঝে' শিখতে সাহায্য করবে।
    """
    h, w = img.shape
    cx, cy = center
    

    angle = random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    
    if random.random() > 0.5:
        alpha = random.uniform(0.8, 1.2) # Contrast
        beta = random.uniform(-20, 20)   # Brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
    return img

class AdvancedRealDataset(Dataset):
    def __init__(self, image_dir, label_dir, augment=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.augment = augment
        self.samples = []
        
        files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
        
        for f in files:
            label_path = os.path.join(label_dir, f)
            base_name = os.path.splitext(f)[0]
            
            img_path = os.path.join(image_dir, base_name + ".png")
            if not os.path.exists(img_path):
                img_path = os.path.join(image_dir, base_name + ".jpg")
            
            if not os.path.exists(img_path): continue
                
            with open(label_path, 'r') as jf:
                data = json.load(jf)
                
            raw_lines = data if isinstance(data, list) else data.get('error_bars', [])
            
            for line in raw_lines:
                for pt in line.get('points', []):
                    try:
                        cx = float(pt['x'])
                        cy = float(pt['y'])
                        top_dist = float(pt.get('topBarPixelDistance', 0))
                        bot_dist = float(pt.get('bottomBarPixelDistance', 0))
                        
                        self.samples.append({
                            'img_path': img_path,
                            'center': (cx, cy),
                            'target': (top_dist, bot_dist)
                        })
                    except: continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        img = cv2.imread(item['img_path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
             return torch.zeros((1, PATCH_H, PATCH_W)), torch.tensor([0.0, 0.0])

        cx, cy = item['center']
        
        if self.augment:
            img = apply_augmentation(img, (cx, cy))

        h, w = img.shape
        
        # Robust Cropping
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

def advanced_fine_tune():
    print("Loading Real Dataset with Augmentation...")
    dataset = AdvancedRealDataset(REAL_IMAGE_DIR, REAL_LABEL_DIR, augment=True)
    
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    
    model = ErrorBarCNN().to(DEVICE)
    
    if os.path.exists(PREVIOUS_MODEL):
        print(f"Loading Phase 2 model: {PREVIOUS_MODEL} for advanced training...")
        try:
            model.load_state_dict(torch.load(PREVIOUS_MODEL, weights_only=False))
        except:
            model.load_state_dict(torch.load(PREVIOUS_MODEL))
    else:
        print("Warning: Previous model not found! Starting from scratch (Not Recommended).")

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    criterion = torch.nn.MSELoss()
    EPOCHS = 50 
    
    print(f"Starting Phase 3 Training (Augmentation Enabled)... Target: {OUTPUT_MODEL}")
    
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
            
        avg_loss = train_loss / len(train_loader)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
        
        scheduler.step(avg_loss)
        
    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"Phase 3 Complete! Final Advanced Model saved as '{OUTPUT_MODEL}' 🚀")

if __name__ == "__main__":
    advanced_fine_tune()
    