import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model_utils_resnet import ErrorBarResNet, ErrorBarDatasetResNet, DEVICE 
import os

def fine_tune_resnet():
    print("🚀 Initializing Fine-Tuning on REAL Company Data...")
    
    
    REAL_IMG_DIR = "company_dataset/images"
    REAL_LBL_DIR = "company_dataset/labels"
    PRETRAINED_MODEL = "error_bar_resnet_best.pth" 
    

    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=2, translate=(0.02, 0.02)), 
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),   
    ])

    # --- Dataset Loading ---
    print(f"📂 Loading Real Data from {REAL_LBL_DIR}...")
    real_dataset = ErrorBarDatasetResNet(
        image_dir=REAL_IMG_DIR, 
        label_dir=REAL_LBL_DIR, 
        transform=train_transform
    )
    
    if len(real_dataset) == 0:
        print("❌ Error: Company dataset not found!")
        return

    
    train_size = int(0.90 * len(real_dataset))
    val_size = len(real_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(real_dataset, [train_size, val_size])
    
    
    BATCH_SIZE = 16 
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Model Setup ---
    model = ErrorBarResNet().to(DEVICE)
    
    
    if os.path.exists(PRETRAINED_MODEL):
        print(f"✅ Loading Pre-trained weights from {PRETRAINED_MODEL}")
        try: model.load_state_dict(torch.load(PRETRAINED_MODEL, weights_only=True))
        except: model.load_state_dict(torch.load(PRETRAINED_MODEL))
    else:
        print("⚠️ Warning: Pre-trained model not found. Starting from scratch (Not Recommended).")

    
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-3)
    
    criterion = torch.nn.SmoothL1Loss()
    
    print(f" Fine-Tuning started for 40 Epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(40):
        model.train()
        train_loss = 0
        
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss/len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, targets).item()
        
        # Zero Division check
        if len(val_loader) > 0:
            avg_val_loss = val_loss/len(val_loader)
        else:
            avg_val_loss = 0
            
        print(f"Epoch {epoch+1:02d}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        
        if avg_val_loss < best_val_loss and len(val_loader) > 0:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "final_interview_model.pth")
            print(f" Best Fine-Tuned model saved!")
            
    print(" Fine-Tuning Complete! Model: 'final_interview_model.pth'")

if __name__ == "__main__":
    fine_tune_resnet()