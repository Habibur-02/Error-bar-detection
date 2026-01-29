import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from model_utils_resnet import ErrorBarResNet, ErrorBarDatasetResNet, DEVICE 
import os

def train_resnet():
    print("Initializing ResNet-18 Training Pipeline...")
    
    # --- Augmentation 
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])


    full_dataset = ErrorBarDatasetResNet(
        image_dir="dataset/images", 
        label_dir="dataset/labels", 
        transform=train_transform
    )
    
    if len(full_dataset) == 0:
        print(" Error: Dataset empty!")
        return

    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    BATCH_SIZE = 128
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    # --- Model Setup ---
    model = ErrorBarResNet().to(DEVICE)
    
    # Optimizer: AdamW (Best for ResNet)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Scheduler: Reduce LR if validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Loss: SmoothL1Loss (Less sensitive to outliers than MSE)
    criterion = torch.nn.SmoothL1Loss()
    
    print(f"Training ResNet-18 on {DEVICE} for 30 Epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(30):
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
        
        avg_val_loss = val_loss/len(val_loader)
        
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1:02d}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            torch.save(model.state_dict(), "error_bar_resnet_best.pth")
            print(f"    🌟 Best ResNet model saved!")
            
    print(" Training Complete! Model: 'error_bar_resnet_best.pth'")

if __name__ == "__main__":
    train_resnet()