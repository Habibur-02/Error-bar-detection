import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model_utils import ErrorBarCNN, ErrorBarDataset, DEVICE 
import os

def train_model():
    print("Initializing Optimized Training...")
    
    
    dataset = ErrorBarDataset(image_dir="dataset/images", label_dir="dataset/labels")
    
    if len(dataset) == 0:
        print(" Error: Dataset empty!")
        return

    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    

    BATCH_SIZE = 256 
    

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    model = ErrorBarCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.L1Loss()
    
    print(f"Training on {DEVICE} with Batch Size {BATCH_SIZE}...")
    
    for epoch in range(15):
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
            
        print(f"Epoch {epoch+1}: Avg Loss: {train_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), "error_bar_model.pth")
    print(" Training Complete!")

if __name__ == "__main__":
    train_model()