import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model_utils import ErrorBarCNN, ErrorBarDataset, DEVICE 
import os

def train():
    
    print("Loading Dataset...")
    
    
    if not os.path.exists("dataset/images") or not os.path.exists("dataset/labels"):
        print("Error: Dataset folder not found! Please run the generator script first.")
        return

    dataset = ErrorBarDataset(image_dir="dataset/images", label_dir="dataset/labels")
    
    if len(dataset) == 0:
        print("Error: No data found in dataset/labels. Make sure .json files exist.")
        return

    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    
    
    model = ErrorBarCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss() 
    
    print(f"Starting training on {DEVICE}...")
    
    NUM_EPOCHS = 20 
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0: 
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx}], Loss: {loss.item():.4f}")
        
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, targets).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"==> Epoch {epoch+1} Complete. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        
        torch.save(model.state_dict(), "best_model.pth")
    
    print("Training Complete! Model saved as 'best_model.pth'")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()