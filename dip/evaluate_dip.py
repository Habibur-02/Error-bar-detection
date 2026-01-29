import cv2
import os
import json
import numpy as np
from tqdm import tqdm


IMG_DIR = "../company_dataset/images"
LBL_DIR = "../company_dataset/labels"

def evaluate_canny_performance():
    print(f"📊 Benchmarking Classical Canny Edge Detection...")
    
    json_files = [f for f in os.listdir(LBL_DIR) if f.endswith('.json')]
    
    total_points = 0
    total_mae = 0.0
    correct_5px = 0
    correct_10px = 0
    
    
    SCAN_BUFFER = 5
    MAX_SCAN = 150
    
    for f in tqdm(json_files):
        
        json_path = os.path.join(LBL_DIR, f)
        img_name = f.replace(".json", ".png")
        img_path = os.path.join(IMG_DIR, img_name)
        
        
        if not os.path.exists(img_path):
            img_path = os.path.join(IMG_DIR, img_name.replace('.png', '.jpg'))
        
        if not os.path.exists(img_path): continue
            
        
        img = cv2.imread(img_path)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- SIZE CALCULATION FOR SAFETY ---
        height, width = gray.shape
        
        # Canny Apply
        edges = cv2.Canny(gray, 50, 150)
        
        with open(json_path, 'r') as jf:
            data = json.load(jf)
            
        series_list = data if isinstance(data, list) else data.get('data_points', []) or data.get('error_bars', [])
        
        for series in series_list:
            for pt in series.get('points', []):
                # Ground Truth
                cx = int(float(pt['x']))
                cy = int(float(pt['y']))
                actual_top = float(pt.get('topBarPixelDistance', 0))
                actual_bot = float(pt.get('bottomBarPixelDistance', 0))
                
                
                if not (0 <= cx < width and 0 <= cy < height):
                    continue

                # --- PIXEL SCANNING PREDICTION ---
                
                # Predict Top Distance
                pred_top = 0
                for y in range(cy - SCAN_BUFFER, cy - MAX_SCAN, -1):
                    # --- SAFETY CHECK 2 ---
                    if y < 0: break
                    
                    if edges[y, cx] > 0: 
                        pred_top = abs(cy - y)
                        break
                
                # Predict Bottom Distance
                pred_bot = 0
                for y in range(cy + SCAN_BUFFER, cy + MAX_SCAN):
                    # --- SAFETY CHECK 3 (Crash Fix) ---
                    if y >= height: break
                    
                    if edges[y, cx] > 0:
                        pred_bot = abs(y - cy)
                        break
                
                
                if pred_top == 0: pred_top = 100 
                if pred_bot == 0: pred_bot = 100

                # --- ERROR CALCULATION ---
                err_top = abs(actual_top - pred_top)
                err_bot = abs(actual_bot - pred_bot)
                
                total_mae += (err_top + err_bot)
                total_points += 2
                
                if err_top <= 5: correct_5px += 1
                if err_bot <= 5: correct_5px += 1
                
                if err_top <= 10: correct_10px += 1
                if err_bot <= 10: correct_10px += 1

    # --- FINAL REPORT ---
    if total_points == 0: return

    avg_mae = total_mae / total_points
    acc_5 = (correct_5px / total_points) * 100
    acc_10 = (correct_10px / total_points) * 100
    
    print("\n" + "="*45)
    print(f" CLASSICAL METHOD (CANNY) PERFORMANCE")
    print("="*45)
    print(f"🔹 Mean Absolute Error (MAE) : {avg_mae:.4f} pixels")
    print("-" * 45)
    print(f" Accuracy (Error <= 5px)   : {acc_5:.2f}%")
    print(f" Accuracy (Error <= 10px)  : {acc_10:.2f}%")
    print("="*45)
    print("Reason for failure: Noise, Grid Lines, and Text overlap.")

if __name__ == "__main__":
    evaluate_canny_performance()