import cv2
import os
import json
import numpy as np
from tqdm import tqdm


IMG_DIR = "company_dataset/images"           
PRED_DIR = "assignment2_output_final"        
VIS_DIR = "visualization_result"            


COLOR_POINT = (0, 0, 255)       
COLOR_BAR = (0, 255, 0)         
COLOR_TEXT = (255, 0, 0)        

def visualize_predictions():
    os.makedirs(VIS_DIR, exist_ok=True)
    
    pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith('.json')])
    print(f"🎨 Visualizing {len(pred_files)} images...")
    
    for f in tqdm(pred_files):
        
        pred_path = os.path.join(PRED_DIR, f)
        
        
        with open(pred_path, 'r') as jf:
            data = json.load(jf)
            
        
        img_name = data.get('image_file', f.replace('.json', '.png'))
        img_path = os.path.join(IMG_DIR, img_name)
        
        
        if not os.path.exists(img_path):
            img_path = os.path.join(IMG_DIR, img_name.replace('.png', '.jpg'))
            
        if not os.path.exists(img_path):
            continue
            
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        # --- DRAWING LOGIC ---
        series_list = data.get('error_bars', [])
        
        for series in series_list:
            for pt in series.get('points', []):
                # Coordinates (Prediction)
                cx = int(float(pt['data_point']['x']))
                cy = int(float(pt['data_point']['y']))
                
                uy = int(float(pt['upper_error_bar']['y']))
                ly = int(float(pt['lower_error_bar']['y']))
                
                # 1. Draw Center Point (Red Dot)
                cv2.circle(img, (cx, cy), 3, COLOR_POINT, -1)
                
                # 2. Draw Vertical Line (Upper)
                cv2.line(img, (cx, cy), (cx, uy), COLOR_BAR, 2)
                
                # 3. Draw Vertical Line (Lower)
                cv2.line(img, (cx, cy), (cx, ly), COLOR_BAR, 2)
                
                # 4. Draw Horizontal Caps (Optional cosmetic touch)
                cap_width = 5
                # Upper Cap
                cv2.line(img, (cx - cap_width, uy), (cx + cap_width, uy), COLOR_BAR, 2)
                # Lower Cap
                cv2.line(img, (cx - cap_width, ly), (cx + cap_width, ly), COLOR_BAR, 2)

        
        cv2.putText(img, "Predicted Error Bars (Green)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

        
        save_path = os.path.join(VIS_DIR, img_name)
        cv2.imwrite(save_path, img)
        
    print(f" Visualization Complete! Check the '{VIS_DIR}' folder.")

if __name__ == "__main__":
    visualize_predictions()