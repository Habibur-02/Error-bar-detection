import cv2
import os
import json
import numpy as np
from tqdm import tqdm


IMG_DIR = "../company_dataset/images"
LBL_DIR = "../company_dataset/labels"
OUTPUT_DIR = "canny_output"

def run_canny_scan():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    json_files = [f for f in os.listdir(LBL_DIR) if f.endswith('.json')]
    print(f"🕵️ Running Canny Edge + Pixel Scanning on {len(json_files)} images...")
    
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
        height, width = gray.shape 
        
        # Canny Edge Detection
        edges = cv2.Canny(gray, 50, 150)
        
        
        vis_img = img.copy()
        
        with open(json_path, 'r') as jf:
            data = json.load(jf)
            
        series_list = data if isinstance(data, list) else data.get('data_points', []) or data.get('error_bars', [])
        
        for series in series_list:
            for pt in series.get('points', []):
                cx = int(float(pt['x']))
                cy = int(float(pt['y']))
                
                
                if not (0 <= cx < width and 0 <= cy < height):
                    continue

                # --- PIXEL SCANNING LOGIC ---
                buffer = 8 
                max_scan = 150
                
                # Top Bar (Scan Upwards)
                top_y = cy
                for y in range(cy - buffer, cy - max_scan, -1):
                    
                    if y < 0: break 
                    
                    if edges[y, cx] > 0: 
                        top_y = y
                        break
                
                # Bottom Bar (Scan Downwards)
                bot_y = cy
                for y in range(cy + buffer, cy + max_scan):
                    
                    if y >= height: break 
                    
                    if edges[y, cx] > 0:
                        bot_y = y
                        break

                # --- DRAWING ---
                cv2.circle(vis_img, (cx, cy), 3, (0, 0, 255), -1) 
                cv2.line(vis_img, (cx, cy), (cx, top_y), (255, 0, 0), 2)
                cv2.line(vis_img, (cx, cy), (cx, bot_y), (255, 0, 0), 2)
                
                
                if 0 <= cx-5 and cx+5 < width:
                    if 0 <= top_y < height:
                        cv2.line(vis_img, (cx-5, top_y), (cx+5, top_y), (255, 0, 0), 2)
                    if 0 <= bot_y < height:
                        cv2.line(vis_img, (cx-5, bot_y), (cx+5, bot_y), (255, 0, 0), 2)

        
        combined = np.hstack((cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), vis_img))
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"canny_{img_name}"), combined)
        
    print(f" Check '{OUTPUT_DIR}' folder to see Canny performance!")

if __name__ == "__main__":
    run_canny_scan()