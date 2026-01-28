import cv2
import json
import os
import numpy as np
from tqdm import tqdm


IMAGE_DIR = "../company_dataset/images"
LABEL_DIR = "../company_dataset/labels"
OUTPUT_DIR = "output_predictions" 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_error_bar_dip(img_path, json_path):
   
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    h, w = img.shape
    

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    
    edges = cv2.Canny(blurred, 50, 150)
    
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    raw_lines = data if isinstance(data, list) else data.get('error_bars', [])
    
    if not raw_lines and isinstance(data, dict):
        raw_lines = data.get('data_points', [])

    output_lines = []

    for line in raw_lines:
        
        line_name = line.get("lineName", "unknown")
        if isinstance(line.get("label"), dict):
             line_name = line["label"].get("lineName", "unknown")

        predicted_points = []
        
        for pt in line.get('points', []):
            try:
                cx = int(float(pt['x']))
                cy = int(float(pt['y']))
                
                
                SEARCH_LIMIT = 150 
                
                # Scan Upwards
                pred_uy = cy
                
                curr_y = cy - 5
                found_up = False
                
                while curr_y > max(0, cy - SEARCH_LIMIT):
                    if edges[curr_y, cx] > 0: 
                        pred_uy = curr_y
                        found_up = True
                        break
                    curr_y -= 1
                
                # Scan Downwards
                pred_ly = cy
                curr_y = cy + 5
                found_down = False
                
                while curr_y < min(h, cy + SEARCH_LIMIT):
                    if edges[curr_y, cx] > 0: 
                        pred_ly = curr_y
                        found_down = True
                        break
                    curr_y += 1
                
                
                predicted_points.append({
                    "data_point": {"x": cx, "y": cy},
                    "upper_error_bar": {"x": cx, "y": float(pred_uy)},
                    "lower_error_bar": {"x": cx, "y": float(pred_ly)}
                })

            except Exception:
                continue
            
        output_lines.append({
            "lineName": line_name,
            "points": predicted_points
        })

    return {
        "image_file": os.path.basename(img_path),
        "error_bars": output_lines
    }

def run_dip_baseline():
    if not os.path.exists(LABEL_DIR):
        print(f"Error: Dataset not found at {LABEL_DIR}")
        print("Make sure you are running this script from inside the 'DIP' folder.")
        return

    files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.json')]
    print(f"Running DIP Baseline (Canny Edge) on {len(files)} files...")
    
    for f in tqdm(files):
        
        base_name = os.path.splitext(f)[0]
        img_path = os.path.join(IMAGE_DIR, base_name + ".png")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGE_DIR, base_name + ".jpg")
            
        if os.path.exists(img_path):
            res = detect_error_bar_dip(img_path, os.path.join(LABEL_DIR, f))
            if res:
                with open(os.path.join(OUTPUT_DIR, f), 'w') as out:
                    json.dump(res, out, indent=2)

if __name__ == "__main__":
    run_dip_baseline()