import torch
import json
import os
import cv2
import numpy as np
from model_utils import ErrorBarCNN, DEVICE, PATCH_H, PATCH_W
from tqdm import tqdm


MODEL_PATH = "error_bar_model.pth"
INPUT_IMG_DIR = "company_dataset/images"
INPUT_JSON_DIR = "company_dataset/labels"
OUTPUT_DIR = "assignment2_output"

def predict_coordinates():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading trained model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found! Please run train.py first.")
        return

    
    model = ErrorBarCNN().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    except:
        # Fallback for older pytorch versions
        model.load_state_dict(torch.load(MODEL_PATH))
        
    model.eval()
    
    json_files = [f for f in os.listdir(INPUT_JSON_DIR) if f.endswith('.json')]
    print(f"🚀 Processing {len(json_files)} files for Assignment 2 Output...")
    
    for f in tqdm(json_files):
        json_path = os.path.join(INPUT_JSON_DIR, f)
        
        
        img_filename = f.replace(".json", ".png")
        img_path = os.path.join(INPUT_IMG_DIR, img_filename)
        
        
        if not os.path.exists(img_path):
            img_filename = f.replace(".json", ".jpg")
            img_path = os.path.join(INPUT_IMG_DIR, img_filename)
            
        if not os.path.exists(img_path):
            continue
            
        with open(json_path, 'r') as jf:
            input_data = json.load(jf)
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        
        final_output = {
            "image_file": img_filename,
            "error_bars": []
        }
        
        if isinstance(input_data, list):
            series_list = input_data
        else:
            series_list = input_data.get('data_points', []) or input_data.get('error_bars', [])

        for series in series_list:
            line_name = series.get('label', {}).get('lineName') or series.get('lineName', "Unknown_Line")
            
            processed_series = {
                "lineName": line_name,
                "points": []
            }

            for pt in series.get('points', []):
                cx = float(pt['x'])
                cy = float(pt['y'])
                
                # --- PATCH EXTRACTION WITH SAFETY CHECKS ---
                pad_h, pad_w = PATCH_H // 2, PATCH_W // 2
                padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=255)
                
                cx_pad, cy_pad = cx + pad_w, cy + pad_h
                x1 = int(cx_pad - PATCH_W // 2)
                y1 = int(cy_pad - PATCH_H // 2)
                
                # Slicing
                patch = padded[y1:y1+PATCH_H, x1:x1+PATCH_W]
                
                
                
                if patch.size == 0:
                    patch = np.zeros((PATCH_H, PATCH_W), dtype=np.uint8)
                
                # 2. Check shape mismatch and resize safely
                if patch.shape != (PATCH_H, PATCH_W):
                    try:
                        patch = cv2.resize(patch, (PATCH_W, PATCH_H))
                    except cv2.error:
                        
                        patch = np.zeros((PATCH_H, PATCH_W), dtype=np.uint8)
                # --- CRITICAL FIX END ---
                
                tensor = torch.from_numpy(patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
                
                # --- PREDICTION ---
                with torch.no_grad():
                    output = model(tensor).cpu().numpy()[0]
                
                top_dist = max(0, float(output[0]))
                bot_dist = max(0, float(output[1]))
                
                
                upper_y = cy - top_dist
                lower_y = cy + bot_dist
                
                error_point_data = {
                    "data_point": {
                        "x": cx,
                        "y": cy
                    },
                    "upper_error_bar": {
                        "x": cx,
                        "y": round(upper_y, 4)
                    },
                    "lower_error_bar": {
                        "x": cx,
                        "y": round(lower_y, 4)
                    }
                }
                
                processed_series["points"].append(error_point_data)
            
            final_output["error_bars"].append(processed_series)
        
        # Save Output JSON
        output_path = os.path.join(OUTPUT_DIR, f)
        with open(output_path, 'w') as out_f:
            json.dump(final_output, out_f, indent=2)
            
    print(f"✅ Assignment 2 Complete! Output saved in folder: '{OUTPUT_DIR}'")

if __name__ == "__main__":
    predict_coordinates()