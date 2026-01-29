import torch
import json
import os
import cv2
import numpy as np

from model_utils_resnet import ErrorBarResNet, DEVICE, PATCH_H, PATCH_W
from tqdm import tqdm

# MODEL_PATH = "error_bar_resnet_best.pth" 
INPUT_IMG_DIR = "company_dataset/images"
INPUT_JSON_DIR = "company_dataset/labels"
# OUTPUT_DIR = "assignment2_output_resnet" 

MODEL_PATH = "final_interview_model.pth" 
OUTPUT_DIR = "assignment2_output_final"  

def predict_coordinates_resnet():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading ResNet Model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(" Model not found! Run 'python train_resnet.py' first.")
        return

    model = ErrorBarResNet().to(DEVICE)
    # Weights Only True for Safety
    try: model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    except: model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    json_files = [f for f in os.listdir(INPUT_JSON_DIR) if f.endswith('.json')]
    print(f" Processing {len(json_files)} files with ResNet-18...")
    
    for f in tqdm(json_files):
        json_path = os.path.join(INPUT_JSON_DIR, f)
        img_filename = f.replace(".json", ".png")
        img_path = os.path.join(INPUT_IMG_DIR, img_filename)
        
        if not os.path.exists(img_path):
            img_filename = f.replace(".json", ".jpg")
            img_path = os.path.join(INPUT_IMG_DIR, img_filename)
            
        if not os.path.exists(img_path): continue
            
        with open(json_path, 'r') as jf: input_data = json.load(jf)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        final_output = {"image_file": img_filename, "error_bars": []}
        series_list = input_data if isinstance(input_data, list) else input_data.get('data_points', []) or input_data.get('error_bars', [])

        for series in series_list:
            line_name = series.get('label', {}).get('lineName') or series.get('lineName', "Unknown_Line")
            processed_series = {"lineName": line_name, "points": []}

            for pt in series.get('points', []):
                cx = float(pt['x'])
                cy = float(pt['y'])
                
                pad_h, pad_w = PATCH_H // 2, PATCH_W // 2
                padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=255)
                cx_pad, cy_pad = cx + pad_w, cy + pad_h
                x1 = int(cx_pad - PATCH_W // 2)
                y1 = int(cy_pad - PATCH_H // 2)
                patch = padded[y1:y1+PATCH_H, x1:x1+PATCH_W]
                
                # Safety Check
                if patch.size == 0 or patch.shape != (PATCH_H, PATCH_W):
                    try: patch = cv2.resize(patch, (PATCH_W, PATCH_H))
                    except: patch = np.zeros((PATCH_H, PATCH_W), dtype=np.uint8)
                
                tensor = torch.from_numpy(patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    output = model(tensor).cpu().numpy()[0]
                
                top_dist = max(0, float(output[0]))
                bot_dist = max(0, float(output[1]))
                
                # Coordinate Calculation
                error_point_data = {
                    "data_point": {"x": cx, "y": cy},
                    "upper_error_bar": {"x": cx, "y": round(cy - top_dist, 4)},
                    "lower_error_bar": {"x": cx, "y": round(cy + bot_dist, 4)}
                }
                processed_series["points"].append(error_point_data)
            
            final_output["error_bars"].append(processed_series)
        
        output_path = os.path.join(OUTPUT_DIR, f)
        with open(output_path, 'w') as out_f: json.dump(final_output, out_f, indent=2)
            
    print(f" ResNet Output saved in: '{OUTPUT_DIR}'")

if __name__ == "__main__":
    predict_coordinates_resnet()