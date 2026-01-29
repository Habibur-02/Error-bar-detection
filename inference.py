import torch
import cv2
import json
import os
import numpy as np
from tqdm import tqdm
from model_utils import ErrorBarCNN, PATCH_H, PATCH_W, DEVICE


INPUT_IMAGE_DIR = "company_dataset/images"
INPUT_LABEL_DIR = "company_dataset/labels"
OUTPUT_DIR = "output_predictions"
# MODEL_PATH = "best_model.pth"  
# MODEL_PATH = "final_model.pth"   # <-- নতুন লাইন
# নতুন লাইন হবে (Phase 3 মডেল):
MODEL_PATH = "final_model_advanced.pth"


os.makedirs(OUTPUT_DIR, exist_ok=True)

def predict_single_image(model, img_path, json_path):
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    h, w = img.shape 
    
    
    with open(json_path, 'r') as f:
        input_data = json.load(f)
    
    
    raw_lines = []
    if isinstance(input_data, list):
        
        raw_lines = input_data
    elif isinstance(input_data, dict):
        
        raw_lines = input_data.get('data_points', [])
        
        if not raw_lines and 'points' in input_data:
             raw_lines = [input_data]

    output_lines = []

    for line_group in raw_lines:
        
        line_name = "unknown"
        if "label" in line_group and isinstance(line_group["label"], dict):
            line_name = line_group["label"].get("lineName", "unknown")
        elif "lineName" in line_group:
             line_name = line_group["lineName"]

        predicted_points = []
        points_list = line_group.get('points', [])
        
        for pt in points_list:
            cx, cy = pt['x'], pt['y']
            
            y1 = int(cy - PATCH_H // 2)
            y2 = y1 + PATCH_H
            x1 = int(cx - PATCH_W // 2)
            x2 = x1 + PATCH_W
            
            img_y1, img_y2 = max(0, y1), min(h, y2)
            img_x1, img_x2 = max(0, x1), min(w, x2)
            
            patch = np.ones((PATCH_H, PATCH_W), dtype=np.uint8) * 255
            
            if img_y1 < img_y2 and img_x1 < img_x2:
                crop_part = img[img_y1:img_y2, img_x1:img_x2]
                out_y1 = max(0, -y1)
                out_x1 = max(0, -x1)
                part_h, part_w = crop_part.shape
                patch[out_y1 : out_y1 + part_h, out_x1 : out_x1 + part_w] = crop_part
            
            # --- Prediction ---
            patch = patch.astype(np.float32) / 255.0
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                preds = model(patch_tensor).cpu().numpy()[0]
            
            up_dist = max(0, preds[0])
            down_dist = max(0, preds[1])
            

            raw_uy = cy - up_dist
            raw_ly = cy + down_dist
            
            final_uy = max(0.0, raw_uy)
            
            final_ly = min(float(h), raw_ly)
            
            predicted_points.append({
                "data_point": {"x": cx, "y": cy},
                "upper_error_bar": {"x": cx, "y": float(round(final_uy, 2))},
                "lower_error_bar": {"x": cx, "y": float(round(final_ly, 2))}
            })
            
        output_lines.append({
            "lineName": line_name,
            "points": predicted_points
        })
        
    return {
        "image_file": os.path.basename(img_path),
        "error_bars": output_lines
    }

def run_inference():
    print(f"Loading model from {MODEL_PATH}...")
    model = ErrorBarCNN().to(DEVICE)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
    except TypeError:
        model.load_state_dict(torch.load(MODEL_PATH))
        
    model.eval()
    
    label_files = [f for f in os.listdir(INPUT_LABEL_DIR) if f.endswith('.json')]
    print(f"Found {len(label_files)} files to process.")
    
    
    for label_file in tqdm(label_files):
        base_name = os.path.splitext(label_file)[0]
        json_path = os.path.join(INPUT_LABEL_DIR, label_file)
        
        img_path = os.path.join(INPUT_IMAGE_DIR, base_name + ".png")
        if not os.path.exists(img_path):
            img_path = os.path.join(INPUT_IMAGE_DIR, base_name + ".jpg")
        
        if os.path.exists(img_path):
            result = predict_single_image(model, img_path, json_path)
            if result:
                output_path = os.path.join(OUTPUT_DIR, label_file)
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)

if __name__ == "__main__":
    run_inference()