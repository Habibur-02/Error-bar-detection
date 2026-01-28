import cv2
import json
import os
import numpy as np
from tqdm import tqdm

IMAGE_DIR = "company_dataset/images"       # আসল ছবি যেখানে আছে
PREDICTION_DIR = "output_predictions"      # আপনার জেনারেট করা JSON যেখানে আছে
OUTPUT_VIS_DIR = "visualization_results"   # যেখানে লাল দাগ দেওয়া ছবি সেভ হবে

COLOR_BAR = (0, 0, 255)    # লাল (Error Bar)
COLOR_POINT = (255, 0, 0)  # নীল (Data Point)

os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

def draw_error_bars(img_path, json_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    raw_lines = []
    if isinstance(data, dict):
        raw_lines = data.get('error_bars', [])
    elif isinstance(data, list):
        raw_lines = data

    for line in raw_lines:
        for pt in line.get('points', []):
            try:
                cx = int(float(pt['data_point']['x']))
                cy = int(float(pt['data_point']['y']))
                
                ux = int(float(pt['upper_error_bar']['x']))
                uy = int(float(pt['upper_error_bar']['y']))
                lx = int(float(pt['lower_error_bar']['x']))
                ly = int(float(pt['lower_error_bar']['y']))

                cv2.line(img, (ux, uy), (lx, ly), COLOR_BAR, 2)

                cap_width = 4
                cv2.line(img, (ux - cap_width, uy), (ux + cap_width, uy), COLOR_BAR, 2)
                cv2.line(img, (lx - cap_width, ly), (lx + cap_width, ly), COLOR_BAR, 2)

                cv2.circle(img, (cx, cy), 3, COLOR_POINT, -1)
            
            except (KeyError, ValueError) as e:
                continue

    cv2.imwrite(save_path, img)

def run_visualization():
    files = [f for f in os.listdir(PREDICTION_DIR) if f.endswith('.json')]
    print(f"Visualizing {len(files)} images from '{PREDICTION_DIR}'...")

    for json_file in tqdm(files):
        base_name = os.path.splitext(json_file)[0]
        
        img_path = os.path.join(IMAGE_DIR, base_name + ".png")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGE_DIR, base_name + ".jpg")

        if os.path.exists(img_path):
            save_path = os.path.join(OUTPUT_VIS_DIR, "annotated_" + base_name + ".png")
            
            draw_error_bars(img_path, os.path.join(PREDICTION_DIR, json_file), save_path)
        else:
            pass

if __name__ == "__main__":
    run_visualization()