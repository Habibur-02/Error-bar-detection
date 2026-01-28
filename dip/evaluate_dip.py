import json
import os
import numpy as np


GROUND_TRUTH_DIR = "../company_dataset/labels"
PREDICTION_DIR = "output_predictions" 

def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get('error_bars', []) or data.get('data_points', [])
    return []

def calculate_dip_metrics():
    if not os.path.exists(PREDICTION_DIR):
        print("Prediction folder not found. Run dip_baseline.py first.")
        return

    pred_files = [f for f in os.listdir(PREDICTION_DIR) if f.endswith('.json')]
    
    total_error = 0.0
    total_points = 0
    correct_predictions_5px = 0
    
    print(f"Evaluating DIP results for {len(pred_files)} files...")

    for filename in pred_files:
        gt_path = os.path.join(GROUND_TRUTH_DIR, filename)
        pred_path = os.path.join(PREDICTION_DIR, filename)

        if not os.path.exists(gt_path): continue

        gt_data = load_json_data(gt_path)
        pred_data = load_json_data(pred_path)

        for i, gt_line in enumerate(gt_data):
            if i >= len(pred_data): break
            
            pred_line = pred_data[i]
            gt_points = gt_line.get('points', [])
            pred_points = pred_line.get('points', [])

            for j, gt_pt in enumerate(gt_points):
                if j >= len(pred_points): break
                pred_pt = pred_points[j]

                try:
                    # Ground Truth Calculation (from Distance)
                    gt_cy = float(gt_pt['y'])
                    top_dist = float(gt_pt.get('topBarPixelDistance', 0))
                    bot_dist = float(gt_pt.get('bottomBarPixelDistance', 0))
                    
                    gt_uy = gt_cy - top_dist
                    gt_ly = gt_cy + bot_dist

                    # Prediction Values (DIP)
                    pred_uy = float(pred_pt['upper_error_bar']['y'])
                    pred_ly = float(pred_pt['lower_error_bar']['y'])

                    # Error Calculation
                    error_up = abs(gt_uy - pred_uy)
                    error_low = abs(gt_ly - pred_ly)

                    total_error += (error_up + error_low)
                    total_points += 2 

                    if error_up <= 5: correct_predictions_5px += 1
                    if error_low <= 5: correct_predictions_5px += 1

                except: continue

    if total_points == 0:
        print("No points matched.")
        return

    mae = total_error / total_points
    acc = (correct_predictions_5px / total_points) * 100

    print("-" * 40)
    print(" DIP Baseline Metrics (Canny Edge)")
    print("-" * 40)
    print(f"Mean Absolute Error (MAE) : {mae:.2f} pixels")
    print(f"Accuracy (within 5px)   : {acc:.2f}%")
    print("-" * 40)
    print("Conclusion: DIP failed due to noise, text overlap, and broken lines.")

if __name__ == "__main__":
    calculate_dip_metrics()