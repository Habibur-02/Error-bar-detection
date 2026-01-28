import json
import os
import numpy as np

GROUND_TRUTH_DIR = "company_dataset/labels"  # আসল উত্তর (Distance Format)
PREDICTION_DIR = "output_predictions"      # আপনার মডেলের উত্তর (Coordinate Format)

def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get('error_bars', []) or data.get('data_points', [])
    return []

def calculate_metrics():
    pred_files = [f for f in os.listdir(PREDICTION_DIR) if f.endswith('.json')]
    
    total_error = 0.0
    total_points = 0
    correct_predictions_5px = 0
    correct_predictions_2px = 0

    print(f"Evaluating {len(pred_files)} files...")

    match_count = 0

    for filename in pred_files:
        gt_path = os.path.join(GROUND_TRUTH_DIR, filename)
        pred_path = os.path.join(PREDICTION_DIR, filename)

        if not os.path.exists(gt_path):
            continue

        gt_data = load_json_data(gt_path)
        pred_data = load_json_data(pred_path)


        
        for i, gt_line in enumerate(gt_data):
            if i >= len(pred_data): break # সেফটি চেক
            
            pred_line = pred_data[i]
            
            gt_points = gt_line.get('points', [])
            pred_points = pred_line.get('points', [])

            for j, gt_pt in enumerate(gt_points):
                if j >= len(pred_points): break
                
                pred_pt = pred_points[j]

                try:

                    gt_cy = float(gt_pt['y'])
                    
                    # Distances
                    top_dist = float(gt_pt.get('topBarPixelDistance', 0))
                    bot_dist = float(gt_pt.get('bottomBarPixelDistance', 0))
                    
                    gt_uy = gt_cy - top_dist
                    gt_ly = gt_cy + bot_dist


                    pred_uy = float(pred_pt['upper_error_bar']['y'])
                    pred_ly = float(pred_pt['lower_error_bar']['y'])

                    # --- ERROR CALCULATION ---
                    error_up = abs(gt_uy - pred_uy)
                    error_low = abs(gt_ly - pred_ly)

                    total_error += (error_up + error_low)
                    total_points += 2 
                    match_count += 1

                    # Accuracy Checking
                    if error_up <= 5: correct_predictions_5px += 1
                    if error_low <= 5: correct_predictions_5px += 1
                    
                    if error_up <= 2: correct_predictions_2px += 1
                    if error_low <= 2: correct_predictions_2px += 1

                except (KeyError, ValueError, TypeError) as e:
                    continue

    if total_points == 0:
        print("No matchable points found! Check if filenames match in both folders.")
        return

    mae = total_error / total_points
    acc_5 = (correct_predictions_5px / total_points) * 100
    acc_2 = (correct_predictions_2px / total_points) * 100

    print("-" * 40)
    print("📊 Evaluation Metrics Report")
    print("-" * 40)
    print(f"Files Processed        : {len(pred_files)}")
    print(f"Total Points Matched   : {match_count}")
    print(f"Mean Absolute Error    : {mae:.2f} pixels")
    print(f"Accuracy (within 5px)  : {acc_5:.2f}%")
    print(f"Accuracy (within 2px)  : {acc_2:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    calculate_metrics()