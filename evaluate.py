import json
import os
import numpy as np
from tqdm import tqdm


GT_DIR = "company_dataset/labels"      
# PRED_DIR = "assignment2_output"         
# PRED_DIR = "assignment2_output_resnet"
PRED_DIR = "assignment2_output_final"

def calculate_accuracy():
    print(f" Evaluating Accuracy...")
    
    gt_files = sorted([f for f in os.listdir(GT_DIR) if f.endswith('.json')])
    pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith('.json')])
    
    total_points = 0
    total_mae = 0.0  
    correct_5px = 0  
    correct_10px = 0 
    
    files_evaluated = 0

    for f in tqdm(gt_files):
        if f not in pred_files:
            continue
            
        gt_path = os.path.join(GT_DIR, f)
        pred_path = os.path.join(PRED_DIR, f)
        
        try:
            with open(gt_path, 'r') as gf: gt_data = json.load(gf)
            with open(pred_path, 'r') as pf: pred_data = json.load(pf)
            

            if isinstance(gt_data, list): gt_series = gt_data
            else: gt_series = gt_data.get('data_points', []) or gt_data.get('error_bars', [])
            
            # Prediction List
            if isinstance(pred_data, list): pred_series = pred_data
            else: pred_series = pred_data.get('error_bars', [])

            # --- COMPARE POINTS ---
            for i, series in enumerate(gt_series):
                if i >= len(pred_series): break
                
                gt_pts = series.get('points', [])
                pred_pts = pred_series[i].get('points', [])
                
                for j, g_pt in enumerate(gt_pts):
                    if j >= len(pred_pts): break
                    
                    p_pt_wrapper = pred_pts[j] 
                    
                    # Ground Truth Distances
                    actual_top = float(g_pt.get('topBarPixelDistance', 0))
                    actual_bot = float(g_pt.get('bottomBarPixelDistance', 0))
                    

                    c_y = float(p_pt_wrapper['data_point']['y'])
                    
                    # Predicted Upper/Lower Y
                    u_y = float(p_pt_wrapper['upper_error_bar']['y'])
                    l_y = float(p_pt_wrapper['lower_error_bar']['y'])
                    
                    # Calculate Predicted Distance
                    pred_top = abs(c_y - u_y)
                    pred_bot = abs(l_y - c_y)
                    
                    # --- ERROR CALCULATION ---
                    err_top = abs(actual_top - pred_top)
                    err_bot = abs(actual_bot - pred_bot)
                    
                    # Stats Update
                    total_mae += (err_top + err_bot)
                    total_points += 2 
                    
                    # Accuracy Threshold Check
                    if err_top <= 5: correct_5px += 1
                    if err_bot <= 5: correct_5px += 1
                    
                    if err_top <= 10: correct_10px += 1
                    if err_bot <= 10: correct_10px += 1
            
            files_evaluated += 1
            
        except Exception as e:
            print(f"Error evaluating {f}: {e}")
            continue

   
    if total_points == 0:
        print(" No matching points found to evaluate!")
        return

    avg_mae = total_mae / total_points
    acc_5 = (correct_5px / total_points) * 100
    acc_10 = (correct_10px / total_points) * 100
    
    print("\n" + "="*40)
    print(f" EVALUATION REPORT ({files_evaluated} Files)")
    print("="*40)
    print(f"🔹 Total Keypoints Evaluated : {total_points}")
    print(f"🔹 Mean Absolute Error (MAE) : {avg_mae:.4f} pixels")
    print("-" * 40)
    print(f" Accuracy (Error <= 5px)   : {acc_5:.2f}%")
    print(f" Accuracy (Error <= 10px)  : {acc_10:.2f}%")
    print("="*40)

if __name__ == "__main__":
    calculate_accuracy()