import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import uuid
import os
import random


os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/labels", exist_ok=True)

def apply_scan_effects(img_path):
    
    img = cv2.imread(img_path)
    
    # (Blur)
    if random.random() > 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # (Dirty Scan Effect)
    if random.random() > 0.4:
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

    # (JPEG Artifacts)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(50, 90)]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    img = cv2.imdecode(encimg, 1)

    cv2.imwrite(img_path, img)

def generate_complex_data(num_images=5):
    
    
    for _ in range(num_images):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        
        #(Line or Bar)
        chart_type = random.choice(['line', 'bar'])
        
        num_series = random.randint(1, 3)
        all_series_data = []
        output_error_bars = []
        
        colors = ['black', 'blue', 'red', 'green', 'gray']
        markers = ['o', 's', '^', 'D']
        
        # X-axis 
        x_points = np.linspace(1, 10, 6) 
        
        for i in range(num_series):
            y_points = np.random.uniform(20, 80, size=len(x_points))
            y_err = np.random.uniform(5, 15, size=len(x_points))
            
            line_name = f"Series_{i+1}"
            color = colors[i % len(colors)]
            
            # --- PLOTTING ---
            if chart_type == 'line':
                marker = markers[i % len(markers)]
                
                x_jitter = x_points + (np.random.uniform(-0.1, 0.1) if num_series > 1 else 0)
                ax.errorbar(x_jitter, y_points, yerr=y_err, fmt=marker, 
                            color=color, ecolor=color, capsize=4, label=line_name)
                current_x = x_jitter
            
            else: 
                width = 0.8 / num_series
                x_bar = x_points + (i * width) - (0.4 if num_series > 1 else 0)
                ax.bar(x_bar, y_points, width=width, yerr=y_err, 
                       capsize=4, color=color, alpha=0.7, label=line_name, error_kw={'ecolor': 'black'})
                current_x = x_bar


            for x, y, err in zip(current_x, y_points, y_err):
                if random.random() > 0.6: 
                    text_label = random.choice(["p<.05", "*", "**", "n=12", "ns"])
                    
                    ax.text(x, y + err + 2, text_label, ha='center', fontsize=8)

            # --- COORDINATE CALCULATION ---
            
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            height = fig.canvas.get_width_height()[1]
            
            series_points_input = []
            series_points_output = []

            for x_val, y_val, err_val in zip(current_x, y_points, y_err):
                # Data Point Pixel (Center)
                c_pix = ax.transData.transform((x_val, y_val))
                cx, cy = c_pix[0], height - c_pix[1]
                
                # Upper Error Tip (y + err)
                u_pix = ax.transData.transform((x_val, y_val + err_val))
                ux, uy = u_pix[0], height - u_pix[1]
                
                # Lower Error Tip (y - err)
                l_pix = ax.transData.transform((x_val, y_val - err_val))
                lx, ly = l_pix[0], height - l_pix[1]
                
                # Input JSON 
                series_points_input.append({
                    "x": round(cx, 2), 
                    "y": round(cy, 2)
                })
                
                # Output JSON 
                series_points_output.append({
                    "data_point": {"x": round(cx, 2), "y": round(cy, 2)},
                    "upper_error_bar": {"x": round(cx, 2), "y": round(uy, 2)},
                    "lower_error_bar": {"x": round(cx, 2), "y": round(ly, 2)}
                })

            all_series_data.append({"lineName": line_name, "points": series_points_input})
            output_error_bars.append({"lineName": line_name, "points": series_points_output})

        
        ax.set_title("Generated Scientific Plot")
        ax.set_xlabel("Time / Group")
        ax.set_ylabel("Value (nmol/L)")
        if num_series > 1: ax.legend()
        
        # --- SAVING ---
        file_id = str(uuid.uuid4())
        img_filename = f"{file_id}.png"
        
        # Image Save
        img_path = f"dataset/images/{img_filename}"
        fig.savefig(img_path)
        plt.close(fig)
        
        # Apply Noise/Blur mimicking your scanned dataset
        apply_scan_effects(img_path)
        
        # Save Output JSON (Ground Truth) - 
        with open(f"dataset/labels/{file_id}_output.json", 'w') as f:
            json.dump({
                "image_file": img_filename,
                "error_bars": output_error_bars
            }, f, indent=2)
            
        # Save Input JSON (To feed the model)
        with open(f"dataset/labels/{file_id}_input.json", 'w') as f:
            json.dump({
                "image_file": img_filename,
                "data_points": all_series_data
            }, f, indent=2)


generate_complex_data(num_images=3000) 
print("Done! Dataset generated with Bar Charts & Noise.")

