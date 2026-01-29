import matplotlib.pyplot as plt
import numpy as np
import os
import json
import uuid
import random
import cv2
import glob


DATASET_DIR = "dataset"
IMG_DIR = os.path.join(DATASET_DIR, "images")
LBL_DIR = os.path.join(DATASET_DIR, "labels")
COMPANY_LBL_DIR = os.path.join("company_dataset", "labels")
NUM_IMAGES = 3000


os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)


LINE_STYLES = ['-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x', '+']
COLORS = ['black', '#333333', '#555555', 'blue', 'red', 'green', 'purple']
PATTERNS = [None, '/', '\\', '|', '-', '+', 'x', '.', '*']
COMMON_LABELS = ["ymin", "ymax", "xmin", "xmax", "peak", "low", "target", ""]


REAL_STATS = {
    "deviations": [0, 15, 30],
    "prob_has_cap": 0.5,
    "error_lengths": [20, 50, 80, 120],
    "prob_point_label": 0.05
}

def format_number(val):

    if val == 0: return 0
    
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return val

def learn_from_real_data():
    global REAL_STATS
    json_files = glob.glob(os.path.join(COMPANY_LBL_DIR, "*.json"))
    
    if not json_files:
        print("⚠️ Warning: No company labels found! Using default random values.")
        return

    print(f"📊 Analyzing {len(json_files)} real label files...")
    
    all_deviations = []
    all_error_lengths = []
    total_points = 0
    points_with_cap = 0
    points_with_text_label = 0

    for jf in json_files:
        try:
            with open(jf, 'r') as f: data = json.load(f)
            series_list = data if isinstance(data, list) else data.get('error_bars', [])
            
            for series in series_list:
                points = series.get('points', [])
                for pt in points:
                    total_points += 1
                    dev = pt.get('deviationPixelDistance', 0)
                    if dev > 0:
                        points_with_cap += 1
                        all_deviations.append(dev)
                    top = pt.get('topBarPixelDistance', 0)
                    if top > 0: all_error_lengths.append(top)
                    lbl = pt.get('label', "").strip()
                    if lbl and not lbl.startswith("Pt_"):
                        points_with_text_label += 1
        except Exception: continue

    if total_points > 0:
        REAL_STATS["prob_has_cap"] = points_with_cap / total_points
        REAL_STATS["prob_point_label"] = points_with_text_label / total_points
        if all_deviations: REAL_STATS["deviations"] = all_deviations
        if all_error_lengths: REAL_STATS["error_lengths"] = all_error_lengths
            
    print(f"✅ Learning Complete! Cap Prob: {REAL_STATS['prob_has_cap']*100:.1f}%")

def apply_realistic_noise(img_path):
    img = cv2.imread(img_path)
    if img is None: return

    rand_val = random.random()

    # HEAVY NOISE (5%)
    if rand_val > 0.95:
        k = random.choice([5, 7])
        img = cv2.GaussianBlur(img, (k, k), 0)
        
        row, col, ch = img.shape
        sigma = 0.08**0.5
        gauss = np.random.normal(0, sigma, (row, col, ch)).reshape(row, col, ch)
        noisy = img + gauss * random.randint(50, 80)
        img = np.clip(noisy, 0, 255).astype(np.uint8)
        
        
        if random.random() > 0.5:
            rows, cols = img.shape[:2]
            angle = random.uniform(-1, 1)
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            img = cv2.warpAffine(img, M, (cols, rows), borderValue=(255,255,255))
            
    # MODERATE NOISE (20%)
    elif rand_val > 0.75:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
        if random.random() > 0.5:
             row, col, ch = img.shape
             sigma = 0.03**0.5
             gauss = np.random.normal(0, sigma, (row, col, ch)).reshape(row, col, ch)
             noisy = img + gauss * 30
             img = np.clip(noisy, 0, 255).astype(np.uint8)

    
    else:
        if random.random() > 0.6:
             img = cv2.GaussianBlur(img, (3, 3), 0.5)

    cv2.imwrite(img_path, img)

def generate_synthetic_data(index):
    DPI = 100
    fig_w = random.randint(7, 11)
    fig_h = random.randint(5, 9)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    ax = fig.add_subplot(111)

    plot_type = 'bar' if random.random() < 0.25 else 'line'
    num_series = random.randint(2, 5)
    all_series_data = []
    num_points = random.randint(8, 15)
    x = np.arange(1, num_points + 1)
    bar_width = 1.2 / num_series if plot_type == 'bar' else 0

    for i in range(num_series):
        base_y = random.uniform(30, 70)
        y = np.random.uniform(base_y - 20, base_y + 20, num_points)
        
        yerr_top = np.random.choice(REAL_STATS["error_lengths"], num_points) * np.random.uniform(0.01, 0.03)
        yerr_bot = np.random.choice(REAL_STATS["error_lengths"], num_points) * np.random.uniform(0.01, 0.03)

        color = random.choice(COLORS)
        label_name = f"Data_{i+1}"
        
        has_cap = random.random() < REAL_STATS["prob_has_cap"]
        cap_size_points = (random.choice(REAL_STATS["deviations"]) * 72.0 / DPI * random.uniform(0.8, 1.2)) if has_cap else 0
        deviation_px = (cap_size_points * DPI) / 72.0

        calc_x = x + np.random.uniform(-0.2, 0.2, num_points) if plot_type == 'line' else (x + (i * bar_width) - 0.6)

        series_info = {
            "x": x, "y": y, "yerr_top": yerr_top, "yerr_bot": yerr_bot,
            "label_name": label_name, "deviation_px": deviation_px,
            "calc_x": calc_x, "point_labels": []
        }
        
        linewidth = random.uniform(1, 2.5)
        markersize = random.uniform(6, 10)

        if plot_type == 'line':
            ls = random.choice(LINE_STYLES)
            mk = random.choice(MARKERS)
            if random.random() < 0.15: ls = ''
            ax.errorbar(calc_x, y, yerr=[yerr_bot, yerr_top], fmt=mk, linestyle=ls, 
                        color=color, ecolor=color, capsize=cap_size_points, 
                        label=label_name, alpha=random.uniform(0.7, 1.0),
                        linewidth=linewidth, markersize=markersize)
        elif plot_type == 'bar':
            pattern = random.choice(PATTERNS)
            ax.bar(calc_x, y, width=bar_width, yerr=[yerr_bot, yerr_top], 
                   color='white', edgecolor=color, hatch=pattern, 
                   capsize=cap_size_points, label=label_name,
                   linewidth=linewidth)

        for j in range(num_points):
            lbl_text = ""
            if random.random() < REAL_STATS["prob_point_label"]:
                lbl_text = random.choice(COMMON_LABELS)
                if lbl_text:
                    ax.text(calc_x[j], y[j] + yerr_top[j] + 2, lbl_text, 
                            fontsize=random.randint(8, 12), color='black', ha='center')
            series_info["point_labels"].append(lbl_text)
        
        all_series_data.append(series_info)

    if random.choice([True, False]): ax.legend(loc='best', fontsize='small')
    if random.random() > 0.7: ax.set_title(f"Measurement Report {index}")
    if random.choice([True, False]): ax.grid(True, linestyle=':', alpha=0.6)

    fig.canvas.draw()
    
    final_json_data = []
    img_height_px = fig_h * DPI

    for s_data in all_series_data:
        points_list = []
        calc_x = s_data["calc_x"]
        y = s_data["y"]
        yerr_top = s_data["yerr_top"]
        yerr_bot = s_data["yerr_bot"]
        dev_px_raw = s_data["deviation_px"]
        pt_lbls = s_data["point_labels"]

        for j in range(len(calc_x)):
            px, py_raw = ax.transData.transform((calc_x[j], y[j]))
            _, py_top_raw = ax.transData.transform((calc_x[j], y[j] + yerr_top[j]))
            _, py_bot_raw = ax.transData.transform((calc_x[j], y[j] - yerr_bot[j]))

            py = img_height_px - py_raw
            py_top = img_height_px - py_top_raw
            py_bot = img_height_px - py_bot_raw

            if px < 0 or py < 0 or px > (fig_w*DPI) or py > (fig_h*DPI): continue

            
            
            
            final_dev = dev_px_raw
            if random.random() < 0.95:
                final_dev = round(final_dev)

            
            dist_top = abs(py - py_top)
            dist_bot = abs(py_bot - py)

            if random.random() < 0.50:
                dist_top = round(dist_top)
            
            if random.random() < 0.50:
                dist_bot = round(dist_bot)

            points_list.append({
                "x": format_number(px), 
                "y": format_number(py),
                "label": pt_lbls[j] if pt_lbls[j] else f"Pt_{j}", 
                "topBarPixelDistance": format_number(dist_top),
                "bottomBarPixelDistance": format_number(dist_bot),
                "deviationPixelDistance": format_number(final_dev)
            })

        final_json_data.append({"label": {"lineName": s_data["label_name"]}, "points": points_list})

    unique_id = str(uuid.uuid4())
    img_path = os.path.join(IMG_DIR, f"{unique_id}.png")
    json_path = os.path.join(LBL_DIR, f"{unique_id}.json")
    plt.savefig(img_path, dpi=DPI)
    plt.close(fig)
    with open(json_path, 'w') as f: json.dump(final_json_data, f, indent=2)
    
    apply_realistic_noise(img_path)

if __name__ == "__main__":
    learn_from_real_data()
    print(f"Generating {NUM_IMAGES} Precision-Matched images...")
    for i in range(NUM_IMAGES):
        generate_synthetic_data(i)
        if (i+1) % 200 == 0:
            print(f"✅ Generated {i+1}/{NUM_IMAGES}")
    
    print("🎉 Dataset Generation Complete!")