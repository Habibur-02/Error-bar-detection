import matplotlib.pyplot as plt
import numpy as np
import json
import os
import uuid


output_dir = "synthetic_dataset"
images_dir = os.path.join(output_dir, "images")
labels_dir = os.path.join(output_dir, "labels")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

def generate_synthetic_plot(idx):
    num_points = np.random.randint(5, 10)
    x_values = np.linspace(0, 10, num_points)

    series_data = []
    num_series = np.random.randint(1, 4)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i in range(num_series):
        base_y = np.random.rand() * 5 + np.random.randn(num_points)
        y_values = np.abs(base_y * 10) + 5

        deviation_values = np.random.rand(num_points) * 5  

        error_bar_lengths = deviation_values * 2

        color = colors[i % len(colors)]
        label_name = f"Series_{i}"

        (_, caps, _) = ax.errorbar(x_values, y_values, yerr=error_bar_lengths,
                                   fmt='-o', label=label_name, color=color, capsize=4)

        series_info = {
            "label": {"lineName": label_name},
            "points": []
        }


        fig.canvas.draw()

        trans = ax.transData.transform


        p0 = trans((0,0))
        p1 = trans((0,1))
        pixels_per_unit_y = abs(p1[1] - p0[1])

        for j in range(num_points):
            x, y = x_values[j], y_values[j]
            deviation_val = deviation_values[j]
            bar_len = error_bar_lengths[j]

            pixel_pos = trans((x, y))
            px, py = pixel_pos[0], pixel_pos[1]

      
            img_height = fig.get_figheight() * fig.get_dpi()
            py_image = img_height - py


            top_dist = bar_len * pixels_per_unit_y
            bottom_dist = bar_len * pixels_per_unit_y
            dev_dist = deviation_val * pixels_per_unit_y

            point_data = {
                "x": px,
                "y": py_image,
                "label": "",
                "topBarPixelDistance": top_dist,
                "bottomBarPixelDistance": bottom_dist,
                "deviationPixelDistance": dev_dist
            }
            series_info["points"].append(point_data)

        series_data.append(series_info)

    ax.legend()
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    file_id = str(uuid.uuid4())
    img_filename = f"{file_id}.png"
    json_filename = f"{file_id}.json"

    plt.savefig(os.path.join(images_dir, img_filename))
    plt.close()

    with open(os.path.join(labels_dir, json_filename), 'w') as f:
        json.dump(series_data, f, indent=2)

for k in range(5):
    generate_synthetic_plot(k)
    print(f"Generated image {k+1}")

print("Dataset generation complete")
