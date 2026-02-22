import csv
import math

# Configuration
WIDTH = 800
HEIGHT = 500
MARGIN_L = 150 # More margin for model names on Left
MARGIN_R = 50
MARGIN_T = 60
MARGIN_B = 60
PLOT_W = WIDTH - MARGIN_L - MARGIN_R
PLOT_H = HEIGHT - MARGIN_T - MARGIN_B

COLORS = {
    'bg': '#FFFFFF',
    'border': '#000000',
    'grid': '#E0E0E0',
    'text': '#000000',
    'val_acc': '#7789B7',    # Darker Blue
    'val_f1': '#9DACCB',     # Lighter Blue
    'train_loss': '#ACBF9F', # Green
    'val_loss': '#EB6969',   # Red
}

INPUT_FILE = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\train\best_model_performance.csv"
OUTPUT_ACC = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\train\final_comparison_acc_f1.svg"
OUTPUT_LOSS = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\train\final_comparison_loss.svg"

def clean_label(name):
    name = name.lower()
    base = ""
    if 'densenet121' in name: base = "DenseNet121"
    elif 'resnet18' in name: base = "ResNet18"
    elif 'resnet50' in name: base = "ResNet50"
    elif 'vit' in name: base = "ViT"
    else: base = name
    
    if 'cbam' in name:
        base += "-CBAM"
    return base

def load_data():
    data = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'name': clean_label(row['ShortName']),
                'val_acc': float(row['Val Acc']),
                'val_f1': float(row['Val F1']),
                'val_loss': float(row['Val Loss']),
                'train_loss': float(row['Train Loss'])
            })
    # Sort by name reverse so first item is at top (if using standard SVG coords where Y increases down)
    # But usually we want alphabetical top to bottom
    data.sort(key=lambda x: x['name'], reverse=True) 
    return data

def create_horizontal_dumbbell(data, metrics, output_file, title, x_label):
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">')
    svg.append(f'<rect width="100%" height="100%" fill="{COLORS["bg"]}"/>')
    
    # Determine X Range (Values)
    vals = []
    for d in data:
        for m, _ in metrics:
            vals.append(d[m])
            
    min_val = min(vals)
    max_val = max(vals)
    
    # Pad Range
    span = max_val - min_val
    if span == 0: span = 0.1
    
    # Add 10% padding
    x_min = min_val - span * 0.1
    x_max = max_val + span * 0.1
    
    # Special scaling for Acc/F1
    if all('acc' in m or 'f1' in m for m, _ in metrics):
        if x_max > 1.0: x_max = 1.005
        if span < 0.05:
            x_min = max(0, min_val - 0.02)
            x_max = min(1.005, max_val + 0.02)
            
    def sx(val):
        return MARGIN_L + ((val - x_min) / (x_max - x_min)) * PLOT_W
        
    def sy(idx):
        # Evenly space rows. 
        # idx 0 is top
        step = PLOT_H / len(data)
        return MARGIN_T + idx * step + step / 2

    # Grid X (Vertical lines)
    n_ticks = 6
    for i in range(n_ticks):
        val = x_min + i * (x_max - x_min) / (n_ticks - 1)
        x = sx(val)
        svg.append(f'<line x1="{x}" y1="{MARGIN_T}" x2="{x}" y2="{MARGIN_T+PLOT_H}" stroke="{COLORS["grid"]}" stroke-width="1" stroke-dasharray="4"/>')
        
        lbl = f"{val:.4f}" if span < 0.05 else f"{val:.2f}"
        svg.append(f'<text x="{x}" y="{MARGIN_T+PLOT_H+20}" font-family="Arial" font-size="12" text-anchor="middle">{lbl}</text>')
        svg.append(f'<line x1="{x}" y1="{MARGIN_T+PLOT_H}" x2="{x}" y2="{MARGIN_T+PLOT_H+5}" stroke="black" stroke-width="1"/>')

    # Y Axis Labels (Models)
    for i, d in enumerate(data):
        y = sy(i)
        svg.append(f'<text x="{MARGIN_L-10}" y="{y+4}" font-family="Arial" font-size="12" text-anchor="end">{d["name"]}</text>')
        # Ticks
        svg.append(f'<line x1="{MARGIN_L}" y1="{y}" x2="{MARGIN_L-5}" y2="{y}" stroke="black" stroke-width="1"/>')

    # Dumbbell Lines & Points
    marker_radius = 6
    
    for i, d in enumerate(data):
        y = sy(i)
        
        # Draw connecting line if we have exactly 2 metrics
        if len(metrics) == 2:
            val1 = d[metrics[0][0]]
            val2 = d[metrics[1][0]]
            x1 = sx(val1)
            x2 = sx(val2)
            svg.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="black" stroke-width="1.5" />')
        
        # Draw points
        for k, (key, label) in enumerate(metrics):
            val = d[key]
            x = sx(val)
            color = COLORS[key]
            svg.append(f'<circle cx="{x}" cy="{y}" r="{marker_radius}" fill="{color}" stroke="black" stroke-width="1"/>')

    # Legend
    leg_x = MARGIN_L + 20
    leg_y = MARGIN_T - 30
    
    for k, (key, label) in enumerate(metrics):
        color = COLORS[key]
        lx = leg_x + k * 180
        svg.append(f'<circle cx="{lx}" cy="{leg_y}" r="{marker_radius}" fill="{color}" stroke="black" stroke-width="1"/>')
        svg.append(f'<text x="{lx+15}" y="{leg_y+5}" font-family="Arial" font-size="12">{label}</text>')

    # Border
    svg.append(f'<rect x="{MARGIN_L}" y="{MARGIN_T}" width="{PLOT_W}" height="{PLOT_H}" stroke="black" stroke-width="1.5" fill="none"/>')
    
    # X Label
    svg.append(f'<text x="{MARGIN_L + PLOT_W/2}" y="{MARGIN_T + PLOT_H + 45}" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle">{x_label}</text>')

    svg.append('</svg>')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Saved {output_file}")

def main():
    data = load_data()
    
    # Chart 1: Acc & F1
    create_horizontal_dumbbell(
        data, 
        [('val_acc', 'Validation Accuracy'), ('val_f1', 'Validation F1')], 
        OUTPUT_ACC, 
        "Model Performance (Accuracy & F1)",
        "Score"
    )
    
    # Chart 2: Loss
    create_horizontal_dumbbell(
        data,
        [('train_loss', 'Train Loss'), ('val_loss', 'Validation Loss')],
        OUTPUT_LOSS,
        "Model Loss",
        "Loss"
    )

if __name__ == '__main__':
    main()
