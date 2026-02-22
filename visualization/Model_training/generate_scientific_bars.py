import csv
import math

# Configuration
WIDTH = 1000
HEIGHT = 600
MARGIN_LEFT = 80
MARGIN_BOTTOM = 100
MARGIN_RIGHT = 150
MARGIN_TOP = 80

PLOT_X = MARGIN_LEFT
PLOT_Y = MARGIN_TOP
PLOT_W = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
PLOT_H = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

# Scientific Colors (Viridis-like / Paired)
COLORS = {
    'bg': '#FFFFFF',
    'text': '#333333',
    'grid': '#EAEAEA',
    'bar_border': '#333333',
    # Metric Colors
    'acc': '#2b7bba',      # Strong Blue
    'f1': '#36a2eb',       # Lighter Blue
    'val_loss': '#d62728', # Brick Red
    'train_loss': '#ff7f0e'# Orange
}

INPUT_FILE = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\train\best_model_performance.csv"
OUT_ACC_F1 = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\train\scientific_comparison_acc_f1.svg"
OUT_LOSS = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\train\scientific_comparison_loss.svg"

def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('ShortName', row.get('Model', 'Unknown'))
            # Clean names
            if 'densenet121' in name: name = 'DenseNet121' + ('-CBAM' if 'cbam' in row['Model'].lower() else '')
            elif 'resnet18' in name: name = 'ResNet18' + ('-CBAM' if 'cbam' in row['Model'].lower() else '')
            elif 'resnet50' in name: name = 'ResNet50' + ('-CBAM' if 'cbam' in row['Model'].lower() else '')
            elif 'vit' in name: name = 'ViT'
            
            try:
                data.append({
                    'name': name,
                    'val_acc': float(row['Val Acc']),
                    'val_f1': float(row['Val F1']),
                    'val_loss': float(row['Val Loss']),
                    'train_loss': float(row['Train Loss'])
                })
            except ValueError:
                continue
    # Sort for better visualization? Maybe by Acc descending
    # data.sort(key=lambda x: x['val_acc'], reverse=True)
    return data

def create_grouped_bar_chart(output_file, data, metrics, title, y_label, y_range=None):
    n_groups = len(data)
    n_bars = len(metrics)
    
    # Spacing
    group_width = PLOT_W / n_groups
    padding_group = group_width * 0.2
    available_width = group_width - padding_group
    bar_width = available_width / n_bars
    
    # Y Scale
    if y_range:
        y_min, y_max = y_range
    else:
        max_val = 0
        for d in data:
            for k, _, _ in metrics:
                max_val = max(max_val, d[k])
        y_min = 0
        y_max = max_val * 1.15 # Space for labels
    
    def scale_y(val):
        return PLOT_Y + PLOT_H - (val - y_min) / (y_max - y_min) * PLOT_H

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">')
    svg.append(f'<rect width="100%" height="100%" fill="{COLORS["bg"]}"/>')
    
    # Styles
    svg.append('''
    <style>
        .title { font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; fill: #333; }
        .label { font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; text-anchor: middle; fill: #333; }
        .tick-label { font-family: Arial, sans-serif; font-size: 12px; text-anchor: end; fill: #555; }
        .x-tick-label { font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; fill: #333; font-weight: bold; }
        .bar-val { font-family: Arial, sans-serif; font-size: 10px; text-anchor: middle; fill: #000; font-weight: bold; }
        .legend-text { font-family: Arial, sans-serif; font-size: 12px; text-anchor: start; fill: #333; }
        .grid { stroke: #E0E0E0; stroke-width: 1; stroke-dasharray: 4; }
        .axis { stroke: #333; stroke-width: 1.5; }
    </style>
    ''')
    
    # Title
    svg.append(f'<text x="{WIDTH/2}" y="40" class="title">{title}</text>')
    
    # Grid Lines (Horizontal)
    n_ticks = 6
    y_step = (y_max - y_min) / (n_ticks - 1)
    for i in range(n_ticks):
        val = y_min + i * y_step
        y_pos = scale_y(val)
        # Grid
        svg.append(f'<line x1="{PLOT_X}" y1="{y_pos}" x2="{PLOT_X+PLOT_W}" y2="{y_pos}" class="grid"/>')
        # Tick Label
        lbl = f"{val:.2f}".rstrip('0').rstrip('.') if val % 1 != 0 else f"{int(val)}"
        svg.append(f'<text x="{PLOT_X-10}" y="{y_pos+4}" class="tick-label">{lbl}</text>')

    # Bars
    for i, item in enumerate(data):
        group_center_x = PLOT_X + i * group_width + group_width/2
        start_x = group_center_x - available_width/2
        
        for j, (key, label, color) in enumerate(metrics):
            val = item[key]
            bar_h = (val - y_min) / (y_max - y_min) * PLOT_H
            if bar_h < 0: bar_h = 0
            
            x = start_x + j * bar_width
            y = scale_y(val)
            
            # Bar
            svg.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" fill="{color}" stroke="{COLORS["bar_border"]}" stroke-width="1"/>')
            
            # Value Label
            svg.append(f'<text x="{x + bar_width/2}" y="{y - 5}" class="bar-val">{val:.4f}</text>')

        # X Label (Model Name)
        svg.append(f'<text x="{group_center_x}" y="{PLOT_Y + PLOT_H + 20}" class="x-tick-label">{item["name"]}</text>')

    # Axis Line (Bottom)
    svg.append(f'<line x1="{PLOT_X}" y1="{PLOT_Y+PLOT_H}" x2="{PLOT_X+PLOT_W}" y2="{PLOT_Y+PLOT_H}" class="axis"/>')
    
    # Y Axis Label
    svg.append(f'<text x="{PLOT_X - 60}" y="{PLOT_Y + PLOT_H/2}" class="label" transform="rotate(-90, {PLOT_X - 60}, {PLOT_Y + PLOT_H/2})">{y_label}</text>')
    
    # Legend
    legend_x = PLOT_X + PLOT_W + 20
    legend_y = PLOT_Y + 20
    for k, (key, label, color) in enumerate(metrics):
        ly = legend_y + k * 30
        svg.append(f'<rect x="{legend_x}" y="{ly}" width="15" height="15" fill="{color}" stroke="{COLORS["bar_border"]}" stroke-width="1"/>')
        svg.append(f'<text x="{legend_x+25}" y="{ly+12}" class="legend-text">{label}</text>')

    svg.append('</svg>')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Saved {output_file}")

def generate_plots():
    data = load_data(INPUT_FILE)
    if not data: return
    
    # 1. Acc & F1 (Zoomed Y for visibility)
    # Range 0.98 to 1.00
    create_grouped_bar_chart(
        OUT_ACC_F1,
        data,
        [('val_acc', 'Val Acc', COLORS['acc']), ('val_f1', 'Val F1', COLORS['f1'])],
        "Model Performance Comparison (Accuracy & F1)",
        "Score",
        y_range=(0.98, 1.005) # Zoomed to show differences
    )
    
    # 2. Loss (Full range)
    create_grouped_bar_chart(
        OUT_LOSS,
        data,
        [('val_loss', 'Val Loss', COLORS['val_loss']), ('train_loss', 'Train Loss', COLORS['train_loss'])],
        "Model Loss Comparison (Train vs Val)",
        "Loss"
    )

if __name__ == '__main__':
    generate_plots()
