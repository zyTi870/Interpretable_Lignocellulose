import csv
import math

# Configuration
WIDTH = 800
HEIGHT = 500
MARGIN_LEFT = 80
MARGIN_BOTTOM = 80
MARGIN_RIGHT = 150
MARGIN_TOP = 50

PLOT_X = MARGIN_LEFT
PLOT_Y = MARGIN_TOP
PLOT_W = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
PLOT_H = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

# User Palette
PALETTE = {
    'bg': '#FFFFFF', # Keep white for scientific paper
    'border': '#000000',
    'grid': '#E0E0E0',
    'blue': '#7789B7',
    'green': '#ACBF9F',
    'red': '#EB6969',
    'grey': '#C6CCDC',
    'dark_green': '#89AA7B',
    'blue_grey': '#9DACCB'
}

INPUT_FILE = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\train\best_model_performance.csv"
OUT_ACC_F1 = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\train\final_comparison_acc_f1.svg"
OUT_LOSS = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\train\final_comparison_loss.svg"

def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('ShortName', row.get('Model', 'Unknown'))
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
    return data

def create_chart(output_file, data, metrics, y_label, y_range=None):
    n_groups = len(data)
    n_bars = len(metrics)
    
    group_width = PLOT_W / n_groups
    padding = group_width * 0.25
    bar_width = (group_width - padding * 2) / n_bars
    
    # Determine Y Range
    if y_range:
        y_min, y_max = y_range
    else:
        max_val = 0
        for d in data:
            for k, _, _ in metrics:
                max_val = max(max_val, d[k])
        y_min = 0
        y_max = max_val * 1.1

    def scale_y(val):
        return PLOT_Y + PLOT_H - (val - y_min) / (y_max - y_min) * PLOT_H

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">')
    svg.append(f'<rect width="100%" height="100%" fill="{PALETTE["bg"]}"/>')
    
    # Style Block
    svg.append('''
    <style>
        .axis { stroke: black; stroke-width: 1.5; fill: none; }
        .tick { stroke: black; stroke-width: 1.5; }
        .label { font-family: Arial; font-size: 14px; font-weight: bold; text-anchor: middle; }
        .tick-label { font-family: Arial; font-size: 12px; text-anchor: end; }
        .x-tick-label { font-family: Arial; font-size: 12px; text-anchor: middle; font-weight: bold; }
        .legend { font-family: Arial; font-size: 12px; }
        .bar-val { font-family: Arial; font-size: 10px; text-anchor: middle; font-weight: bold; }
    </style>
    ''')
    
    # Bars
    for i, item in enumerate(data):
        group_center = PLOT_X + i * group_width + group_width/2
        start_x = group_center - (n_bars * bar_width)/2
        
        for j, (key, label, color) in enumerate(metrics):
            val = item[key]
            h = (val - y_min) / (y_max - y_min) * PLOT_H
            if h < 0: h = 0
            
            x = start_x + j * bar_width
            y = scale_y(val)
            
            # Bar with black border
            svg.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{h}" fill="{color}" stroke="black" stroke-width="1"/>')
            
            # Value Label
            fmt = "{:.4f}"
            svg.append(f'<text x="{x + bar_width/2}" y="{y - 5}" class="bar-val">{fmt.format(val)}</text>')
            
        # X Label
        svg.append(f'<text x="{group_center}" y="{PLOT_Y + PLOT_H + 25}" class="x-tick-label">{item["name"]}</text>')

    # Axes Box (Closed)
    svg.append(f'<rect x="{PLOT_X}" y="{PLOT_Y}" width="{PLOT_W}" height="{PLOT_H}" class="axis"/>')
    
    # Y Ticks (Inward, Left & Right)
    n_ticks = 6
    step = (y_max - y_min) / (n_ticks - 1)
    for i in range(n_ticks):
        val = y_min + i * step
        y = scale_y(val)
        # Left Tick
        svg.append(f'<line x1="{PLOT_X}" y1="{y}" x2="{PLOT_X+5}" y2="{y}" class="tick"/>')
        # Right Tick
        svg.append(f'<line x1="{PLOT_X+PLOT_W}" y1="{y}" x2="{PLOT_X+PLOT_W-5}" y2="{y}" class="tick"/>')
        
        # Label
        lbl = f"{val:.2f}" if step >= 0.01 else f"{val:.3f}"
        if abs(val) < 1e-9: lbl = "0"
        svg.append(f'<text x="{PLOT_X-8}" y="{y+4}" class="tick-label">{lbl}</text>')

    # X Ticks (Inward, Top & Bottom) - Just marks between groups? Or at group centers?
    # Usually categorical charts don't have X ticks, just labels. 
    # But for "closed box" feel, we can add ticks at group separators or centers.
    # Let's put ticks at group centers.
    for i in range(n_groups):
        cx = PLOT_X + i * group_width + group_width/2
        # Bottom
        svg.append(f'<line x1="{cx}" y1="{PLOT_Y+PLOT_H}" x2="{cx}" y2="{PLOT_Y+PLOT_H-5}" class="tick"/>')
        # Top
        svg.append(f'<line x1="{cx}" y1="{PLOT_Y}" x2="{cx}" y2="{PLOT_Y+5}" class="tick"/>')

    # Y Label
    svg.append(f'<text x="{PLOT_X - 60}" y="{PLOT_Y + PLOT_H/2}" class="label" transform="rotate(-90, {PLOT_X - 60}, {PLOT_Y + PLOT_H/2})">{y_label}</text>')
    
    # Legend
    leg_x = PLOT_X + PLOT_W + 20
    leg_y = PLOT_Y + 20
    for k, (key, label, color) in enumerate(metrics):
        ly = leg_y + k * 30
        svg.append(f'<rect x="{leg_x}" y="{ly}" width="20" height="20" fill="{color}" stroke="black" stroke-width="1"/>')
        svg.append(f'<text x="{leg_x+30}" y="{ly+15}" class="legend">{label}</text>')

    svg.append('</svg>')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Saved {output_file}")

def generate_final():
    data = load_data(INPUT_FILE)
    if not data: return

    # 1. Acc & F1 (Zoomed)
    create_chart(
        OUT_ACC_F1,
        data,
        [('val_acc', 'Val Accuracy', PALETTE['blue']), ('val_f1', 'Val F1', PALETTE['green'])],
        "Score",
        y_range=(0.98, 1.002)
    )
    
    # 2. Loss
    create_chart(
        OUT_LOSS,
        data,
        [('val_loss', 'Val Loss', PALETTE['red']), ('train_loss', 'Train Loss', PALETTE['grey'])],
        "Loss"
    )

if __name__ == '__main__':
    generate_final()
