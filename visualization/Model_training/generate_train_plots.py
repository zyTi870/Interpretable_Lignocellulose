import csv
import math

# Configuration
WIDTH = 800
HEIGHT = 500
MARGIN_LEFT = 80
MARGIN_BOTTOM = 100 # More space for model names
MARGIN_RIGHT = 150
MARGIN_TOP = 50

PLOT_X = MARGIN_LEFT
PLOT_Y = MARGIN_TOP
PLOT_W = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
PLOT_H = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

COLORS = {
    'bg': '#FFFFFF',
    'axis': '#000000',
    'text': '#000000',
    'acc': '#7789B7',      # Blue-ish
    'f1': '#ACBF9F',       # Green-ish
    'loss_val': '#EB6969', # Red-ish
    'loss_train': '#C6CCDC',# Grey-blue
    'grid': '#E0E0E0'
}

INPUT_FILE = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\train\best_model_performance.csv"
OUT_ACC_F1 = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\train\model_comparison_acc_f1.svg"
OUT_LOSS = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\train\model_comparison_loss.svg"

def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean up model names
            name = row.get('ShortName', row.get('Model', 'Unknown'))
            # Simplify names if needed
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

def create_bar_chart(output_file, data, metrics, title, y_label):
    # metrics: list of (key, label, color)
    
    n_groups = len(data)
    n_bars = len(metrics)
    
    # Bar width calculations
    group_width = PLOT_W / n_groups
    bar_padding = group_width * 0.1
    bar_width = (group_width - 2 * bar_padding) / n_bars
    
    # Determine Y range
    max_val = 0
    for d in data:
        for key, _, _ in metrics:
            max_val = max(max_val, d[key])
    
    y_min = 0
    y_max = max_val * 1.1
    
    def scale_y(val):
        return PLOT_Y + PLOT_H - (val - y_min) / (y_max - y_min) * PLOT_H

    svg_content = []
    svg_content.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">')
    svg_content.append(f'<rect width="100%" height="100%" fill="{COLORS["bg"]}"/>')
    
    # Style
    svg_content.append('''
    <style>
        .axis { stroke: black; stroke-width: 1.5; fill: none; }
        .tick { stroke: black; stroke-width: 1; }
        .grid { stroke: #E0E0E0; stroke-width: 1; stroke-dasharray: 4; }
        .label { font-family: Arial; font-size: 14px; font-weight: bold; text-anchor: middle; }
        .tick-label { font-family: Arial; font-size: 12px; text-anchor: middle; }
        .legend { font-family: Arial; font-size: 12px; }
        .bar-label { font-family: Arial; font-size: 10px; text-anchor: middle; }
    </style>
    ''')
    
    # Draw Bars
    for i, item in enumerate(data):
        group_x = PLOT_X + i * group_width + bar_padding
        
        for j, (key, label, color) in enumerate(metrics):
            val = item[key]
            bar_h = (val / (y_max - y_min)) * PLOT_H
            x = group_x + j * bar_width
            y = PLOT_Y + PLOT_H - bar_h
            
            svg_content.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" fill="{color}"/>')
            # Value label on top
            # svg_content.append(f'<text x="{x + bar_width/2}" y="{y - 5}" class="bar-label">{val:.3f}</text>')

    # Axes Box
    svg_content.append(f'<rect x="{PLOT_X}" y="{PLOT_Y}" width="{PLOT_W}" height="{PLOT_H}" class="axis"/>')
    
    # Y Ticks & Grid
    # Use simple linear steps
    y_step = max_val / 5
    # Nice step logic
    mag = 10 ** math.floor(math.log10(y_step)) if y_step > 0 else 0.1
    norm = y_step / mag
    if norm < 1.5: step = 1 * mag
    elif norm < 3.5: step = 2 * mag
    elif norm < 7.5: step = 5 * mag
    else: step = 10 * mag
    
    curr_y = 0
    while curr_y <= y_max:
        sy = scale_y(curr_y)
        svg_content.append(f'<line x1="{PLOT_X}" y1="{sy}" x2="{PLOT_X+5}" y2="{sy}" class="tick"/>')
        # Grid line? Optional.
        # svg_content.append(f'<line x1="{PLOT_X}" y1="{sy}" x2="{PLOT_X+PLOT_W}" y2="{sy}" class="grid"/>')
        
        lbl = f"{curr_y:.2f}".rstrip('0').rstrip('.') if step < 1 else f"{int(curr_y)}"
        svg_content.append(f'<text x="{PLOT_X-8}" y="{sy+4}" class="tick-label" text-anchor="end">{lbl}</text>')
        curr_y += step

    # X Labels (Model Names)
    for i, item in enumerate(data):
        cx = PLOT_X + i * group_width + group_width/2
        # Rotate labels if too many
        svg_content.append(f'<text x="{cx}" y="{PLOT_Y + PLOT_H + 20}" class="tick-label" transform="rotate(0, {cx}, {PLOT_Y + PLOT_H + 20})">{item["name"]}</text>')

    # Axis Titles
    svg_content.append(f'<text x="{PLOT_X - 50}" y="{PLOT_Y + PLOT_H/2}" class="label" transform="rotate(-90, {PLOT_X - 50}, {PLOT_Y + PLOT_H/2})">{y_label}</text>')
    
    # Legend
    legend_x = PLOT_X + PLOT_W + 20
    legend_y = PLOT_Y + 50
    
    for k, (key, label, color) in enumerate(metrics):
        ly = legend_y + k * 30
        svg_content.append(f'<rect x="{legend_x}" y="{ly}" width="20" height="20" fill="{color}"/>')
        svg_content.append(f'<text x="{legend_x+30}" y="{ly+15}" class="legend">{label}</text>')

    svg_content.append('</svg>')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg_content))
    print(f"Saved {output_file}")

def generate_plots():
    data = load_data(INPUT_FILE)
    if not data:
        print("No data found")
        return

    # Plot 1: Acc & F1
    # Zoom in Y axis for Acc/F1 since they are close to 1.0?
    # User said "scientific grade". Usually bar charts start at 0.
    # But if all are 0.98+, 0-1 range hides differences.
    # Let's check ranges. All ~0.98.
    # For bar charts, cutting the Y-axis is often discouraged but sometimes necessary.
    # Let's stick to 0-1 for now unless it looks bad. Or we can auto-scale min too.
    # Actually for 0.98 vs 0.99, a 0.9-1.0 range is better.
    # Let's modify create_bar_chart to support min_y if needed, but standard bars start at 0.
    # I'll stick to 0-base for bars to be "correct", but maybe add values on top?
    # Or I can make a dot plot? The user asked for "like above", which was line charts.
    # But for categorical data, bars are standard.
    
    # Wait, the user said "like above". Above was line charts for spectra.
    # But here we have model names. Bars are best.
    
    create_bar_chart(
        OUT_ACC_F1, 
        data, 
        [('val_acc', 'Validation Accuracy', COLORS['acc']), ('val_f1', 'Validation F1-Score', COLORS['f1'])],
        "Model Performance",
        "Score"
    )
    
    create_bar_chart(
        OUT_LOSS, 
        data, 
        [('val_loss', 'Validation Loss', COLORS['loss_val']), ('train_loss', 'Training Loss', COLORS['loss_train'])],
        "Model Loss",
        "Loss"
    )

if __name__ == '__main__':
    generate_plots()
