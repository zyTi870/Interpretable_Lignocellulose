import csv
import math

# Configuration
WIDTH = 800
HEIGHT = 500
MARGIN_LEFT = 150 # More space for model names on the left
MARGIN_BOTTOM = 60
MARGIN_RIGHT = 50
MARGIN_TOP = 60

PLOT_X = MARGIN_LEFT
PLOT_Y = MARGIN_TOP
PLOT_W = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
PLOT_H = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

COLORS = {
    'bg': '#FFFFFF',
    'axis': '#000000',
    'text': '#000000',
    'train': '#ACBF9F',      # Light Green (Train)
    'val': '#EB6969',        # Red (Val)
    'connector': '#999999',
    'grid': '#E0E0E0'
}

INPUT_FILE = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\train\training_summary.csv"
OUT_LOSS = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\train\scientific_loss_dumbbell.svg"
OUT_ACC = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\train\scientific_acc_dot.svg"

def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['Model']
            # Clean names
            if 'densenet121_cbam' in name: name = 'DenseNet121-CBAM'
            elif 'densenet121' in name: name = 'DenseNet121'
            elif 'resnet18_cbam' in name: name = 'ResNet18-CBAM'
            elif 'resnet18' in name: name = 'ResNet18'
            elif 'resnet50' in name: name = 'ResNet50'
            elif 'vit' in name: name = 'ViT'
            
            try:
                data.append({
                    'name': name,
                    'train_loss': float(row['train_loss']),
                    'val_loss': float(row['val_loss']),
                    'train_acc': float(row['train_acc']),
                    'val_acc': float(row['val_acc'])
                })
            except ValueError:
                continue
    # Sort by Val Loss (best on top for chart, so lowest val loss first? 
    # Usually top-down means 0 at top. Let's sort by performance.)
    # For chart Y-axis: item 0 at top.
    # Let's sort so best model is at the top.
    # Best model = Lowest Val Loss.
    data.sort(key=lambda x: x['val_loss']) 
    return data

def create_dumbbell_chart(output_file, data, key1, label1, color1, key2, label2, color2, title, x_label):
    # Horizontal Chart
    # Y-axis: Models
    # X-axis: Metric
    
    n_items = len(data)
    row_height = PLOT_H / n_items
    
    # Determine X range
    vals = [d[key1] for d in data] + [d[key2] for d in data]
    x_min = 0
    x_max = max(vals) * 1.1
    
    def scale_x(val):
        return PLOT_X + (val - x_min) / (x_max - x_min) * PLOT_W
        
    def scale_y(idx):
        # Center of row
        return PLOT_Y + idx * row_height + row_height/2

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
        .tick-label { font-family: Arial; font-size: 12px; text-anchor: end; }
        .model-label { font-family: Arial; font-size: 12px; font-weight: bold; text-anchor: end; }
        .legend { font-family: Arial; font-size: 12px; }
        .val-text { font-family: Arial; font-size: 10px; text-anchor: middle; }
    </style>
    ''')
    
    # Title
    svg_content.append(f'<text x="{WIDTH/2}" y="30" class="label" font-size="16">{title}</text>')
    
    # Grid (Vertical lines for X axis)
    x_step = x_max / 5
    # Normalize step
    mag = 10 ** math.floor(math.log10(x_step)) if x_step > 0 else 0.1
    if x_step/mag < 2: step = 1*mag
    elif x_step/mag < 5: step = 2*mag
    else: step = 5*mag
    
    curr_x = 0
    while curr_x <= x_max:
        sx = scale_x(curr_x)
        svg_content.append(f'<line x1="{sx}" y1="{PLOT_Y}" x2="{sx}" y2="{PLOT_Y+PLOT_H}" class="grid"/>')
        # X-axis labels at bottom
        val_str = f"{curr_x:.4f}".rstrip('0').rstrip('.') if step < 1 else f"{int(curr_x)}"
        svg_content.append(f'<text x="{sx}" y="{PLOT_Y+PLOT_H+20}" class="tick-label" text-anchor="middle">{val_str}</text>')
        curr_x += step

    # Draw Data
    for i, item in enumerate(data):
        y = scale_y(i)
        x1 = scale_x(item[key1])
        x2 = scale_x(item[key2])
        
        # Connector line
        svg_content.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{COLORS["connector"]}" stroke-width="2"/>')
        
        # Point 1 (Train)
        svg_content.append(f'<circle cx="{x1}" cy="{y}" r="6" fill="{color1}"/>')
        
        # Point 2 (Val)
        svg_content.append(f'<circle cx="{x2}" cy="{y}" r="6" fill="{color2}"/>')
        
        # Model Label
        svg_content.append(f'<text x="{PLOT_X-10}" y="{y+4}" class="model-label">{item["name"]}</text>')

    # Legend
    svg_content.append(f'<circle cx="{PLOT_X}" cy="{PLOT_Y-15}" r="6" fill="{color1}"/>')
    svg_content.append(f'<text x="{PLOT_X+10}" y="{PLOT_Y-11}" class="legend" text-anchor="start">{label1}</text>')
    
    svg_content.append(f'<circle cx="{PLOT_X+100}" cy="{PLOT_Y-15}" r="6" fill="{color2}"/>')
    svg_content.append(f'<text x="{PLOT_X+110}" y="{PLOT_Y-11}" class="legend" text-anchor="start">{label2}</text>')
    
    # X Axis Label
    svg_content.append(f'<text x="{PLOT_X + PLOT_W/2}" y="{PLOT_Y + PLOT_H + 40}" class="label">{x_label}</text>')
    
    svg_content.append('</svg>')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg_content))
    print(f"Saved {output_file}")

def create_dot_plot_zoom(output_file, data, key, label, color, title, x_label):
    # Horizontal Dot Plot
    n_items = len(data)
    row_height = PLOT_H / n_items
    
    vals = [d[key] for d in data]
    x_min = min(vals) * 0.99 # Zoom in a bit? Or 0.98?
    # Actually for 0.984 vs 0.994, range 0.98 to 1.0 is good.
    x_min = 0.98
    x_max = 1.002
    
    def scale_x(val):
        return PLOT_X + (val - x_min) / (x_max - x_min) * PLOT_W
        
    def scale_y(idx):
        return PLOT_Y + idx * row_height + row_height/2

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
        .model-label { font-family: Arial; font-size: 12px; font-weight: bold; text-anchor: end; }
        .val-text { font-family: Arial; font-size: 11px; font-weight: bold; text-anchor: start; }
    </style>
    ''')
    
    # Title
    svg_content.append(f'<text x="{WIDTH/2}" y="30" class="label" font-size="16">{title}</text>')
    
    # Grid
    step = 0.005
    curr_x = x_min
    while curr_x <= x_max:
        sx = scale_x(curr_x)
        if sx > PLOT_X + PLOT_W: break
        svg_content.append(f'<line x1="{sx}" y1="{PLOT_Y}" x2="{sx}" y2="{PLOT_Y+PLOT_H}" class="grid"/>')
        val_str = f"{curr_x:.3f}"
        svg_content.append(f'<text x="{sx}" y="{PLOT_Y+PLOT_H+20}" class="tick-label">{val_str}</text>')
        curr_x += step

    # Draw Data
    for i, item in enumerate(data):
        y = scale_y(i)
        x = scale_x(item[key])
        
        # Dashed line from name to dot
        svg_content.append(f'<line x1="{PLOT_X}" y1="{y}" x2="{x}" y2="{y}" stroke="{COLORS["grid"]}" stroke-width="1" stroke-dasharray="2"/>')
        
        # Dot
        svg_content.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{color}"/>')
        
        # Text Value next to dot
        svg_content.append(f'<text x="{x+12}" y="{y+4}" class="val-text">{item[key]:.4f}</text>')
        
        # Model Label
        svg_content.append(f'<text x="{PLOT_X-10}" y="{y+4}" class="model-label">{item["name"]}</text>')

    # X Axis Label
    svg_content.append(f'<text x="{PLOT_X + PLOT_W/2}" y="{PLOT_Y + PLOT_H + 40}" class="label">{x_label}</text>')
    
    svg_content.append('</svg>')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg_content))
    print(f"Saved {output_file}")

def generate_scientific_plots():
    data = load_data(INPUT_FILE)
    if not data: return

    # 1. Dumbbell Plot for Loss (Train vs Val)
    # This shows overfitting clearly (Train << Val)
    create_dumbbell_chart(
        OUT_LOSS,
        data,
        'train_loss', 'Train Loss', COLORS['train'],
        'val_loss', 'Val Loss', COLORS['val'],
        "Model Loss Comparison (Generalization Gap)",
        "Loss (Lower is Better)"
    )
    
    # 2. Dot Plot for Accuracy (Zoomed)
    # Since values are close (0.984 vs 0.994), we zoom in to show difference
    # And we label the exact values.
    # Sort data by accuracy for this one?
    data_acc = sorted(data, key=lambda x: x['val_acc'], reverse=True)
    create_dot_plot_zoom(
        OUT_ACC,
        data_acc,
        'val_acc', 'Validation Accuracy', '#7789B7',
        "Model Validation Accuracy",
        "Accuracy"
    )

if __name__ == '__main__':
    generate_scientific_plots()
