import csv
import math

# Configuration
WIDTH = 1800
HEIGHT = 1200
PADDING = 100
SUBPLOT_GAP = 150

# Colors (Viridis Palette Simulation for 6 items)
# 0: Purple, 1: Blue, 2: Teal, 3: Green, 4: Lime, 5: Yellow
VIRIDIS = [
    '#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725', 
    '#ebf345', '#fce725' # Extras just in case
]

INPUT_FILE = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\train\best_model_performance.csv"
OUTPUT_FILE = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\train\training_comparison_seaborn.svg"

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
                    'Val Acc': float(row['Val Acc']),
                    'Val F1': float(row['Val F1']),
                    'Val Loss': float(row['Val Loss']),
                    'Train Loss': float(row['Train Loss'])
                })
            except ValueError:
                continue
    # Sort alphabetically by name for consistent coloring? 
    # Or by performance? analyze_training.py just sorted(keys).
    data.sort(key=lambda x: x['name'])
    return data

def create_subplot(svg, x_offset, y_offset, w, h, data, metric_key, title):
    # Seaborn Style: White grid
    # Draw Background
    svg.append(f'<rect x="{x_offset}" y="{y_offset}" width="{w}" height="{h}" fill="white" />')
    
    # Calculate Range
    vals = [d[metric_key] for d in data]
    y_max = max(vals) * 1.15 # Add headroom for text
    y_min = 0 
    
    # Grid Lines (Y axis)
    n_ticks = 6
    y_step = (y_max - y_min) / (n_ticks - 1)
    
    # Draw Grid first (behind bars)
    for i in range(n_ticks):
        val = y_min + i * y_step
        y_pos = y_offset + h - (val / y_max * h)
        # Grid line
        svg.append(f'<line x1="{x_offset}" y1="{y_pos}" x2="{x_offset+w}" y2="{y_pos}" stroke="#EAEAF2" stroke-width="1.5" />') # Solid/Dashed? Seaborn uses solid usually or dashed? Code said linestyle='--'
        # Actually code said: ax.grid(axis='y', linestyle='--', alpha=0.5)
        svg.append(f'<line x1="{x_offset}" y1="{y_pos}" x2="{x_offset+w}" y2="{y_pos}" stroke="#000000" stroke-width="1" stroke-dasharray="5,5" opacity="0.3" />')
        
        # Tick Label
        lbl = f"{val:.4f}".rstrip('0').rstrip('.')
        svg.append(f'<text x="{x_offset-10}" y="{y_pos+5}" font-family="Arial" font-size="12" text-anchor="end" fill="#333">{lbl}</text>')

    # Bars
    n_bars = len(data)
    bar_width = w / n_bars * 0.7
    gap = w / n_bars * 0.3
    
    for i, item in enumerate(data):
        val = item[metric_key]
        bar_h = (val / y_max) * h
        bx = x_offset + i * (w / n_bars) + gap/2
        by = y_offset + h - bar_h
        
        color = VIRIDIS[i % len(VIRIDIS)]
        
        # Bar with border (alpha=0.8 simulation by color choice, usually just use solid)
        svg.append(f'<rect x="{bx}" y="{by}" width="{bar_width}" height="{bar_h}" fill="{color}" stroke="black" stroke-width="1.5" fill-opacity="0.8" />')
        
        # Text on top
        svg.append(f'<text x="{bx + bar_width/2}" y="{by - 10}" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="black">{val:.4f}</text>')
        
        # X Label (Rotated)
        svg.append(f'<text x="{bx + bar_width/2}" y="{y_offset + h + 25}" font-family="Arial" font-size="12" text-anchor="end" transform="rotate(-45, {bx + bar_width/2}, {y_offset + h + 25})">{item["name"]}</text>')

    # Title
    svg.append(f'<text x="{x_offset + w/2}" y="{y_offset - 20}" font-family="Arial" font-size="18" font-weight="bold" text-anchor="middle" fill="#333">{title}</text>')
    
    # Border (Spines) - Seaborn 'whitegrid' usually has left/bottom spines or full box?
    # Code says: ax.grid(axis='y', ...) usually implies white background with horizontal lines.
    # Let's add a bottom axis line
    svg.append(f'<line x1="{x_offset}" y1="{y_offset+h}" x2="{x_offset+w}" y2="{y_offset+h}" stroke="black" stroke-width="1" />')


def generate_svg():
    data = load_data(INPUT_FILE)
    if not data: return

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" style="background-color: white;">')
    
    # Layout 2x2
    # Width of each subplot
    sp_w = (WIDTH - 2*PADDING - SUBPLOT_GAP) / 2
    sp_h = (HEIGHT - 2*PADDING - SUBPLOT_GAP) / 2
    
    # 1. Val Acc (Top Left)
    create_subplot(svg, PADDING, PADDING, sp_w, sp_h, data, 'Val Acc', 'Best Validation Accuracy (Higher is Better)')
    
    # 2. Val F1 (Top Right)
    create_subplot(svg, PADDING + sp_w + SUBPLOT_GAP, PADDING, sp_w, sp_h, data, 'Val F1', 'Best Validation F1 Score (Higher is Better)')
    
    # 3. Val Loss (Bottom Left)
    create_subplot(svg, PADDING, PADDING + sp_h + SUBPLOT_GAP, sp_w, sp_h, data, 'Val Loss', 'Best Validation Loss (Lower is Better)')
    
    # 4. Train Loss (Bottom Right)
    create_subplot(svg, PADDING + sp_w + SUBPLOT_GAP, PADDING + sp_h + SUBPLOT_GAP, sp_w, sp_h, data, 'Train Loss', 'Final Training Loss (Lower is Better)')
    
    svg.append('</svg>')
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Saved {OUTPUT_FILE}")

if __name__ == '__main__':
    generate_svg()
