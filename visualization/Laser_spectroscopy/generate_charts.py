import csv
import math

# Configuration
WIDTH = 600
HEIGHT = 450
MARGIN_LEFT = 80
MARGIN_BOTTOM = 60
MARGIN_RIGHT = 50
MARGIN_TOP = 40

PLOT_X = MARGIN_LEFT
PLOT_Y = MARGIN_TOP
PLOT_W = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
PLOT_H = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

COLORS = {
    'bg': '#FFFFFF',
    'axis': '#000000',
    'text': '#000000',
    '405_line': '#7789B7',      # Blue-ish
    '405_band': '#9DACCB',      # Blue-grey (Matching blue theme)
    '552_line': '#89AA7B',      # Darker Green
    '552_band': '#CBD7C3',      # Light Green-ish
    'grid': '#E0E0E0'
}

FILE_405 = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\lasergraph\405曲线.csv"
FILE_552 = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\lasergraph\552曲线.csv"
OUT_405 = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\lasergraph\spectrum_405.svg"
OUT_552 = r"C:\Users\pettertiz\Documents\trae_projects\drawfiber\lasergraph\spectrum_552.svg"

def load_data(filepath):
    data = []
    encodings = ['utf-8', 'utf-16', 'gbk']
    
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                lines = f.readlines()
                
            # Parse manually to handle the weird quoting
            # Skip first 2 lines
            if len(lines) < 3: continue
            
            for line in lines[2:]:
                line = line.strip()
                if not line: continue
                # Remove surrounding quotes if present
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]
                
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        x = float(parts[0].strip())
                        y = float(parts[1].strip())
                        data.append((x, y))
                    except ValueError:
                        continue
            
            if data: return data
            
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading {filepath} with {enc}: {e}")
            continue
            
    return data

def scale_x(val, x_min, x_max):
    if x_max == x_min: return PLOT_X
    return PLOT_X + (val - x_min) / (x_max - x_min) * PLOT_W

def scale_y(val, y_min, y_max):
    if y_max == y_min: return PLOT_Y + PLOT_H
    return PLOT_Y + PLOT_H - (val - y_min) / (y_max - y_min) * PLOT_H

def create_chart(output_file, data, title, line_color, band_range, band_color, band_label):
    if not data:
        print(f"No data for {output_file}")
        return

    x_vals = [p[0] for p in data]
    y_vals = [p[1] for p in data]
    
    x_min = min(x_vals)
    x_max = max(x_vals)
    
    # Auto Y-range with padding
    y_min_data = min(y_vals)
    y_max_data = max(y_vals)
    y_range_data = y_max_data - y_min_data
    
    # If range is very small, center it
    if y_range_data == 0:
        y_min = y_min_data - 1
        y_max = y_max_data + 1
    else:
        y_min = max(0, y_min_data - y_range_data * 0.1) # Try to start at 0 if close, else adapt
        # For 405 (values ~2.0-2.8), starting at 0 makes it look flat.
        # User complained "lower half empty". 
        # Let's adapt more aggressively if min is far from 0.
        if y_min_data > y_range_data * 2: # e.g. min=2, range=0.8. 2 > 1.6.
            y_min = y_min_data * 0.9
        else:
            y_min = 0
            
        y_max = y_max_data + y_range_data * 0.1
    
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
    </style>
    ''')
    
    # Draw Band
    if band_range:
        bx1, bx2 = band_range
        # Clamp band to view
        bx1 = max(bx1, x_min)
        bx2 = min(bx2, x_max)
        
        if bx2 > bx1:
            lx1 = scale_x(bx1, x_min, x_max)
            lx2 = scale_x(bx2, x_min, x_max)
            svg_content.append(f'<rect x="{lx1}" y="{PLOT_Y}" width="{lx2-lx1}" height="{PLOT_H}" fill="{band_color}" fill-opacity="0.4"/>')
            svg_content.append(f'<text x="{(lx1+lx2)/2}" y="{PLOT_Y + 20}" class="label" font-size="12">{band_label}</text>')

    # Draw Line
    # Clip data to x range? Already done by logic usually, but ensure d path is valid
    d_parts = []
    first = True
    for x, y in data:
        if x < x_min or x > x_max: continue
        sx = scale_x(x, x_min, x_max)
        sy = scale_y(y, y_min, y_max)
        if first:
            d_parts.append(f"M {sx} {sy}")
            first = False
        else:
            d_parts.append(f"L {sx} {sy}")
    
    if d_parts:
        svg_content.append(f'<path d="{" ".join(d_parts)}" stroke="{line_color}" stroke-width="2" fill="none"/>')
    
    # Axes Box
    svg_content.append(f'<rect x="{PLOT_X}" y="{PLOT_Y}" width="{PLOT_W}" height="{PLOT_H}" class="axis"/>')
    
    # Ticks & Grid (X)
    x_range = x_max - x_min
    x_step = 50 if x_range > 100 else 20
    
    curr_x = math.ceil(x_min / x_step) * x_step
    while curr_x <= x_max:
        sx = scale_x(curr_x, x_min, x_max)
        svg_content.append(f'<line x1="{sx}" y1="{PLOT_Y+PLOT_H}" x2="{sx}" y2="{PLOT_Y+PLOT_H-5}" class="tick"/>')
        svg_content.append(f'<text x="{sx}" y="{PLOT_Y+PLOT_H+20}" class="tick-label">{int(curr_x)}</text>')
        curr_x += x_step
        
    # Ticks & Grid (Y)
    y_range = y_max - y_min
    # Better tick logic
    # Aim for 5-6 ticks
    # Example: range 0.8 -> step 0.1 or 0.2
    # Example: range 25 -> step 5
    
    raw_step = y_range / 5
    # Find nice step
    mag = 10 ** math.floor(math.log10(raw_step))
    norm_step = raw_step / mag # 1..10
    if norm_step < 1.5: step = 1 * mag
    elif norm_step < 3.5: step = 2 * mag
    elif norm_step < 7.5: step = 5 * mag
    else: step = 10 * mag
    
    y_step = step
    
    curr_y = math.ceil(y_min / y_step) * y_step
    # Fix float precision issues for loop
    while curr_y <= y_max + y_step * 0.001:
        if curr_y < y_min - y_step * 0.001: 
            curr_y += y_step
            continue
            
        sy = scale_y(curr_y, y_min, y_max)
        svg_content.append(f'<line x1="{PLOT_X}" y1="{sy}" x2="{PLOT_X+5}" y2="{sy}" class="tick"/>')
        
        # Format label
        if y_step < 1:
            lbl = f"{curr_y:.2f}".rstrip('0').rstrip('.')
        else:
            lbl = f"{int(round(curr_y))}" if abs(curr_y - round(curr_y)) < 0.001 else f"{curr_y:.1f}"
            
        svg_content.append(f'<text x="{PLOT_X-8}" y="{sy+4}" class="tick-label" text-anchor="end">{lbl}</text>')
        curr_y += y_step

    # Axis Labels
    svg_content.append(f'<text x="{PLOT_X + PLOT_W/2}" y="{PLOT_Y + PLOT_H + 40}" class="label">λ (nm)</text>')
    svg_content.append(f'<text x="{PLOT_X - 50}" y="{PLOT_Y + PLOT_H/2}" class="label" transform="rotate(-90, {PLOT_X - 50}, {PLOT_Y + PLOT_H/2})">Mean Intensity</text>')
    
    svg_content.append('</svg>')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg_content))
    
    print(f"Saved {output_file}")

def generate_charts():
    # 405 Chart
    data_405 = load_data(FILE_405)
    create_chart(
        OUT_405, 
        data_405, 
        "405 nm Excitation", 
        COLORS['405_line'], 
        (435, 500), 
        COLORS['405_band'], 
        "Lignin"
    )
    
    # 552 Chart
    data_552 = load_data(FILE_552)
    # Filter 552 data >= 555
    data_552 = [p for p in data_552 if p[0] >= 555]
    
    create_chart(
        OUT_552, 
        data_552, 
        "552 nm Excitation", 
        COLORS['552_line'], 
        (570, 620), 
        COLORS['552_band'], 
        "Cellulose+Congo Red"
    )

if __name__ == '__main__':
    generate_charts()
