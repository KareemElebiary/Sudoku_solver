from PIL import Image
import numpy as np

def detect_grid_bounds(img_path):
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    
    # Invert if white background
    if arr.mean() > 128:
        arr = 255.0 - arr
        
    # High energy rows/cols (lines and digits)
    row_energy = arr.sum(axis=1)
    col_energy = arr.sum(axis=0)
    
    # Threshold at 15% of max
    r_thresh = row_energy.max() * 0.15
    c_thresh = col_energy.max() * 0.15
    
    valid_rows = np.where(row_energy > r_thresh)[0]
    valid_cols = np.where(col_energy > c_thresh)[0]
    
    if len(valid_rows) > 0 and len(valid_cols) > 0:
        y0, y1 = valid_rows[0], valid_rows[-1]
        x0, x1 = valid_cols[0], valid_cols[-1]
        print(f"Detected grid box: x=[{x0}, {x1}], y=[{y0}, {y1}] from image size {img.size}")
        return (x0, y0, x1, y1)
    return (0, 0, img.size[0], img.size[1])

detect_grid_bounds('temp_upload.png')
