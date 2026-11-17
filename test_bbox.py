from PIL import Image
import numpy as np

def find_grid_bounding_box(img_gray: Image.Image) -> tuple[int, int, int, int]:
    arr = np.array(img_gray, dtype=np.float32)
    if arr.mean() > 128:
        dark_mask = arr < 180
    else:
        dark_mask = arr > 75
        
    ys, xs = np.where(dark_mask)
    if len(xs) > 100 and len(ys) > 100:
        x0 = int(np.percentile(xs, 0.2))
        x1 = int(np.percentile(xs, 99.8))
        y0 = int(np.percentile(ys, 0.2))
        y1 = int(np.percentile(ys, 99.8))
        if (x1 - x0) > 100 and (y1 - y0) > 100:
            return (x0, y0, x1, y1)
    return (0, 0, img_gray.size[0], img_gray.size[1])

img = Image.open('temp_upload.png').convert("L")
bbox = find_grid_bounding_box(img)
print("Detected bounding box:", bbox, "from size:", img.size)
