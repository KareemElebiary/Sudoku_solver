from PIL import Image
import numpy as np

img = Image.open('temp_upload.png')
print("Image format:", img.format, "size:", img.size, "mode:", img.mode)
