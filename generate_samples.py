import os
import numpy as np
from PIL import Image, ImageFilter

os.makedirs("samples", exist_ok=True)

# Generate 3 anatomically suggestive sample CXRs for immediate UI testing
for cls in ["Normal", "Pneumonia", "Tuberculosis"]:
    img = np.zeros((224, 224), dtype=np.uint8) + 25
    # Rib cage and lungs
    for y, rad in [(80, 42), (110, 44), (140, 40)]:
        rr = int(rad)
        for dx in range(-rr, rr):
            for dy in range(-rr, rr):
                if dx*dx + dy*dy <= rr*rr:
                    img[min(223, max(0, y + dy)), min(223, max(0, 70 + dx))] = 160
                    img[min(223, max(0, y + dy)), min(223, max(0, 154 + dx))] = 160
                    
    # Cardiac silhouette
    for dx in range(-25, 25):
        for dy in range(-35, 35):
            if (dx/25)**2 + (dy/35)**2 <= 1.0:
                img[min(223, max(0, 135 + dy)), min(223, max(0, 115 + dx))] = 50

    if cls == "Pneumonia":
        for dx in range(-22, 22):
            for dy in range(-20, 20):
                if (dx/22)**2 + (dy/20)**2 <= 1.0:
                    img[min(223, max(0, 140 + dy)), min(223, max(0, 155 + dx))] = 220
    elif cls == "Tuberculosis":
        for dx in range(-16, 16):
            for dy in range(-16, 16):
                dist = dx*dx + dy*dy
                if dist <= 16*16:
                    img[min(223, max(0, 75 + dy)), min(223, max(0, 68 + dx))] = 210
                if dist <= 7*7:
                    img[min(223, max(0, 75 + dy)), min(223, max(0, 68 + dx))] = 40

    pil_img = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=3))
    pil_img.save(f"samples/sample_{cls.lower()}.png")

print("Created sample images in samples/")
