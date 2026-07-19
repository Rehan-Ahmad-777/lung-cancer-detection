import os

CT_PATH = "datasets/CT_Scan"
HISTO_PATH = "datasets/Histopathology"

print("=" * 50)
print("LUNG CANCER DATASET INFORMATION")
print("=" * 50)

# Count CT Images
ct_images = 0
for root, dirs, files in os.walk(CT_PATH):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            ct_images += 1

# Count Histopathology Images
histo_images = 0
for root, dirs, files in os.walk(HISTO_PATH):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            histo_images += 1

print(f"CT Scan Images        : {ct_images}")
print(f"Histopathology Images : {histo_images}")
print("=" * 50)