import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ==========================================
# LOAD MODELS
# ==========================================

ct_model = load_model("models/ct_model.keras")
histo_model = load_model("models/histo_model.keras")

# ==========================================
# CLASS NAMES
# ==========================================

ct_classes = [
    "Benign",
    "Malignant",
    "Normal"
]

histo_classes = [
    "Adenocarcinoma",
    "Benign",
    "Squamous Cell Carcinoma"
]

# ==========================================
# SELECT MODEL
# ==========================================

print("=" * 50)
print("LUNG CANCER DETECTION SYSTEM")
print("=" * 50)

print("\nChoose Model")
print("1. CT Scan")
print("2. Histopathology")

choice = input("\nEnter Choice: ")

# ==========================================
# CT SCAN
# ==========================================

if choice == "1":

    model = ct_model
    class_names = ct_classes

    folder = "datasets/Test cases"

    images = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".png")
    ])

# ==========================================
# HISTOPATHOLOGY
# ==========================================

elif choice == "2":

    model = histo_model
    class_names = histo_classes

    print("\nSelect Histopathology Dataset")

    print("1. Adenocarcinoma")
    print("2. Benign")
    print("3. Squamous Cell Carcinoma")

    hist_choice = input("\nEnter Choice: ")

    if hist_choice == "1":
        folder = "datasets/Histopathology/adenocarcinoma"

    elif hist_choice == "2":
        folder = "datasets/Histopathology/benign"

    elif hist_choice == "3":
        folder = "datasets/Histopathology/squamous_cell_carcinoma"

    else:
        print("Invalid Choice")
        exit()

    images = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

else:
    print("Invalid Choice")
    exit()

# ==========================================
# SHOW IMAGES
# ==========================================

print("\nAvailable Images\n")

for i, img in enumerate(images):
    print(f"{i+1}. {img}")

index = int(input("\nChoose Image Number: ")) - 1

image_path = os.path.join(folder, images[index])

# ==========================================
# IMAGE PREPROCESSING
# ==========================================

img = image.load_img(image_path, target_size=(224,224))

img_array = image.img_to_array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(img_array, verbose=0)

prediction = prediction[0]

predicted_class = np.argmax(prediction)

confidence = prediction[predicted_class] * 100

# ==========================================
# RESULTS
# ==========================================

print("\n" + "=" * 50)
print("Prediction Result")
print("=" * 50)

print(f"Image           : {images[index]}")
print(f"Prediction      : {class_names[predicted_class]}")
print(f"Confidence      : {confidence:.2f}%")

print("\nProbability of Each Class")
print("-" * 40)

for i, score in enumerate(prediction):
    print(f"{class_names[i]:25} : {score*100:.2f}%")

print("=" * 50)

# ==========================================
# DISPLAY IMAGE
# ==========================================

plt.imshow(img)
plt.axis("off")
plt.title(class_names[predicted_class])

plt.show()