import cv2
import os

source_folder = '../others_datasets/zz_Train/train2_kibo-pcwgo_grayscale/images'
destination_folder = '../others_datasets/zz_Train/train2_kibo-pcwgo_grayscale/img_ch'

th = 165

os.makedirs(destination_folder, exist_ok=True)

# Loop over all image files in source folder
for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg')):
        img_path = os.path.join(source_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"❌ Skipping {filename}, couldn't read.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV)

        save_path = os.path.join(destination_folder, filename)
        cv2.imwrite(save_path, gray)
        print(f"✅ Saved {filename} to output.")