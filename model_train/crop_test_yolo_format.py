import cv2
import numpy as np

def yolo_to_pixel_coords(bbox, img_width, img_height):
    """
    Convert YOLO normalized bbox to pixel coordinates.
    bbox: [x_center, y_center, width, height] (normalized)
    """
    x_center, y_center, w, h = bbox
    x_center *= img_width
    y_center *= img_height
    w *= img_width
    h *= img_height

    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)

    return max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)

def crop_objects_from_image(image_path, label_path):
    """
    Loads an image and YOLO-style label file, then crops objects.
    Returns a list of cropped object images.
    """
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]

    crops = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            bbox = parts[1:5]
            x1, y1, x2, y2 = yolo_to_pixel_coords(bbox, img_width, img_height)
            crop = image[y1:y2, x1:x2]
            crops.append((class_id, crop))

    return crops

# Example usage
image_path = "0b01ef90-candy_26.jpg"
label_path = "0b01ef90-candy_26.txt"
cropped_images = crop_objects_from_image(image_path, label_path)

# Optional: save the cropped objects
for idx, (class_id, crop) in enumerate(cropped_images):
    cv2.imwrite(f"crop_{idx}_class_{class_id}.jpg", crop)
