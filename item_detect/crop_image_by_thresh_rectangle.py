import cv2
import numpy as np
import os

def crop_brightest_rectangle(image_path):
    # Load and preprocess
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate bright areas (adjust threshold if needed)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest rectangular-like contour
    best_rect = None
    max_area = 0
    for contour in contours:
        # Approximate contour to a polygon
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):  # Likely a rectangle
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                best_rect = approx

    if best_rect is not None:
        x, y, w, h = cv2.boundingRect(best_rect)
        cropped = image[y:y+h, x:x+w]
        return cropped
    else:
        print("No bright rectangular region found.")
        return None

def getAllImg(folder_path, extensions={'.jpg', '.png'}):
    image_paths = []

    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in extensions:
            full_path = os.path.join(folder_path, filename)
            image_paths.append(full_path)

    return image_paths

# Example usage
if __name__ == "__main__":
    folder_path = "../../item_detect/test_items_2/"
    for img_path in getAllImg(folder_path):
        cropped = crop_brightest_rectangle(img_path)
        if cropped is not None:
            cv2.imshow("Cropped", cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
