import cv2

# Load the original image
img = cv2.imread("test_items_2/2.jpg")
# img = cv2.imread("test_items/1.png")
# img = cv2.imread("items/coin.png")

scale_percent = 1  # adjust as needed
width = int(img.shape[1] * scale_percent)
height = int(img.shape[0] * scale_percent)
resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

min_wh = 30
max_wh = 90
th = 200

# Convert to grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Threshold to binary (invert to make objects white on black)
_, thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cropped_objects = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    
    if w > min_wh and h > min_wh and w < max_wh and h < max_wh:
        cv2.rectangle(resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cropped = gray[y:y+h, x:x+w]
        cropped_objects.append(cropped)

print(f"Total detected objects: {len(cropped_objects)}")

# Show result
# cv2.imshow("Detected Objects", resized)
cv2.imshow("thresh", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()