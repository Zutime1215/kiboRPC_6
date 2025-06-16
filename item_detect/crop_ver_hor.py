import cv2
import numpy as np
import math
import os

def crop_angular_slices(image_path, step_deg=10, crop_percentage=100, output_dir='slices'):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute radius as a percentage of the shortest center dimension
    radius = int(min(center) * (crop_percentage / 100.0))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Loop through angles
    for angle in range(0, 360, step_deg):
        # Create black mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # Define polygon for the angular slice (fan-shaped)
        theta1 = math.radians(angle)
        theta2 = math.radians(angle + step_deg)

        # Number of points along the arc
        arc_points = 100
        pts = [center]
        for i in range(arc_points + 1):
            theta = theta1 + i * (theta2 - theta1) / arc_points
            x = int(center[0] + radius * math.cos(theta))
            y = int(center[1] + radius * math.sin(theta))
            pts.append((x, y))

        pts = np.array([pts], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)

        # Apply mask to image
        result = cv2.bitwise_and(image, image, mask=mask)

        # Save result
        out_path = os.path.join(output_dir, f'slice_{angle:03d}.png')
        cv2.imwrite(out_path, result)

    print(f"Saved {360 // step_deg} slices in '{output_dir}' with {crop_percentage}% radius.")

# Example usage
crop_angular_slices("items/coin.png", step_deg=10, crop_percentage=70)
