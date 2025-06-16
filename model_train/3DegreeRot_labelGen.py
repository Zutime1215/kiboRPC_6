import cv2
import numpy as np
import os


def rotate_image_3d(img, pitch=0, yaw=0, roll=0, fov=60, background_color=(255, 255, 255)):
    h, w = img.shape[:2]
    f = w / (2 * np.tan(np.deg2rad(fov / 2)))

    # Convert angles to radians
    pitch_rad = np.deg2rad(pitch)
    yaw_rad = np.deg2rad(yaw)
    roll_rad = np.deg2rad(roll)

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad),  np.cos(pitch_rad)]
    ])

    Ry = np.array([
        [ np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])

    Rz = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad),  np.cos(roll_rad), 0],
        [0, 0, 1]
    ])

    # Combined rotation: Roll → Pitch → Yaw
    R = Rz @ Rx @ Ry

    # 3D corners centered
    corners = np.array([
        [-w/2, -h/2, 0],
        [ w/2, -h/2, 0],
        [ w/2,  h/2, 0],
        [-w/2,  h/2, 0]
    ])

    def project(points3D):
        rotated = points3D @ R.T + np.array([0, 0, f])
        projected = rotated[:, :2] / rotated[:, 2, np.newaxis]
        return projected * f + np.array([w / 2, h / 2])

    dst_pts = project(corners).astype(np.float32)
    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=background_color)
    return warped


def main():
    image_dir = 'items'
    cls_lst = ['coin', 'compass', 'coral', 'crystal', 'diamond', 'emerald', 'fossil', 'key', 'letter', 'shell', 'treasure_box']
    output_dir = "out_img"
    label_dir = "out_lbl"

    step_angle = 10
    step_angle_roll = 30
    x_center, y_center = 0.5, 0.5
    width, height = 1.0, 1.0

    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(('.jpg', '.png')):
            it = img_file.split('.')[0]
            class_id = cls_lst.index(it)
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)

            # === Process rotations Yaw Pitch ===
            for angle in range(step_angle, 45 + step_angle, step_angle):
                for k, sign in {'m': -1, 'p': 1}.items():
                    a = sign * angle
                    anm = k + '_' + str(angle)

                    # Pitch
                    pitched = rotate_image_3d(img, pitch=a)
                    nm = f'{it}_pt_{anm}'; print(nm)
                    cv2.imwrite(os.path.join(output_dir, nm + '.png'), pitched)
                    label_path = os.path.join(label_dir, (nm + '.txt'))
                    with open(label_path, 'w') as f:
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}")

                    # Yaw
                    yawed = rotate_image_3d(img, yaw=a)
                    nm = f'{it}_yw_{anm}'; print(nm)
                    cv2.imwrite(os.path.join(output_dir, nm + '.png'), yawed)
                    label_path = os.path.join(label_dir, (nm + '.txt'))
                    with open(label_path, 'w') as f:
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}")

            # === Process rotations Roll ===
            for angle in range(step_angle_roll, 360, step_angle_roll):
                a = angle
                anm = str(angle)

                # Roll
                yawed = rotate_image_3d(img, roll=a)
                nm = f'{it}_rl_{anm}'; print(nm)
                cv2.imwrite(os.path.join(output_dir, nm + '.png'), yawed)
                label_path = os.path.join(label_dir, (nm + '.txt'))
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}")

img_path = 'test2.png'
img = cv2.imread(img_path)
yawed = rotate_image_3d(img, roll=50)
cv2.imwrite('test_rot.png', yawed)