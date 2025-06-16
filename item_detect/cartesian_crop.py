import cv2
import cv2.aruco as aruco
import numpy as np


def findAruco(img, draw=False):
    markerSize=5
    totalMarkers=250
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.getPredefinedDictionary(key)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict, parameters)
    corners, ids, rejected = detector.detectMarkers(img)

    if ids is not None:
        # print("Detected IDs:", ids.flatten())
        if draw:
        	aruco.drawDetectedMarkers(img, corners, ids)
    return corners


def cropImgCalc(pts):
    global theta, rel_length_marker

    pz = pts[0]
    po = pts[1]
    pt = pts[2]
    pth = pts[3]

    ltrl = lambda d: (rel_length_marker * d) / 5   # length to relative length

    A = [pz[0] - ltrl(22), pz[1] - ltrl(4)]
    B = [po[0] + ltrl(4), po[1] - ltrl(4)]
    C = [pt[0] + ltrl(4), pt[1] + ltrl(11)]
    D = [pth[0] - ltrl(22), pth[1] + ltrl(11)]

    return [A, B, C, D]


def rotate_image(image):
    global theta, center
    (h, w) = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D(center, np.degrees(theta), scale=1.0)
    rotated = cv2.warpAffine(image, rot_mat, (w, h))
    return rotated

def transform_point(point):
    global theta, center
    a,b = center
    x,y = point
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Translate point to origin
    x_shifted = x - a
    y_shifted = y - b

    # Rotate
    x_new = x_shifted * cos_theta + y_shifted * sin_theta + a
    y_new = -x_shifted * sin_theta + y_shifted * cos_theta + b

    return [x_new, y_new]


def main():
    global theta, rel_length_marker, center

    img = cv2.imread("test_items_2/4.jpg")
    corners = findAruco(img)

    # Calculate Relative Length of Marker and Angle with respect to X axis
    mZero = corners[0][0][0]
    mOne = corners[0][0][1]
    mTwo = corners[0][0][2]
    mThree = corners[0][0][3]

    dx = mZero[0] - mOne[0]
    dy = mZero[1] - mOne[1]
    rel_length_marker = ((dx**2 + dy**2)**0.5)
    theta = np.arctan2(mZero[1] - mOne[1], mZero[0] - mOne[0])
    center = ( (mZero[0]+mTwo[0])/2, (mZero[1]+mTwo[1])/2 )
    theta -= 3.141592654

    img = rotate_image(img)

    nmZero = transform_point(mZero)
    nmOne = transform_point(mOne)
    nmTwo = transform_point(mTwo)
    nmThree = transform_point(mThree)

    pts = cropImgCalc([nmZero, nmOne, nmTwo, nmThree])
    
    pts = np.array([pts[0], pts[1], pts[2], pts[3]])
    x, y, w, h = cv2.boundingRect(pts)
    cropped = img[y:y+h, x:x+w]

    cv2.imshow("rotated", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    theta = 3.1415
    rel_length_marker = 1
    center = (0,0)
    main()