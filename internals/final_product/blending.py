import numpy as np
import cv2

def set_mask_area(mask, coords, val, blur_size):
    cv2.fillConvexPoly(mask, coords, val)
    gauss_mask = mask.astype(np.float)
    gauss_mask = cv2.GaussianBlur(gauss_mask, (blur_size, blur_size), 0)

    return gauss_mask

def create_mask(img, landmarks):
    mask = np.full((img.shape[0], img.shape[1]), 0.0)
    face = np.concatenate((landmarks[0:17], landmarks[17:27][::-1]), axis=0)
    mask = set_mask_area(mask, face, 0.6, 25)
    teeth = landmarks[60:68]
    mask = set_mask_area(mask, teeth, 0.1, 3)
    left_eye = landmarks[36:42]
    mask = set_mask_area(mask, left_eye, 0.80, 1)
    right_eye = landmarks[42:48]
    mask = set_mask_area(mask, right_eye, 0.80, 5)
    forehead = landmarks[[16, 78,74,79,73,72,80,71,70,69,68,76,75,77]]
    mask = set_mask_area(mask, forehead, 0.6, 5)
    return mask

def blend(src, dst, landmarks):
    mask = create_mask(dst, landmarks)
    height1, width1 = src.shape[:2]
    height2, width2 = dst.shape[:2]
    result = dst
    for y in range(height2):
        for x in range(width2):
            if y < height1 and x < width1:
                result[y, x] = src[y, x] * mask[y, x] + result[y, x] * (1 - mask[y, x])

    return result