import dlib
import numpy as np
import imageio
import matplotlib.pyplot as plt


def find_features(path):
    predictor_path = 'shape_predictor_81_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    frame = imageio.imread(path)
    dets = detector(frame, 1)
    faces = []
    for d in dets:
        shape = predictor(frame, d)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        faces.append(landmarks)

    # can be changed to have more faces if we want
    return faces[0]

def compute_homography(t1, t2):
    H = np.eye(3)
    z = np.ones((t1.shape[0], 1))
    new_t1 = np.hstack((t1, z))
    zeros = np.zeros_like(new_t1)
    first = np.stack([new_t1.T, zeros.T],axis=-1).reshape(3,-1).T
    second = np.stack([zeros.T, new_t1.T],axis=-1).reshape(3,-1).T
    fs = np.append(first, second, axis=1)
    
    p_u = -1 * new_t1 * t2[:, 0][:, np.newaxis]
    p_v = -1 * new_t1 * t2[:, 1][:, np.newaxis]
    third = np.stack([p_u.T, p_v.T],axis=-1).reshape(3,-1).T
    L = np.append(fs, third, axis=1)
    LTL = np.matmul(L.T, L)
    w, v = np.linalg.eig(LTL)
    H = v.T[np.where(w == np.amin(w))].reshape(3, 3)
    
    return H

def warp_image(input_image, ref_image, H, top=True):
    merge_image = None
    
    r, c, _ = input_image.shape
    corners = np.array([[0, 0, 1], [c - 1, 0, 1], [0, r - 1, 1], [c - 1, r - 1, 1]])
    new_corners = []
    for corner in corners:
        scaled = np.matmul(H, corner)
        scaled = scaled[0:2] / scaled[2]
        new_corners.append(scaled.tolist())
    result_corners = np.array(new_corners)
    
    offset = np.array([int(result_corners[:,0].min()), int(result_corners[:,1].min()), 0])
    c_off = offset[0] * -1 if offset[0] < 0 else 0
    r_off = offset[1] * -1 if offset[1] < 0 else 0
    
    c_pos_off = offset[0] if offset[0] > 0 else 0
    r_pos_off = offset[1] if offset[1] > 0 else 0
    
    r_length = int(result_corners[:,1].max()) - int(result_corners[:,1].min())
    c_length = int(result_corners[:,0].max()) - int(result_corners[:,0].min())
    if not top:
        warped = np.zeros((r_length, c_length, 3))
    else:
        warped = np.ones((r_length, c_length, 3)) * -1
    
    r_merge_length = max(int(result_corners[:,1].max()), ref_image.shape[0]) + r_off
    c_merge_length = max(int(result_corners[:,0].max()), ref_image.shape[1]) + c_off
    if not top:
        merged = np.zeros((r_merge_length, c_merge_length, 3))
    else:
        merged = np.ones((r_merge_length, c_merge_length, 3)) * -1
    
    inv_H = np.linalg.inv(H)
    xx, yy = np.meshgrid(np.arange(warped.shape[1]), np.arange(warped.shape[0]))
    warp_pairs = np.dstack([xx, yy]).reshape(-1, 2)
    
    z = np.ones((warp_pairs.shape[0], 1))
    warp_pairs_3 = np.hstack((warp_pairs, z))
    
    for pair in warp_pairs_3:
        moved_pair = pair + offset
        scaled = np.matmul(inv_H, moved_pair)
        scaled = scaled[0:2] / scaled[2]
        if scaled[0] >= 0 and scaled[0] < c and scaled[1] >= 0 and scaled[1] < r:
            warped[int(pair[1]), int(pair[0])] = input_image[int(scaled[1]), int(scaled[0])]
            merged[int(pair[1] + r_pos_off), int(pair[0] + c_pos_off)] = input_image[int(scaled[1]), int(scaled[0])]
    # warp_image = warped.astype(np.uint8)
    
    merge_image = merged.astype(np.uint8)
    if not top:
        merge_image[r_off:(r_off + ref_image.shape[0]), c_off:(c_off + ref_image.shape[1]), :] = ref_image
    else:
        temp = np.copy(ref_image)
        layer = merged[:,:, 0]
        filled = np.where(layer >= 0)
        temp[filled[0], filled[1], :] = merged[filled[0], filled[1], :]
        merge_image = temp
    return merge_image