import dlib
import numpy as np
import imageio
import matplotlib.pyplot as plt
import sys
from face_detection import check_face
from face_features import find_features, compute_homography, warp_image
from blending import blend
import texture_transfer

if len(sys.argv) != 3:
    print("You need to include the path to the image then the artwork.")
    exit()

photo_path = sys.argv[1]
art_path = sys.argv[2]

#check if face in photo and art
if not check_face(photo_path):
    print("The photo must include a face")
    exit()

has_face = check_face(art_path)

#if face find face in both and map it
if has_face:
    photo_features = find_features(photo_path)
    art_features = find_features(art_path)

    img1 = np.asarray(imageio.imread(photo_path))
    img2 = np.asarray(imageio.imread(art_path))
    H = compute_homography(photo_features, art_features)

    warped = warp_image(img1, img2, H)

    blended = blend(warped, img2, art_features)
    plt.imshow(blended)
    plt.show()
    

#if not then do texture mapping
else:
    face = np.asarray(imageio.imread(photo_path))
    portrait = np.asarray(imageio.imread(art_path))
    final = texture_transfer.textureTransfer(face, portrait)
    plt.imshow(final)
    plt.show()