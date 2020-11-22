import dlib
import numpy as np
import imageio
import matplotlib.pyplot as plt

predictor_path = 'shape_predictor_81_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

frame = imageio.imread("./final_product/monaLisa.jpg")
dets = detector(frame, 1)
faces = []
print(faces)
for d in dets:
    shape = predictor(frame, d)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    faces.append(landmarks)


implot = plt.imshow(frame)

for face in faces:
    plt.scatter(x=face[:,0], y=face[:,1], s=5)

plt.show()