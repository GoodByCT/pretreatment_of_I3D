import os
import cv2

vide_path = "data/"
wirte_path = "data_224/"


for path in os.listdir(vide_path):
    video = cv2.VideoCapture(vide_path+path)

    fps = video.get(cv2.CAP_PROP_FPS)
    size = (224, 224)

    writer = cv2.VideoWriter(wirte_path+path, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), int(fps), size)

    i = 0

    while True:
        success, frame = video.read()
        if success:
            i += 1
            if (i >= 1 and i <= 8000):
                frame = cv2.resize(frame, size)
                writer.write(frame)

            if (i > 8000):
                print("success resize")
                break
        else:
            print('end')
            break