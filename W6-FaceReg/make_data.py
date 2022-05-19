import cv2
from facenet_pytorch import MTCNN
import torch
import os
from utils import automatic_brightness_and_contrast

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

IMG_PATH = './data/test_images/'
count = 50
usr_name = input("Input ur name: ")
USR_PATH = os.path.join(IMG_PATH, usr_name)
leap = 1

mtcnn = MTCNN(keep_all=False, post_process=False, device = device) # Lấy được face
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
while cap.isOpened() and count:
    isSuccess, frame = cap.read()
    if mtcnn(frame) is not None and leap%2:
        path = str(USR_PATH + '/' + str(count) + '.jpg')
        frame, _, _ = automatic_brightness_and_contrast(frame)
        face_img = mtcnn(frame, save_path = path)
        count-=1
    leap+=1
    cv2.imshow('Face Capturing', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
