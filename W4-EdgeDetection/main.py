import cv2
import numpy as np

video = cv2.VideoCapture("images/line.mp4")
roi = cv2.imread('images/roi.jpg', 0)
low_yellow = np.array([20, 140, 125])
up_yellow = np.array([30, 255, 255])

def apply_roi(img, roi):
    img = cv2.bitwise_and(img, img, mask=roi)
    return img

while True:
    ret, frame = video.read()
    if not ret:
        video = cv2.VideoCapture("images/30s.mp4")
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, low_yellow, up_yellow)
    white_mask = cv2.inRange(gray, 150, 255)

    mask = apply_roi(yellow_mask + white_mask, roi)

    edges = cv2.Canny(mask, 50, 100)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=30)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            frame = cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0))

    cv2.imshow('mask', white_mask)
    cv2.imshow('edges', edges)
    cv2.imshow('line', frame)

    key = cv2.waitKey(25)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()