import cv2
import mediapipe as mp

#读取摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)#镜头编号0，编号1不能打开摄像头

while True:
    ret,img = cap.read()#cap.read会回传两个数据
    if ret:
        cv2.imshow('img',img)

    if cv2.waitKey(20) ==ord('q'):
        break