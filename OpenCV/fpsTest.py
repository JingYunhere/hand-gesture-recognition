import cv2
import time

#读取摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)#镜头编号0，自带摄像头。编号1自定义摄像头
pTime = 0
cTime = 0
time_list = []

while True:
    start = time.time()
    ret,img = cap.read()#cap.read会回传两个数据
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#将opencv2读到的BGR图片转换成RGB给mediapipe用
    imgHidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
    imgHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img, f"FPS:{int(fps)}", (30,50), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 3)

    cv2.imshow("image", img)
    cv2.waitKey(1)
    end = time.time()
    time_list.append(end-start)
    avarge = sum(time_list)/len(time_list)
    print('time',avarge)