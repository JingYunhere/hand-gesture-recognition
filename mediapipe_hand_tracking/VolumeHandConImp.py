#####
#程序改写未完成。待完成工作：将HandTrackingModule库和while中涉及cv2的程序全放到cv2ThreadModule
#####
import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
#fps标准模组
import fpsBasicsModule
#cv2多线程模组
from threading import Event
import cv2ThreadModule as cv2t
#音量获取
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#fps开始计时
camera = fpsBasicsModule.FPS()
camera.begin()

width, height = 640, 480
# cap = cv2.VideoCapture(0)
# cap.set(3, wCam)
# cap.set(4, hCam)
kill_event = Event()
cam = cv2t.CameraThread(kill_event, height = height, width = width)
cam.start()

detector = htm.handDetector(detectionCon=0.8)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]#音量下限
maxVol = volRange[1]#音量上限

while True:
    # ret, img = cap.read()
    img = cam.read()
    img = detector.findHands(img,draw=False)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x2 - x1, y2 - y1)#计算两指尖间距离
        # 指尖范围（距离摄像头20cm） 50 - 300
        # 音量范围-63 - 0
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length <50:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        if length >300:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
    camera.update()
    camera.stop()
    num = float(format(camera.fps()))
    cv2.putText(img, f"Average number:{int(num)}", (30,50), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 3)
    cv2.imshow("Img", img)
    if cv2.waitKey(1) == 27:#27按键Esc
        break
kill_event.set()
cv2.destroyAllWindows()