from cv2ThreadModule import CameraThread
from threading import Event
import cv2
import fpsBasicsModule

width = 640
height = 480

kill_event = Event()
cam = CameraThread(kill_event, height = height, width = width)
cam.start()
detector = fpsBasicsModule.FPS()
detector.begin()

i = 0

while True:
    for i in range(10000):
        if i == 9999:
            break
    i = 0

    img = cam.read()
    detector.update()
    detector.stop()
    numfps = float(format(detector.fps()))
    cv2.putText(img, f"Average number:{int(numfps)}", (30,50), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 3)

    cv2.imshow('webcam', img)
    if cv2.waitKey(1) == 27:#27按键Esc
        break
kill_event.set()
cv2.destroyAllWindows()
