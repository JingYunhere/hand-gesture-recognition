from threading import Thread, Lock
from datetime import datetime
import time
import cv2

time_cycle = 80

class CameraThread(Thread):
    def __init__(self, kill_event, src = 0, width = 320, height = 240):
        self.kill_event = kill_event
        
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        (self.ret, self.img) = self.cap.read()
        self.read_lock = Lock()#初始化锁

        Thread.__init__(self, args = kill_event)

    def update(self):
        (ret, img) = self.cap.read()
        self.read_lock.acquire()#激活锁
        self.ret, self.img = ret, img
        self.read_lock.release()#释放锁

    def read(self):
        self.read_lock.acquire()
        img = self.img.copy()
        self.read_lock.release()
        return img

    def run(self):
        while not self.kill_event.is_set():
            start_time = datetime.now()
            self.update()

            finish_time = datetime.now()
            dt = finish_time - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            if ms < time_cycle:
                time.sleep((time_cycle - ms) / 1000.0)

    def form(self):
        self.read_lock.acquire()
        imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)#将opencv2读到的BGR图片转换成RGB给mediapipe用
        self.read_lock.release()
        return imgRGB

    def drawing(self, length, img, x1, y1, x2, y2):
        self.read_lock.acquire()
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        if length <50:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        if length >300:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
        self.read_lock.release()
        return img