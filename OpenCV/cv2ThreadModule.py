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
