from collections import deque
import cv2 as cv


class CvFpsCalc(object):
    #属性定义，属性是类的数据成员
    def __init__(self, buffer_len=1):
        self._start_tick = cv.getTickCount()    #保存了开始计时的时钟周期数（tick count
        self._freq = 1000.0 / cv.getTickFrequency() #保存了时钟周期的频率，用于将时钟周期转换为毫秒
        self._difftimes = deque(maxlen=buffer_len)  #是一个双向队列，用于存储最近一段时间内的帧间隔时间。

    #方法定义，方法是类的函数成员
    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded
