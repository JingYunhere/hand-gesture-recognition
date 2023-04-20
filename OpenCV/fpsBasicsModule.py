import datetime
import cv2

class FPS:
	def __init__(self):
		# 存储开始时间、结束时间和在开始和结束间隔之间检查的帧总数
		self.start = 0
		self.end = 0
		self.numFrames = 0

	def begin(self):
		# 开始计时器
		self.start = datetime.datetime.now()

	def stop(self):
		# 停止计时器
		self.end = datetime.datetime.now()

	def update(self):
		# 增加在开始和结束间隔期间的总帧数
		self.numFrames += 1

	def elapsed(self):
		# 返回开始和结束间隔之间的总秒数
		return (self.end - self.start).total_seconds()

	def fps(self):
		# 平均1秒执行几次
		return self.numFrames / self.elapsed()

