import numpy as np
import cv2

class RGBHistogram:
	def __init__(self, bin):
		self.bin = bin
	def describe(self, image):
		hist = cv2.calcHist([image], [0, 1, 2], None, self.bin, [0, 256, 0, 256, 0, 256])
		hist = cv2.normalize(hist, hist)
		#print(type(hist))
		#flatten返回一个折叠乘一维的数组
		return hist.flatten()