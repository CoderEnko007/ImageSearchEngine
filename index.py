from tools.rgbhistogram import RGBHistogram
import numpy as np
import argparse
import pickle
import glob
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True)
ap.add_argument("-i", "--index", required = True)
args = vars(ap.parse_args())

index = {}

desc = RGBHistogram([8, 8, 8])
for imagePath in glob.glob(args["dataset"]+"\\*.png"):
	k = imagePath[imagePath.rfind("\\") + 1:]
	image = cv2.imread(imagePath)
	feature = desc.describe(image)
	index[k] = feature
	#print(k)
	#print(type(index[k]))

#print(type(index))
f = open(args["index"], "wb")
f.write(pickle.dumps(index))
f.close()