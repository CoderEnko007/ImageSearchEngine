from tools.searcher import Searcher
from tools.rgbhistogram import RGBHistogram
import numpy as np
import argparse
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True)
ap.add_argument("-i", "--index", required = True)
ap.add_argument("-q", "--query", required = True)
args = vars(ap.parse_args())

query = cv2.imread(args["query"])
cv2.imshow("query", query)

desc = RGBHistogram([8, 8, 8])
queryFeatures = desc.describe(query)

index = pickle.loads(open(args["index"], 'rb').read())
searcher = Searcher(index)
results = searcher.search(queryFeatures)

montageA = np.zeros((166 * 4, 400, 3), dtype = "uint8")
montageB = np.zeros((166 * 4, 400, 3), dtype = "uint8")
 
for j in range(0, 8):
	(score, imageName) = results[j]
	path = args["dataset"] + "/%s" % (imageName)
	result = cv2.imread(path)
	print("\t%d. %s : %.3f" % (j + 1, imageName, score))

	if j < 4:
		montageA[j * 166:(j + 1) * 166, :] = result
	else:
		montageB[(j - 4) * 166:((j - 4) + 1) * 166, :] = result

cv2.imshow("Results 1-4", montageA)
cv2.imshow("Results 5-8", montageB)
cv2.waitKey(0)