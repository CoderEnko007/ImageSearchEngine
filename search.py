from tools.searcher import Searcher
import numpy as np
import argparse
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True)
ap.add_argument("-i", "--index", required = True)
args = vars(ap.parse_args())

index = pickle.loads(open(args["index"], 'rb').read())
#print(index.items())
searcher = Searcher(index)

for (query, queryFeatures) in index.items():
	results = searcher.search(queryFeatures)
	path = args["dataset"] + "\\%s" % (query)
	#print(path)
	queryImage = cv2.imread(path)
	cv2.imshow("Query", queryImage)
	print("query: %s" % (query))

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