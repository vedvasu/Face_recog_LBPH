import cv2, sys
import numpy as np
i=1
# Get user supplied values
for j in range(1,536):
	imgPath = 'photos/test_cases2/sample ('+str(j)+').jpg'
	print j
	cascPath = "haarcascade_frontalface_default.xml"

	# Create the haar cascade
	faceCascade = cv2.CascadeClassifier(cascPath)

	# Read the image
	img = cv2.imread(imgPath)
	#img  = np.array(img)
	#img = np.transpose(img)
	RP = img.shape[0] / 800 if img.shape[0] <= img.shape[1] else img.shape[1] / 800
	imgResized = cv2.resize (img, (img.shape[1] / RP, img.shape[0] / RP))
	gray = cv2.cvtColor(imgResized, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
	    gray,
	    scaleFactor = 1.1,
	    minNeighbors = 5,
	    minSize = (30, 30),
	    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	print "Found {0} faces!".format(len(faces))

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		x1 = x
		y1 = y
		x2 = x+w
		y2 = y+h
		face_only = imgResized[y1:y2, x1:x2]
		cv2.imshow('face',face_only)
		cv2.waitKey(500)
		#cv2.rectangle(imgResized, (x1, y1), (x2, y2), (0, 0, 255), 2)
		cv2.imwrite('faces/face'+str(i)+'.jpg',face_only)
		i+=1
# Face Recognizer
recognizer = cv2.createLBPHFaceRecognizer()


cv2.imshow("Faces found", imgResized)
cv2.waitKey(0)
# print img.shape, imgResized.shape