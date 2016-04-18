
import sys
import os
import cv2
import numpy as np

#import matplotlib.pyplot as plt	


class Detector:
	def detect(self, src):
		raise NotImplementedError("Every Detector must implement the detect method.")

class CascadedDetector(Detector):
	"""
	Uses the OpenCV cascades to perform the detection. Returns the Regions of Interest, where
	the detector assumes a face. You probably have to play around with the scaleFactor, 
	minNeighbors and minSize parameters to get good results for your use case. From my 
	personal experience, all I can say is: there's no parameter combination which *just 
	works*.	
	"""
	def __init__(self, cascade_fn="./cascades/haarcascade_frontalface_alt2.xml", scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE):
		if not os.path.exists(cascade_fn):
			raise IOError("No valid cascade found for path=%s." % cascade_fn)
		self.cascade = cv2.CascadeClassifier(cascade_fn)
		self.scaleFactor = scaleFactor
		self.minNeighbors = minNeighbors
		self.minSize = minSize
	
	def detect(self, src):
		if np.ndim(src) == 3:
			src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
		src = cv2.equalizeHist(src)
		rects = self.cascade.detectMultiScale(src, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors, minSize=self.minSize)
		if len(rects) == 0:
			return np.ndarray((0,))
		rects[:,2:] += rects[:,:2]
		return rects


def crop_out(img,x1,y1,x2,y2,b,h):
        '''
        This function is used to crop the image into desired dimmensions.
        img: image from which the rectangle is to be cropped
        x1,y1: top left vertex parameter
        x2,y2: bottom right parameter
        b,h: dimmesions of the cropped image
        '''
        xa=x1
        xb=x2
        xc=x1
        xd=x2
        ya=y1
        yb=y1
        yc=y2
        yd=y2

        pts1 = np.float32([[xa,ya],[xc,yc],[xb,yb],[xd,yd]]) 
        pts2 = np.float32([[0,0],[0,h],[b,0],[b,h]])
        persM = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,persM,(b,h))

        return dst

        


#detection begins here
img = cv2.imread('sample1.jpg')
#img = cv2.resize(img,(500,250))
img = np.array(img, dtype=np.uint8)

print img
imgOut = img.copy()
print imgOut

# set up detectors
detector = CascadedDetector(cascade_fn="D:\e-yantra\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml")
eyesDetector = CascadedDetector(scaleFactor=1.1,minNeighbors=5, minSize=(20,20), cascade_fn="D:\e-yantra\opencv\sources\data\haarcascades\haarcascade_eye.xml")
noseDetector = CascadedDetector(scaleFactor=1.1,minNeighbors=5, minSize=(20,20), cascade_fn="D:\e-yantra\opencv\sources\data\haarcascades\haarcascade_mcs_nose.xml")

# detection
for i,r in enumerate(detector.detect(img)):
        x0,y0,x1,y1 = r
        cv2.rectangle(imgOut, (x0,y0),(x1,y1),(0,255,0),1)
        face = imgOut[y0:y1,x0:x1]
        crop = crop_out(img,x0,y0,x1,y1,200,200)
        cv2.imwrite('face_out.jpg', crop)

for j,r1 in enumerate(eyesDetector.detect(img)):
        ex0,ey0,ex1,ey1 = r1
        cv2.rectangle(imgOut, (ex0,ey0),(ex1,ey1),(0,255,0),1)
        eye = crop_out(img,ex0,ey0,ex1,ey1,200,200)
        cv2.imwrite('eye.jpg',eye)


for i,r in enumerate(noseDetector.detect(img)):
        fx0,fy0,fx1,fy1 = r
        cv2.rectangle(imgOut, (fx0,fy0),(fx1,fy1),(255,0,0),1)
        face = imgOut[fy0:fy1,fx0:fx1]
        crop = crop_out(img,fx0,fy0,fx1,fy1,200,200)
        cv2.imwrite('nose.jpg', crop)

 
# display image or write to file

#cv2.imshow('crop', crop)
cv2.imshow('faces', imgOut)

#cv2.imwrite('out.jpg', imgOut) 


cv2.waitKey(0)

