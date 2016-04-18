import cv2
import numpy as np

face_height = 65
face_width = 65
face_image = np.load('face_image_self.npy')
name = np.load('name_self.npy')

#print face_image, face_image.shape
#print
#print name, name.shape

recogniser = cv2.createLBPHFaceRecognizer()
recogniser.train(face_image, np.array(name))

def check_for_this_orientation(img):
	faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	faces = faceCascade.detectMultiScale(img,scaleFactor = 1.1, minNeighbors = 5,minSize = (30, 30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
	flag = 0 
	for (x, y, w, h) in faces:
		flag+=1
		face_only = img[y: y + h, x: x + w]
		face_only = cv2.resize(face_only,(face_height,face_width))
		#cv2.imshow('face',face_only)
		#cv2.waitKey(50)
		nbr_predicted, conf = recogniser.predict(face_only)
		cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
		if (nbr_predicted == 1):
			cv2.putText(image, 'Abheet', (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
			print 'prediction = Abheet  with confidence = ',conf
		elif (nbr_predicted == 2):
			cv2.putText(image, 'Abhiuday', (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
			print 'prediction = Abhiuday with confidence = ',conf
		elif (nbr_predicted == 3):
			cv2.putText(image, 'Deepesh', (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
			print 'prediction = Deepesh with confidence = ',conf
		elif (nbr_predicted == 4):
			cv2.putText(image, 'Diksha', (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
			print 'prediction = Diksha with confidence = ',conf
		elif (nbr_predicted == 5):
			cv2.putText(image, 'Harshit', (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 3)
			print 'prediction = Harshit with confidence = ',conf
		elif (nbr_predicted == 6):
			cv2.putText(image, 'Karan', (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)
			print 'prediction = Karan with confidence = ',conf
		elif (nbr_predicted == 7):
			cv2.putText(image, 'Prerit', (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,255,127), 3)
			print 'prediction = Prerit with confidence = ',conf
		elif (nbr_predicted == 8):
			cv2.putText(image, 'Ved', (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,150,255), 3)
			print 'prediction = Ved with confidence = ',conf
			
	cv2.imshow('Original_Image',image)

	cv2.waitKey(0)
	return flag

for i in range(1,8):
	
	for j in range(0,1):
		
		#img = cv2.imread('photos/test_cases1/sample ('+str(i)+').jpg')
		img = cv2.imread('photos/test_cases3/sample ('+str(i)+').jpg')


		rows,cols,channels = img.shape
		#img = cv2.warpAffine(img,cv2.getRotationMatrix2D((cols/2,rows/2),270*j,1),(cols,rows))
		image = img.copy()

		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		RP = img.shape[0] / 500 if img.shape[0] <= img.shape[1] else img.shape[1] / 800
		img = cv2.resize (img, (img.shape[1] / RP, img.shape[0] / RP))

		image = cv2.resize (image, (image.shape[1] / RP, image.shape[0] / RP))

		img = np.array(img,'uint8')

		flag = check_for_this_orientation(img)
		print 'for sample ',i
		if flag == 0:
			continue
		else:
			break
	cv2.destroyAllWindows()