import cv2
import numpy as np

face_height = 65
face_width = 65


def get_trainer_data():
	face_image = []
	name = []
	for i in range(1,3):
		for j in range(1,11):
			img = cv2.imread('FACES_self_dataset/s'+str(i)+'/'+str(j)+'.jpg',0)
			img = cv2.resize(img,(500,250))
			img = np.array(img,'uint8')
			img = np.transpose(img)
			#cv2.imshow('image',img)
			#cv2.waitKey(50)
			#print img
			faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
			faces = faceCascade.detectMultiScale(img)

			for (x, y, w, h) in faces:
				print x,y,w,h
				face_only = img[y: y + h, x: x + w]
				face_only = cv2.resize(face_only,(face_height,face_width))
				face_image.append(face_only)

				#face_image = face_image.reshape(face_image.size,1)
				#print face_image
				
				name.append(i)
				
				#print name
				#face_image.append(face_only)
				#name.append(i)
				cv2.imshow("image",img)
				cv2.imshow("Adding faces to traning set...", img[y: y + h, x: x + w])
				cv2.waitKey(50)
		print 'saving data_Set....',i
	return face_image,name

def get_trainer_data_without_facedetect():
	face_image = []
	name = []
	for i in range(1,9):
		for j in range(1,11):

			face_only = cv2.imread('FACES_self_dataset/s'+str(i)+'/1 ('+str(j)+').jpg',0)
			print face_only.shape
					
			face_only = cv2.resize(face_only,(face_height,face_width))
			face_image.append(face_only)
			name.append(i)
			
			cv2.imshow("image",face_only)
			cv2.waitKey(500)
		
		print 'saving data_Set....',i
	return face_image,name


#face_image, name = get_trainer_data()
face_image, name = get_trainer_data_without_facedetect()
#face_image = np.array(face_image,'float32')
#print face_image
#print face_image.shape

np.save('face_image_self.npy', face_image)
np.save('name_self.npy', name)

print face_image#,face_image.shape
print 
print name

recogniser = cv2.createLBPHFaceRecognizer()
recogniser.train(face_image, np.array(name))

cv2.waitKey(0)
cv2.destroyAllWindows()