import cv2
import numpy as np

face_height = 65
face_width = 65

def get_trainer_data():
	face_image = np.empty((0,face_height*face_width),'float32')
	name = []
	for i in range(1,3):
		for j in range(1,11):
			img = cv2.imread('FACES/s'+str(i)+'/'+str(j)+'.pgm',0)
			img = np.array(img,'uint8')
			#print img
			faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
			faces = faceCascade.detectMultiScale(img)

			for (x, y, w, h) in faces:
				
				face_only = img[y: y + h, x: x + w]
				face_only = cv2.resize(face_only,(face_height,face_width))
				face_only = face_only.reshape(1,face_height*face_width)
				face_image = np.append(face_image, face_only,0)

				#face_image = face_image.reshape(face_image.size,1)
				#print face_image
				
				name.append(i)
				#print name
				#face_image.append(face_only)
				#name.append(i)
				#cv2.imshow("image",img)
				#cv2.imshow("Adding faces to traning set...", img[y: y + h, x: x + w])
				#cv2.waitKey(50)
		print 'saving data_Set....',i
	return face_image,name

face_image, name = get_trainer_data()

face_image = np.array(face_image,'float32')
#print face_image
#print face_image.shape

#np.savetxt('face_image.txt', face_image)
#np.savetxt('name.txt', name)

print face_image,face_image.shape
print 
print name

cv2.waitKey(0)
cv2.destroyAllWindows()