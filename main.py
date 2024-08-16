from matplotlib import pyplot as plt
import cv2

image = cv2.imread("nima.jpg")
model= cv2.CascadeClassifier("model.xml")

face=model.detectMultiScale(image)


x=face[0][0]
y=face[0][1]
a=face[0][2]
b=face[0][3]

image= cv2.rectangle(image ,(x,y),(x+a,y+b) ,(0,255,0),4)
image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
plt.imshow(image)
